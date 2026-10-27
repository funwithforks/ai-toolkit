"""Ragged batches: one batch, per-sample spatial shapes.

A MOSS training batch is normally a stacked (B, C, H, W) tensor, which forces
every sample in the batch to share a spatial shape -- i.e. aspect-ratio
buckets. A RaggedBatch holds the same batch as a list of per-sample
(C, h_i, w_i) tensors so shapes can differ within one batch; the DiT packs
the variable-length token runs into a single stream (see
extensions_built_in/diffusion_models/moss_image/src/transformer.py
_forward_packed_ragged).

The API is deliberately narrow. It supports the handful of operations the
training path actually performs (device/dtype moves, elementwise +,- against
another ragged batch or a scalar/per-sample weight, mapping a per-item
function, and the batch-major doubling the CFG path uses). Anything else
raises instead of guessing: there is no ``.shape``, no ``.dim()``, no
reduction, and no ``__torch_function__`` magic, so shared code that still
assumes a stacked rectangle fails immediately and visibly at the real
assumption rather than broadcasting a wrong shape silently.
"""
from typing import Callable, List, Optional, Sequence

import torch


class RaggedBatch:
    """Per-sample tensors (each ``(C, h_i, w_i)``) treated as one batch.

    Items stay separate on purpose: they are never concatenated, so no
    padding rows are ever created. Order is batch-major and load-bearing --
    the packed stream's ``cu``/row maps and the LoRA multiplier rows are
    indexed by position, and the CFG doubling must append the second half
    after the first (``double_batch_major``), never interleave.
    """

    __slots__ = ("items",)

    def __init__(self, items: Sequence[torch.Tensor]):
        if not isinstance(items, (list, tuple)) or len(items) == 0:
            raise ValueError(
                "RaggedBatch needs a non-empty list of per-sample tensors")
        items = list(items)
        for i, it in enumerate(items):
            if not torch.is_tensor(it):
                raise ValueError(
                    f"RaggedBatch item {i} is {type(it).__name__}, expected "
                    "a tensor (C, h, w)")
            if it.ndim != 3:
                raise ValueError(
                    f"RaggedBatch item {i} has shape {tuple(it.shape)}; "
                    "expected a single sample (C, h, w) -- a batch dimension "
                    "here means the caller stacked something by mistake")
        channels = items[0].shape[0]
        for i, it in enumerate(items[1:], start=1):
            if it.shape[0] != channels:
                raise ValueError(
                    f"RaggedBatch item {i} has {it.shape[0]} channels, "
                    f"item 0 has {channels}; channel count must match within "
                    "a batch")
        self.items = items

    # ------------------------------------------------------------- structure
    @property
    def batch_size(self) -> int:
        return len(self.items)

    @property
    def channels(self) -> int:
        return self.items[0].shape[0]

    @property
    def shapes(self) -> List:
        return [(int(t.shape[-2]), int(t.shape[-1])) for t in self.items]

    @property
    def token_counts(self) -> List[int]:
        return [h * w for h, w in self.shapes]

    @property
    def device(self):
        return self.items[0].device

    @property
    def dtype(self):
        return self.items[0].dtype

    @property
    def is_uniform(self) -> bool:
        """True when every sample shares a shape (a ragged batch that could
        have been stacked -- what the parity test runs through both paths)."""
        s = self.shapes
        return all(x == s[0] for x in s)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __repr__(self) -> str:
        return (f"RaggedBatch(n={len(self.items)}, C={self.channels}, "
                f"shapes={self.shapes}, dtype={self.dtype}, "
                f"device={self.device})")

    # ---------------------------------------------------------------- moves
    def to(self, device=None, dtype=None) -> "RaggedBatch":
        return RaggedBatch([
            t.to(device=device, dtype=dtype) if (device is not None or dtype is not None)
            else t
            for t in self.items
        ])

    def detach(self) -> "RaggedBatch":
        return RaggedBatch([t.detach() for t in self.items])

    def clone(self) -> "RaggedBatch":
        return RaggedBatch([t.clone() for t in self.items])

    def float(self) -> "RaggedBatch":
        return RaggedBatch([t.float() for t in self.items])

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "RaggedBatch":
        """Apply fn per sample. The escape hatch for per-item ops (noise
        draws, the scheduler) that are already per-sample upstream."""
        return RaggedBatch([fn(t) for t in self.items])

    def double_batch_major(self) -> "RaggedBatch":
        """CFG-style doubling: the second copy follows the first, matching
        ``torch.cat([x] * 2, dim=0)`` on a stacked batch. Batch-major order is
        an invariant the packed row maps depend on."""
        return RaggedBatch(self.items + list(self.items))

    # ---------------------------------------------------------- arithmetic
    def _aligned(self, other: "RaggedBatch", op: str) -> "RaggedBatch":
        if not isinstance(other, RaggedBatch):
            raise TypeError(
                f"ragged {op}: cannot combine RaggedBatch with "
                f"{type(other).__name__}. A stacked tensor and a ragged batch "
                "in the same expression means one branch of the code missed "
                "the ragged case -- fix it, do not broadcast.")
        if len(other) != len(self):
            raise ValueError(
                f"ragged {op}: batch size {len(other)} != {len(self)}")
        for i, (a, b) in enumerate(zip(self.items, other.items)):
            if a.shape != b.shape:
                raise ValueError(
                    f"ragged {op}: item {i} shape {tuple(a.shape)} != "
                    f"{tuple(b.shape)} -- per-sample shapes must line up")
        return other

    def _binary(self, other, op: str, fn) -> "RaggedBatch":
        if isinstance(other, RaggedBatch):
            self._aligned(other, op)
            return RaggedBatch([fn(a, b) for a, b in zip(self.items, other.items)])
        if isinstance(other, torch.Tensor):
            if other.numel() == len(self):
                # per-sample weight: (B,) broadcast over the item's own elems
                return RaggedBatch(
                    [fn(a, other[i].to(a.device, a.dtype))
                     for i, a in enumerate(self.items)])
            if other.numel() == 1:
                o = other.to(self.device, self.dtype)
                return RaggedBatch([fn(a, o) for a in self.items])
            raise ValueError(
                f"ragged {op}: tensor with {other.numel()} elements cannot "
                "broadcast onto a ragged batch (needs 1 or the batch size)")
        if isinstance(other, (int, float)):
            return RaggedBatch([fn(a, other) for a in self.items])
        return NotImplemented

    def __add__(self, other):
        return self._binary(other, "add", lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, "sub", lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary(other, "rsub", lambda a, b: b - a)

    def __mul__(self, other):
        return self._binary(other, "mul", lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, "div", lambda a, b: a / b)

    def __neg__(self):
        return RaggedBatch([-a for a in self.items])

    # ------------------------------------------------------------- guards
    @property
    def shape(self):
        # Deliberately unavailable: a ragged batch HAS no single shape. Any
        # code reaching for it assumed the rectangle and must branch.
        raise AttributeError(
            "a ragged batch has no single .shape (per-sample shapes are "
            f"{self.shapes}); branch on isinstance(x, RaggedBatch)")

    def dim(self, *a, **k):
        raise AttributeError("a ragged batch has no .dim(); branch on "
                             "isinstance(x, RaggedBatch)")

    def numel(self, *a, **k):
        raise AttributeError("a ragged batch has no .numel(); branch on "
                             "isinstance(x, RaggedBatch)")


def is_ragged(x) -> bool:
    return isinstance(x, RaggedBatch)


def as_ragged(x) -> RaggedBatch:
    """Wrap a stacked (B, C, H, W) tensor as a ragged batch (split on the
    batch dim). Ragged input passes through."""
    if isinstance(x, RaggedBatch):
        return x
    if not torch.is_tensor(x) or x.ndim != 4:
        raise ValueError(
            f"cannot build a ragged batch from {type(x).__name__} "
            f"ndim={getattr(x, 'ndim', '?')} (expected a 4D stacked tensor)")
    return RaggedBatch([x[i] for i in range(x.shape[0])])
