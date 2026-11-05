"""Weight placement primitive.

End goal (agreed): one universal reader with explicit placement control —
on disk (lazy), cpu, or gpu — and a queryable location, replacing the
per-model load paths during the pipeline redesign. Today only the
materialize side exists; every new consumer goes through it instead of
growing another bespoke loader.

    src = WeightSource.open(path)        # lazy: on disk, nothing read yet
    src.get(key, device)                 # fault one tensor where wanted
    src.materialize(device) -> dict      # whole file; peak ram = largest
                                         # tensor (gpu) or the whole file
                                         # (cpu -- opt-in allow_cpu=True)

Location is queryable: tensors answer via .device; a lazy source reports
' disk'. Dtype is never cast; storage is exactly as shipped.
"""

import json
import os
import struct

import torch

from toolkit.paths import MODELS_PATH

# safetensors header dtype strings -> torch dtypes (the seek-read fallback
# parses these straight from the header)
_SFT_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _is_unmappable(e: Exception) -> bool:
    """True for the mmap refusals the seek-read fallback can stand in for
    (heuristic overcommit refuses single mappings larger than ram+swap --
    a 46 GB checkpoint on a 32 GB box -- and similar allocation failures).
    Anything else propagates: a missing file is not this fallback's job."""
    text = str(e)
    return "mmap" in text or "Cannot allocate memory" in text


class _SeekReader:
    """Per-key seek reads for checkpoints too big to mmap. Same tensors as
    the mmap path -- the same bytes from the same offsets, one at a time;
    the page cache does the rest. Reads to cpu and moves when a gpu device
    was requested (the mmap path faults directly; same result)."""

    def __init__(self, file_path: str, device, owner: "WeightSource"):
        self._file_path = file_path
        self._device = device
        self._owner = owner
        self._file = None

    def _ensure_open(self):
        if self._file is None:
            self._file = open(self._file_path, "rb")

    def keys(self) -> list:
        return list(self._header().keys())

    def _header(self) -> dict:
        return self._owner._cached_header()

    def get_tensor(self, key: str) -> torch.Tensor:
        self._ensure_open()
        meta = self._header()[key]
        dtype = _SFT_DTYPES[meta["dtype"]]
        start, end = meta["data_offsets"]
        self._file.seek(start)
        raw = self._file.read(end - start)
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(
            meta["shape"]
        )
        if self._device is not None and torch.device(self._device).type != "cpu":
            tensor = tensor.to(self._device)
        return tensor

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


class WeightSource:
    """A safetensors checkpoint exposed with explicit device placement."""

    def __init__(self, file_path: str, handle=None):
        self.file_path = file_path
        self._handle = handle

    # -- construction ---------------------------------------------------

    @classmethod
    def resolve(cls, ref: str, category: str, token=None) -> "WeightSource":
        """Resolve a weight reference to a local file: an existing local
        path, the settings models path (local_dir layout, then bare
        filename), or the hub -- downloaded into MODELS_PATH/<category>/,
        never the system hf cache. That download policy is the contract for
        every weight fetch in the toolkit: 'settings value if set, system
        cache only when it is not'."""
        if os.path.isfile(ref):
            return cls(ref)
        splits = ref.split("/")
        if len(splits) < 3:
            raise ValueError(
                f"Invalid weight reference: {ref!r}. Must be a local file or "
                "'org/repo/filename.safetensors'."
            )
        rel_path = "/".join(splits[2:])
        filename = splits[-1]
        for candidate in (
            os.path.join(MODELS_PATH, rel_path),
            os.path.join(MODELS_PATH, filename),
        ):
            if os.path.isfile(candidate):
                return cls(candidate)
        import huggingface_hub

        path = huggingface_hub.hf_hub_download(
            repo_id="/".join(splits[:2]),
            filename=rel_path,
            token=token,
            local_dir=os.path.join(MODELS_PATH, category),
        )
        return cls(path)

    @classmethod
    def open(cls, file_path: str) -> "WeightSource":
        """Lazy: validate the file, read nothing. Tensors stay on disk
        until get()/materialize() asks for them."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)
        return cls(file_path)

    # -- inspection -----------------------------------------------------

    @property
    def location(self) -> str:
        """'disk' while lazy; the materialized device is carried by the
        returned tensors themselves."""
        return "disk" if self._handle is None else "materialized"

    def keys(self) -> list:
        with self._reader() as f:
            return list(f.keys())

    # -- reading --------------------------------------------------------

    def get(self, key: str, device) -> torch.Tensor:
        """Fault a single tensor onto ``device`` (weights land there
        directly; nothing cpu-sized is ever allocated for it)."""
        with self._reader(device) as f:
            return f.get_tensor(key)

    def materialize(self, device, *, allow_cpu: bool = False) -> dict:
        """Read the whole file with every tensor on ``device``.

        Memory behavior by destination:

        - gpu: each tensor is read to host ram and moved to the device
          immediately, one at a time, so peak host ram is the largest
          single tensor in the file, never the file's total size. (The
          reader's own device path would stage every tensor through
          host ram at model size; it is not used here.)
        - cpu: the entire state dict is built in host ram -- peak ram
          is the file's total size, which is what ooms at transformer
          sizes. Opt-in via ``allow_cpu=True`` for consumers that need
          the whole cpu dict in one piece (requantization, low_vram
          parking).

        Dtype is never cast; tensors are exactly as shipped.
        """
        dev = self._canon(device)
        if dev.type == "cpu" and not allow_cpu:
            raise ValueError(
                f"materialize('cpu') on {os.path.basename(self.file_path)} "
                "builds the entire state dict in host ram (peak rss = the "
                "file's total size). Pass allow_cpu=True if this consumer "
                "needs the whole cpu dict in one piece (requantization, "
                "low_vram parking); otherwise stream to a gpu device or "
                "fault tensors per-key with get()."
            )
        if dev.type == "cpu":
            with self._reader(dev) as f:
                keys = f.keys()
                out = {}
                for i, k in enumerate(keys):
                    out[k] = f.get_tensor(k)
                    self._probe_rss(i, len(keys))
                return out
        out = {}
        with self._reader("cpu") as f:
            keys = f.keys()
            total = len(keys)
            for i, k in enumerate(keys):
                out[k] = f.get_tensor(k).to(dev)
                self._probe_rss(i, total)
        return out

    def _probe_rss(self, i: int, total: int) -> None:
        """Live ram probe every 200 tensors: the rss trajectory names the
        culprit if an oom kills a load mid-stream (per-tensor paths cap at
        largest-tensor ram; a whole-dict path ramps to model size)."""
        if i % 200:
            return
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                print(
                    f"[load-timing] {os.path.basename(self.file_path)}: "
                    f"streamed {i}/{total}, rss {int(line.split()[1]) / 1024:.0f}MB"
                )
                break

    # -- internals ------------------------------------------------------

    @staticmethod
    def _canon(device) -> torch.device:
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            # the rust reader wants an indexed device string ('cuda:0')
            dev = torch.device("cuda", torch.cuda.current_device())
        return dev

    def _cached_header(self) -> dict:
        """Parsed safetensors header, memoized (the seek-read fallback asks
        for it per tensor; parsing 1 MB of json 6000 times is silly)."""
        if getattr(self, "_header_cache", None) is None:
            with open(self.file_path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            self._header_cache = header
        return self._header_cache

    def _reader(self, device=None):
        from safetensors import safe_open

        if device is None:
            device = "cpu"  # keys()/inspection do not need a gpu
        try:
            return safe_open(self.file_path, framework="pt", device=str(self._canon(device)))
        except (RuntimeError, OSError) as e:
            if not _is_unmappable(e):
                raise
            if not getattr(self, "_warned_unmappable", False):
                self._warned_unmappable = True
                print(
                    f"[weight-source] {os.path.basename(self.file_path)}: mmap "
                    f"refused ({e}); reading per-key with seek reads instead"
                )
            return _SeekReader(self.file_path, self._canon(device), self)


def load_file_to_device(file_path: str, device) -> dict:
    """Back-compat shim: WeightSource.open(path).materialize(device) --
    same memory behavior, including the cpu opt-in."""
    return WeightSource.open(file_path).materialize(device)
