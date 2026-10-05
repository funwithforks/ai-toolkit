"""Batch composition accounting: wraps packing.build_packed_sequence to
record the exact per-sequence row breakdown (text/cond/audio/video) and
batches them per training step, so dataset-vs-dataset batch-size
differences are read, not inferred. Counters only; flush on atexit.
"""

import atexit
from collections import defaultdict

_out = None
_done = {"patched": False, "step": 0}
_step_rows = defaultdict(int)          # breakdown key -> rows this step
_step_calls = defaultdict(int)
_step_batches = defaultdict(int)
_hist = []


def _flush():
    if _out is None:
        return
    data = {
        "probe": "batchshapes",
        "steps_done": _done["step"],
        "batches": _hist,
        "counters": {k: v for k, v in _step_rows.items()},
    }
    tmp = _out + ".tmp"
    with open(tmp, "w") as f:
        import json
        json.dump(data, f, indent=1)
    import os
    os.replace(tmp, _out)


def INSTALL(artifacts_path):
    global _out
    _out = artifacts_path
    atexit.register(_flush)


def WRAP(hook):
    def wrapped(self, batch):
        _patch_once()
        _step_rows.clear()
        _step_calls.clear()
        _step_batches.clear()
        out = hook(self, batch)
        _done["step"] += 1
        _hist.append({
            "step": _done["step"],
            "rows": dict(_step_rows),
            "calls": dict(_step_calls),
        })
        return out
    return wrapped


def _patch_once():
    if _done["patched"]:
        return
    _done["patched"] = True
    from extensions_built_in.diffusion_models.minimax_h3 import src as _pkg
    from extensions_built_in.diffusion_models.minimax_h3.src import packing

    orig = packing.build_packed_sequence

    def patched(*a, **kw):
        layout = orig(*a, **kw)
        try:
            n_text = int(layout.token_tags.shape[0])
            v = int(layout.video_indices.numel()) if hasattr(layout.video_indices, "numel") else -1
            au = int(layout.audio_indices.numel()) if hasattr(layout.audio_indices, "numel") else -1
        except Exception:
            n_text = v = au = -1
        tags = getattr(layout, "token_tags", None)
        kinds = {}
        if tags is not None:
            try:
                uniq, cnt = tags.unique(return_counts=True)
                kinds = {str(int(u)): int(c) for u, c in zip(uniq, cnt)}
            except Exception:
                pass
        _step_rows["total"] += n_text
        _step_rows["video_rows"] += v
        _step_rows["audio_rows"] += au
        for k, c in kinds.items():
            _step_rows[f"tag{k}"] += c
        _step_calls["seqs"] += 1
        return patched

    patched = _make(orig, _step_rows, _step_calls)
    packing.build_packed_sequence = patched
    # transformer module imports the symbol directly; repatch there too
    import sys
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        if getattr(mod, "build_packed_sequence", None) is orig:
            mod.build_packed_sequence = patched


def _make(orig, rows, calls):
    def patched(*a, **kw):
        layout = orig(*a, **kw)
        try:
            total = int(layout.sequence_length)
            v = int(layout.video_indices.numel())
            au = int(layout.audio_indices.numel())
            tags = getattr(layout, "token_tags", None)
            kinds = {}
            if tags is not None:
                uniq, cnt = tags.unique(return_counts=True)
                kinds = {str(int(u)): int(c) for u, c in zip(uniq, cnt)}
        except Exception:
            return layout
        rows["total"] += total
        rows["video_rows"] += v
        rows["audio_rows"] += au
        rows["cond_rows"] += total - v - au - kinds.get("TEXT", 0) * 0
        for k, c in kinds.items():
            rows[f"tag{k}"] += c
        calls["seqs"] += 1
        return layout
    return patched
