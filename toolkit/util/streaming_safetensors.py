"""Streaming safetensors writer.

Appends tensor bytes to a temp data file while recording the header, then
assembles <u64 header_len><json header><data> on close. Peak RAM is one
tensor: a whole quantized checkpoint can be emitted layer by layer without
ever holding its tensors in memory (safetensors cannot append, so this is
the append-shaped stand-in).

close() assembles to a sibling temp file and os.replace()s, so the
destination path is never a half-written checkpoint.

Layout follows reference save_file exactly (verified against its output):
tensors packed tight in insertion order, offsets relative to the data
region, no alignment padding, __metadata__ values strings.
"""

import json
import os
import struct

import torch

_DT = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


class StWriter:
    def __init__(self, path, metadata=None):
        self.path, self.tmp = path, path + ".data.tmp"
        self.f = open(self.tmp, "wb")
        self.off, self.hdr = 0, {}
        self.metadata = {k: str(v) for k, v in (metadata or {"format": "pt"}).items()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # an aborted pass must not leave either temp file behind
        if exc_type is not None:
            self.f.close()
            if os.path.exists(self.tmp):
                os.remove(self.tmp)
            return False
        self.close()
        return False

    def add(self, name, t):
        if name in self.hdr:
            raise ValueError(f"{name}: written twice")
        t = t.detach().cpu().contiguous()
        if t.dtype not in _DT:
            raise TypeError(f"{name}: unsupported dtype {t.dtype}")
        # raw little-endian storage copy; the header tag stays the real dtype
        # (numpy has no bf16, so the uint8 view is the only byte path)
        raw = t.view(torch.uint8).numpy().tobytes()
        expect = t.numel() * t.element_size()
        if len(raw) != expect:
            raise RuntimeError(f"{name}: byte count {len(raw)} != {expect}")
        # no alignment padding: reference save_file packs tensors tight
        # (verified against it: odd-sized tensors sit at odd offsets and
        # every reader accepts it); offsets stay relative to the data region
        begin = self.off
        self.f.write(raw)
        self.off += len(raw)
        self.hdr[name] = {
            "dtype": _DT[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [begin, begin + len(raw)],
        }

    def add_alias(self, name, source):
        # safetensors allows two names over the same bytes; save_file does not
        if name in self.hdr:
            raise ValueError(f"{name}: written twice")
        src = self.hdr[source]
        self.hdr[name] = {
            "dtype": src["dtype"],
            "shape": list(src["shape"]),
            "data_offsets": list(src["data_offsets"]),
        }

    def close(self):
        if self.f is None:
            return
        self.f.close()
        self.f = None
        h = dict(self.hdr)
        h["__metadata__"] = self.metadata
        hb = json.dumps(h, separators=(",", ":")).encode("utf-8")
        final = self.path + ".assemble.tmp"
        try:
            with open(final, "wb") as out, open(self.tmp, "rb") as data:
                out.write(struct.pack("<Q", len(hb)))
                out.write(hb)
                while chunk := data.read(1 << 20):
                    out.write(chunk)
            os.replace(final, self.path)
        finally:
            for junk in (self.tmp, final):
                if os.path.exists(junk):
                    os.remove(junk)
