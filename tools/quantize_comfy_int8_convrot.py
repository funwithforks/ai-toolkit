#!/usr/bin/env python
"""Stream a bf16 comfy transformer checkpoint to int8 ConvRot.

Model-agnostic file->file conversion in the comfy_quant layout the toolkit
already imports (toolkit/util/comfy_quant_import.py): every planned module
gets {int8 weight, fp32 per-row weight_scale, comfy_quant marker}; everything
else passes through byte-identical. The quantizer, the group-size rule and
the GEMM shape gate all come from toolkit/util/convrot_quant.py — this script
reimplements no math, so what it writes is exactly what the runtime loads.

Peak memory is one layer: weights are faulted one key at a time, quantized
on the target device, and the int8 results stream straight to disk through
StWriter. Default (no search flags) reproduces the toolkit's own convrot8
quantization bit-for-bit: gs = min(256, largest_pow4_divisor(in)), c = 1.0.

Search (--gs-search / --clip-search) is a per-layer proxy over the weight
MSE only. c is legitimate for it (it only lands in stored weight scales);
a gs WIN IS A HYPOTHESIS — the group size re-bases the runtime activation
rotation, which this score cannot see. Promote gs changes only against real
denoise samples.

The exclusion list below is the LTX-2.5 shipped scheme (verified against the
official int8 file's marker set); pass --exclude to adjust for a model.

Usage:
    python tools/quantize_comfy_int8_convrot.py SRC.safetensors DST.safetensors
    python tools/quantize_comfy_int8_convrot.py SRC DST --dry-run
    python tools/quantize_comfy_int8_convrot.py SRC DST --clip-search
"""

import argparse
import fnmatch
import json
import os
import sys

import torch
from safetensors import safe_open
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from toolkit.util.comfy_quant_import import import_comfy_quantized_layers, parse_comfy_quant_blob
from toolkit.util.convrot_quant import (
    ConvRotInt8Quantizer,
    largest_pow4_divisor,
    quantize_int8_rows,
    rotate,
)
from toolkit.util.streaming_safetensors import StWriter

# LTX shipped scheme: everything 2-D and shape-qualifying is quantized
# except these (file-key-space fnmatch on the module path, no .weight
# suffix). The steering layers (timestep embedders, scale-shift tables,
# caption projection) stay full precision in the Lightricks int8-convrot
# releases: rounding them is how a quantized DiT dies. The shape gate
# currently catches the tables by accident (out dims 2/5/9); the names
# make that intent permanent.
LTX_EXCLUDES = (
    "*to_gate_logits",
    "*adaln_single*",
    "*patchify_proj",
    "*proj_out",
    "*caption_projection*",
    "*scale_shift_table*",
    "*timestep_embedder*",
)

GS_CANDIDATES = (16, 64, 256, 1024)
CLIP_GRID = (1.0, 0.99, 0.97, 0.95, 0.9, 0.85)
_CHUNK_BYTES = 256 * 1024 * 1024  # matches ConvRotInt8Quantizer._QUANT_CHUNK_BYTES


def default_gs(in_features):
    return min(256, largest_pow4_divisor(in_features))


def shape_gate(in_features, out_features):
    # the real gate, on a meta shell: zero allocation, zero drift
    with torch.device("meta"):
        lin = torch.nn.Linear(in_features, out_features, bias=False)
    return ConvRotInt8Quantizer(rot_size=256).can_quantize(lin)


def build_plan(src_keys, excludes, only=None):
    plan = {}
    for key in src_keys:
        if not key.endswith(".weight"):
            continue
        base = key[: -len(".weight")]
        if any(fnmatch.fnmatch(base, pat) for pat in excludes):
            continue
        if only is not None and not any(fnmatch.fnmatch(base, pat) for pat in only):
            continue
        plan[base] = {}  # spec filled at emit time (search) or left default
    return plan


def make_marker(gs, rotated):
    conf = {"format": "int8_tensorwise", "convrot": bool(rotated)}
    if rotated:
        conf["convrot_groupsize"] = int(gs)
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _rows_chunked(W, gs, clip):
    # chunked rotate+quantize: one fp32 chunk transient, never one layer
    rows, inn = W.shape
    chunk = max(1, _CHUNK_BYTES // (inn * 4))
    q = torch.empty(rows, inn, dtype=torch.int8, device=W.device)
    scales = torch.empty(rows, dtype=torch.float32, device=W.device)
    for i in range(0, rows, chunk):
        wr = rotate(W[i : i + chunk].float(), gs)
        q[i : i + chunk], scales[i : i + chunk] = quantize_int8_rows(wr, clip=clip)
    return q, scales


@torch.no_grad()
def search_gs_c(W, use_gs, use_clip):
    # per-layer proxy: chunked SSE per candidate, float64 accumulators, one
    # wr resident at a time; the winner is requantized once by the caller
    rows, inn = W.shape
    chunk = max(1, _CHUNK_BYTES // (inn * 4))
    gss = GS_CANDIDATES if use_gs else (default_gs(inn),)
    cs = CLIP_GRID if use_clip else (1.0,)
    best = None  # (sse, gs, c)
    for gs in gss:
        if inn % gs != 0:
            continue
        sse = {c: torch.zeros((), device=W.device, dtype=torch.float64) for c in cs}
        for i in range(0, rows, chunk):
            wr = rotate(W[i : i + chunk].float(), gs)
            for c in cs:
                q, scales = quantize_int8_rows(wr, clip=c)
                sse[c] += (q.float() * scales.unsqueeze(1) - wr).square().sum()
            del wr
        for c in cs:
            if best is None or sse[c] < best[0]:
                best = (sse[c], gs, c)
    return best[1], best[2]


def emit_quant(writer, base, weight, spec, device, use_gs, use_clip, stats, prev=None):
    weight = weight.to(device)
    if weight.dtype == torch.int8:
        # requantizing an already-quantized source: rebuild the float the
        # runtime actually sees (scale, then inverse rotation) — quantizing
        # the int8 codes directly would rotate garbage
        if prev is None or prev[0] is None:
            raise ValueError(f"{base}: int8 weight without scale/marker sidecars")
        rot_old = int(prev[1].get("convrot_groupsize", 256)) if prev[1].get("convrot") else 1
        weight = rotate(
            weight.float() * prev[0].to(weight.device).float().reshape(-1, 1), rot_old
        )
    inn = weight.shape[1]
    if "gs" in spec or "clip" in spec:
        gs, clip = spec.get("gs", default_gs(inn)), spec.get("clip", 1.0)
    elif use_gs or use_clip:
        gs, clip = search_gs_c(weight, use_gs, use_clip)
    else:
        gs, clip = default_gs(inn), 1.0
    rotated = inn % gs == 0 and gs >= 16
    eff_gs = gs if rotated else 1
    q, scales = _rows_chunked(weight, eff_gs, clip)
    writer.add(f"{base}.weight", q.cpu())
    writer.add(f"{base}.weight_scale", scales.unsqueeze(1).cpu())
    writer.add(f"{base}.comfy_quant", make_marker(gs, rotated))
    stats.setdefault("gs", {}).setdefault(eff_gs, 0)
    stats["gs"][eff_gs] += 1
    stats.setdefault("chosen", {})[base] = (eff_gs, clip)
    stats["clips"] = stats.get("clips", set()) | {clip}
    del weight, q, scales


def convert(src, dst, plan, device, use_gs, use_clip):
    with safe_open(src, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        all_keys = set(f.keys())
        stats = {}
        seen = set()
        # sidecars of planned layers are always replaced by the re-emit, no
        # matter what order this file lists them in (a planned base with no
        # .weight is a hard error at the end)
        replaced_sidecars = {
            f"{b}.{suf}" for b in plan for suf in ("weight_scale", "comfy_quant")
        }
        with StWriter(dst, metadata=dict(metadata or {"format": "pt"})) as w:
            for key in tqdm(list(f.keys()), desc="converting"):
                if key.endswith(".weight") and key[: -len(".weight")] in plan:
                    base = key[: -len(".weight")]
                    prev = None
                    if f"{base}.comfy_quant" in all_keys:
                        prev = (
                            f.get_tensor(f"{base}.weight_scale"),
                            parse_comfy_quant_blob(f.get_tensor(f"{base}.comfy_quant")),
                        )
                    emit_quant(w, base, f.get_tensor(key), plan[base], device,
                               use_gs, use_clip, stats, prev=prev)
                    seen.add(base)
                elif key in replaced_sidecars:
                    continue  # replaced by this pass
                else:
                    # passthrough (also carries sidecars of un-planned layers,
                    # so partial re-quantization of an already-quantized file
                    # stays valid)
                    w.add(key, f.get_tensor(key))
    missing = set(plan) - seen
    if missing:
        raise KeyError(f"plan entries with no .weight in source: {sorted(missing)[:5]}")
    return stats


@torch.no_grad()
def self_check(src, dst, plan, stats, n_passthrough=8):
    with safe_open(src, framework="pt", device="cpu") as fs, \
            safe_open(dst, framework="pt", device="cpu") as fd:
        src_keys, dst_keys = set(fs.keys()), set(fd.keys())
        # only a re-emitted layer's stale sidecars may disappear; everything
        # else passes through under its own name
        for k in src_keys - dst_keys:
            assert k.endswith((".weight_scale", ".comfy_quant")) and k.rsplit(".", 1)[0] in plan, k
        triples = {f"{b}.{suf}" for b in plan for suf in ("weight", "weight_scale", "comfy_quant")}
        assert (dst_keys - src_keys) <= triples, sorted((dst_keys - src_keys) - triples)[:5]
        for base in plan:
            for suf in ("weight", "weight_scale", "comfy_quant"):
                assert f"{base}.{suf}" in dst_keys, f"{base}.{suf}"
            conf = parse_comfy_quant_blob(fd.get_tensor(f"{base}.comfy_quant"))
            assert conf["format"] == "int8_tensorwise" and "convrot" in conf, conf
        for key in dst_keys:
            fd.get_tensor(key)  # every tensor reads back (offset/dtype validity)
        pt = [
            k for k in sorted(src_keys & dst_keys)
            if not (k.endswith(".weight") and k[: -len(".weight")] in plan)
            and not (k.endswith((".weight_scale", ".comfy_quant")) and k.rsplit(".", 1)[0] in plan)
        ][:n_passthrough]
        for key in pt:
            assert torch.equal(fs.get_tensor(key), fd.get_tensor(key)), key
    print(f"self-check: {len(dst_keys)} tensors read back, {len(plan)} triples, "
          f"{len(pt)} passthrough samples byte-identical")
    # runtime round-trip on the smallest planned layer: the file must attach
    # through the real importer and convert exactly one module
    with safe_open(src, framework="pt", device="cpu") as fs, \
            safe_open(dst, framework="pt", device="cpu") as fd:
        smallest = min(plan, key=lambda b: fd.get_slice(f"{b}.weight").get_shape()[0])
        q = fd.get_tensor(f"{smallest}.weight")
        root = torch.nn.Module()
        *parents, attr = smallest.split(".")
        node = root
        for part in parents:
            child = torch.nn.Module()
            setattr(node, part, child)
            node = child
        setattr(node, attr, torch.nn.Linear(q.shape[1], q.shape[0], bias=False, device="meta"))
        remaining, n = import_comfy_quantized_layers(
            root, {
                f"{smallest}.weight": q,
                f"{smallest}.weight_scale": fd.get_tensor(f"{smallest}.weight_scale"),
                f"{smallest}.comfy_quant": fd.get_tensor(f"{smallest}.comfy_quant"),
            }
        )
        assert n == 1 and not remaining
        # code-level check: recompute the smallest layer's quantization from
        # the source float and require bit-identical codes and scales. A
        # wrong Hadamard or wrong grid passes everything above; it cannot
        # survive this comparison
        eff_gs, clip_used = stats["chosen"][smallest]
        ws = fs.get_tensor(f"{smallest}.weight")
        if ws.dtype == torch.int8:
            conf_s = parse_comfy_quant_blob(fs.get_tensor(f"{smallest}.comfy_quant"))
            rot_old = int(conf_s.get("convrot_groupsize", 256)) if conf_s.get("convrot") else 1
            ws = rotate(
                ws.float() * fs.get_tensor(f"{smallest}.weight_scale").float().reshape(-1, 1),
                rot_old,
            )
        q2, s2 = quantize_int8_rows(rotate(ws.float(), eff_gs), clip=clip_used)
        assert torch.equal(q2, q), f"codes differ on {smallest}"
        assert torch.equal(
            s2, fd.get_tensor(f"{smallest}.weight_scale").reshape(-1)
        ), f"scales differ on {smallest}"
    print(f"self-check: importer round-trip + code recompute OK on {smallest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--exclude", action="append", default=None,
                    help="fnmatch on the module path (replaces the LTX default set)")
    ap.add_argument("--only", action="append",
                    help="restrict the plan to these module-path globs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gs-search", action="store_true",
                    help="per-layer group-size search (hypothesis; verify on samples)")
    ap.add_argument("--clip-search", action="store_true",
                    help="per-layer weight clip-ratio search over the stored scale")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--no-selfcheck", action="store_true")
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        ap.error("refusing to overwrite the source file")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    excludes = args.exclude if args.exclude is not None else list(LTX_EXCLUDES)

    with safe_open(args.src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    plan = build_plan(keys, excludes, args.only)
    # shape-gate the plan with the real runtime rule (drops e.g. 1-D weights
    # and GEMM-unsupported linears, which stay full precision like the gate
    # leaves them in a live load); header-only, no tensor reads
    gated = {}
    with safe_open(args.src, framework="pt", device="cpu") as f:
        for base in plan:
            shape = f.get_slice(f"{base}.weight").get_shape()
            if len(shape) == 2 and shape_gate(shape[1], shape[0]):
                gated[base] = plan[base]
            else:
                print(f"plan: {base} shape {tuple(shape)} fails the int8 gate -> full precision")
    plan = gated
    if not plan:
        ap.error("plan is empty after --only/--exclude/gate filtering — "
                 "patterns match base names WITHOUT the .weight suffix")
    print(f"plan: {len(plan)} modules quantized, "
          f"{sum(1 for k in keys if k.endswith('.weight')) - len(plan)} weights kept")
    if args.dry_run:
        for base in sorted(plan)[:40]:
            print("  ", base)
        if len(plan) > 40:
            print(f"   ... ({len(plan) - 40} more)")
        return
    stats = convert(args.src, args.dst, plan, device, args.gs_search, args.clip_search)
    print(f"wrote {args.dst}: group sizes {dict(sorted(stats.get('gs', {}).items()))}, "
          f"clip values {sorted(stats.get('clips', []))}")
    if not args.no_selfcheck:
        self_check(args.src, args.dst, plan, stats)


if __name__ == "__main__":
    main()
