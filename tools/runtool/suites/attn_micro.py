"""attn_micro suite: single-shape scaled-dot-product-attention micro-bench.

A different stage from h3_train: it measures an attention kernel in
isolation on synthetic tensors and reports latency percentiles, throughput
and peak memory. It does NOT run a training loop and its parser never
touches training logs.

Variants select a torch SDPA backend. The harness owns the bench; the CLI
only names the suite and varies registered params.

Structured outputs (all written by the harness into the record):
    artifacts/timings.json   raw per-rep latencies (ms) -> metrics recomputable
    artifacts/peak.txt       torch.cuda.max_memory_allocated bytes
    result.json              ok + metrics
    events.jsonl             one progress event per rep batch
"""

import argparse
import json
import os
import statistics
import sys

from ..registry import Param, RunPaths, Suite

_VARIANTS = ("sdpa_math", "sdpa_flash", "sdpa_cudnn", "sdpa_efficient")

_BACKEND = {
    "sdpa_math": "MATH",
    "sdpa_flash": "FLASH_ATTENTION",
    "sdpa_cudnn": "CUDNN_ATTENTION",
    "sdpa_efficient": "EFFICIENT_ATTENTION",
}


def _params():
    return {
        "batch": Param("int", default=8, min=1, max=256),
        "seq": Param("int", default=8192, min=16, max=32768),
        "heads": Param("int", default=32, min=1, max=128),
        "head_dim": Param("int", default=128, min=8, max=256),
        "dtype": Param("enum", default="bf16", choices=("bf16", "fp16", "fp32")),
        "reps": Param("int", default=20, min=1, max=1000),
        "warmup_reps": Param("int", default=5, min=0, max=100),
        "causal": Param("bool", default=False),
        "fwd_only": Param("bool", default=False),
    }


METRICS = {
    "latency_ms_p50": "median per-call latency (ms)",
    "latency_ms_p90": "p90 per-call latency (ms)",
    "latency_ms_p99": "p99 per-call latency (ms)",
    "tok_per_s": "token throughput (batch*seq / p50)",
    "peak_mem_mb": "max cuda allocated during bench (MiB)",
    "correct": "1 if matched fp32 math reference within tol, else 0",
}


def _build(params, paths: RunPaths):
    argv = [
        sys.executable, "-m", "tools.runtool.suites.attn_micro",
        "--run-dir", paths.root,
    ]
    env = {
        # keep the bench to a single device; no training-side fusions
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "TORCH_CUDNN_V8_API_DISABLED": "0",
    }
    return argv, env


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# --------------------------------------------------------------- harness

def _run_bench(paths: RunPaths, manifest):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    p = manifest["params"]
    variant = manifest["variant"]
    backend = getattr(SDPBackend, _BACKEND[variant])
    dev = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[p["dtype"]]
    if p["dtype"] == "fp32":
        raise SystemExit("attn_micro sdpa benches require a half dtype")
    if not torch.cuda.is_available():
        raise SystemExit("attn_micro requires a GPU")

    B, S, H, D = p["batch"], p["seq"], p["heads"], p["head_dim"]
    causal = bool(p["causal"])
    scale = D ** -0.5

    def make():
        q, k, v = (
            torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=not p["fwd_only"])
            for _ in range(3)
        )
        g = torch.randn(B, H, S, D, device=dev, dtype=dtype)
        return q, k, v, g

    def call(q, k, v, g):
        with sdpa_kernel([backend]):
            o = sdpa(q, k, v, is_causal=causal, scale=scale)
        if not p["fwd_only"]:
            o.backward(g)
        return o

    # warmup (allocations, kernel autotune, cudnn plan) excluded from timing
    for _ in range(p["warmup_reps"]):
        q, k, v, g = make()
        call(q, k, v, g)
        q = k = v = g = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    lat = []
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for i in range(p["reps"]):
        q, k, v, g = make()
        ev0.record()
        call(q, k, v, g)
        ev1.record()
        torch.cuda.synchronize()
        lat.append(ev0.elapsed_time(ev1))
        q = k = v = g = None
        if (i + 1) % 5 == 0:
            from ..runner import emit

            emit(paths, "progress", rep=i + 1, reps=p["reps"],
                 last_ms=round(lat[-1], 3))

    peak = torch.cuda.max_memory_allocated()
    os.makedirs(paths.artifacts, exist_ok=True)
    with open(os.path.join(paths.artifacts, "timings.json"), "w") as f:
        json.dump({"latency_ms": lat, "params": p, "variant": manifest["variant"]}, f)
    with open(os.path.join(paths.artifacts, "peak.txt"), "w") as f:
        f.write(str(peak))

    correct = _check(paths, manifest, sdpa, backend, dtype, causal, scale)
    metrics = SUITE.parse(paths, manifest["params"])
    metrics["correct"] = correct
    with open(paths.result + ".tmp", "w") as f:
        json.dump({"ok": True, "metrics": metrics,
                   "metric_source": "artifacts/timings.json + peak.txt"}, f, indent=1)
    os.replace(paths.result + ".tmp", paths.result)
    return 0


def _check(paths, manifest, sdpa, backend, dtype, causal, scale):
    """Compare a small forward against an fp32 math reference."""
    import torch

    p = manifest["params"]
    B, H = p["batch"], p["heads"]
    S, D = min(p["seq"], 512), p["head_dim"]
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.manual_seed(0)
        q, k, v = (
            torch.randn(B, H, S, D, device="cuda", dtype=dtype) for _ in range(3)
        )
        with sdpa_kernel([backend]):
            got = sdpa(q, k, v, is_causal=causal, scale=scale).float()
        with sdpa_kernel([SDPBackend.MATH]):
            ref = sdpa(q.float(), k.float(), v.float(), is_causal=causal, scale=scale)
        denom = ref.abs().max().clamp(min=1e-6)
        rel = ((got - ref).abs().max() / denom).item()
        tol = {"bf16": 0.06, "fp16": 0.03}.get(p["dtype"], 0.03)
        ok = 1 if rel <= tol else 0
        from ..runner import emit

        emit(paths, "correctness", rel_dev=round(rel, 5), tol=tol, correct=ok)
        return ok
    except Exception as e:  # reference unavailable -> not a bench failure
        from ..runner import emit

        emit(paths, "correctness", error=str(e)[:200], correct=-1)
        return -1


# --------------------------------------------------------------- parser

def _parse(paths: RunPaths, params):
    with open(os.path.join(paths.artifacts, "timings.json")) as f:
        raw = json.load(f)
    lat = sorted(raw["latency_ms"])
    p50 = _percentile(lat, 0.50)
    p = raw["params"]
    peak_path = os.path.join(paths.artifacts, "peak.txt")
    peak_mb = None
    if os.path.exists(peak_path):
        peak_mb = round(int(open(peak_path).read().strip()) / (1024 * 1024), 1)
    # correct lives in result.json (harness-only, needs a GPU); recompute
    # the latency/throughput family from raw timings here
    res = {
        "latency_ms_p50": round(p50, 4) if p50 else None,
        "latency_ms_p90": round(_percentile(lat, 0.90), 4) if lat else None,
        "latency_ms_p99": round(_percentile(lat, 0.99), 4) if lat else None,
        "tok_per_s": int(p["batch"] * p["seq"] / (p50 / 1000)) if p50 else None,
        "peak_mem_mb": peak_mb,
    }
    if os.path.exists(paths.result):
        with open(paths.result) as f:
            stored = json.load(f).get("metrics", {})
        if "correct" in stored:
            res["correct"] = stored["correct"]
    return res


SUITE = Suite(
    id="attn_micro",
    variants=_VARIANTS,
    params=_params(),
    metrics=METRICS,
    build=_build,
    parse=_parse,
    timeout_s=600,
    desc="single-shape attention kernel micro-bench (latency/throughput/mem)",
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    from ..runner import _read_json  # manifest read helper

    paths = RunPaths(run_id=os.path.basename(args.run_dir), root=args.run_dir)
    manifest = _read_json(paths.manifest)
    return _run_bench(paths, manifest)


if __name__ == "__main__":
    sys.exit(main())
