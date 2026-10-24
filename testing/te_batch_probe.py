"""TE activation-size probe for phased-loading models (batched-embed-cache planning).

The embedding-cache phase runs with the text encoder as the ONLY resident
component, and the cache loop can batch N captions per forward. Choosing N
per model needs one number: activation bytes per prompt token. This script
measures it directly -- it sums the REAL activation objects (module-boundary
outputs via forward hooks + the gated-delta internals via the accounting
probe in qwen35_fast.ACTIVITY_PROBE), never allocator watermarks -- over a
grid of prompt token lengths x batch sizes, then prints the processed model
(fit intercept + rate per token + R^2) and saves the raw values as JSON.

This is part of the phased-loading rollout checklist: run it once per model
as it gains phased support (moss_image is the reference) before enabling
batched caption caching for it.

Usage:
  python testing/te_batch_probe.py --arch moss_image
  python testing/te_batch_probe.py --arch moss_image \
      --lengths 16 64 128 256 512 1024 2048 3072 4096 \
      --batches 1 2 4 8 16 32 --json tmp/moss_act_probe.json

Only the text encoder is loaded (phase-2 state); OOM cells are skipped and
marked, so the grid can run past what a given card can hold.
"""

import argparse
import json
import os
import sys
import time

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLKIT_ROOT)

from dotenv import load_dotenv

load_dotenv()

import torch

# Generic filler only -- the probe never touches user captions. Words chosen
# to tokenize near 1:1 so calibration converges fast.
_FILLER = "sample object shape color texture quantity light"


def calibrate(tokenizer, template_fn, target_tokens):
    """Build a prompt whose TEMPLATE-APPLIED token count hits target_tokens.

    Returns (prompt, achieved_tokens). template_fn is the model's own chat
    presentation so the measured length is the length the encoder sees."""
    template = template_fn("")
    base = len(tokenizer(template, add_special_tokens=False).input_ids)
    if base >= target_tokens:
        return "", base
    need = target_tokens - base
    words = _FILLER.split()
    prompt = " ".join((words * ((need // len(words)) + 1))[: need])
    # tighten to target within a couple of tokens
    for _ in range(8):
        t = template_fn(prompt)
        got = len(tokenizer(t, add_special_tokens=False).input_ids)
        if abs(got - target_tokens) <= 2:
            return prompt, min(got, target_tokens)
        diff = target_tokens - got
        if diff > 0:
            prompt += " " + " ".join(words[: max(1, diff)])
        else:
            drop = min(-diff + 2, max(1, len(prompt) // 8))
            prompt = prompt[: max(0, len(prompt) - drop * 4)].rstrip()
            if not prompt:
                break
    t = template_fn(prompt)
    return prompt, min(len(tokenizer(t, add_special_tokens=False).input_ids), target_tokens)


class SizeAccount:
    """Collects (category, nbytes) records from module-boundary hooks."""

    def __init__(self):
        self.records = []
        self.enabled = False

    def hook(self, path):
        def _fn(module, inputs, output):
            if not self.enabled:
                return
            tensors = []
            def collect(x):
                if torch.is_tensor(x):
                    tensors.append(x)
                elif isinstance(x, (tuple, list)):
                    for y in x:
                        collect(y)
                elif isinstance(x, dict):
                    for y in x.values():
                        collect(y)
            collect(output)
            nbytes = sum(t.numel() * t.element_size() for t in tensors)
            if nbytes:
                self.records.append((path, nbytes))
        return _fn

    def attach(self, te):
        for name, module in te.named_modules():
            cls_name = type(module).__name__
            if cls_name.endswith("DecoderLayer"):
                module.register_forward_hook(self.hook("layer_stream"))
                continue
            if list(module.children()):
                continue
            if "mlp" in name:
                category = "mlp_internal"
            elif any(tok in name for tok in ("in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj")):
                category = "projections"
            elif "conv1d" in name:
                # count the silu(conv(x)) copy as a second same-sized object:
                # two identical hook records = two real buffers per call
                module.register_forward_hook(self.hook("gdn_conv"))
                module.register_forward_hook(self.hook("gdn_conv"))
                continue
            elif "embed" in name:
                category = "embedding"
            elif "norm" in name:
                category = "norms"
            else:
                continue
            module.register_forward_hook(self.hook(category))

    def total(self):
        return sum(n for _, n in self.records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[16, 64, 128, 256, 512, 1024, 2048, 3072, 4096])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--max-total-tokens", type=int, default=131072)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from toolkit.config_modules import ModelConfig
    from toolkit.util.get_model import get_model_class
    from test_model_loading import MODEL_TESTS

    if args.arch not in MODEL_TESTS:
        print(f"arch {args.arch!r} not in MODEL_TESTS: {sorted(MODEL_TESTS)}")
        sys.exit(1)
    entry = MODEL_TESTS[args.arch]
    cfg = dict(entry["model"])
    cfg.setdefault("arch", args.arch)
    mc = ModelConfig(**cfg)
    cls = get_model_class(mc)
    sd = cls(args.device, mc, dtype="bf16")

    print(f"loading text encoder only (phase-2 state) for {args.arch}", flush=True)
    sd.load_text_encoder()
    te = sd.text_encoder
    te.eval()

    device = torch.device(args.device)
    resident = sum(p.numel() * p.element_size() for p in te.parameters())
    free_b, total_b = torch.cuda.mem_get_info(device)
    print(f"TE resident: {resident / 2**20:.0f} MB | vram free now: {free_b / 2**30:.1f} GiB", flush=True)

    from toolkit.models.v2.text_encoders import qwen35_fast

    account = SizeAccount()
    account.attach(te)

    template_fn = (
        sd._chat if hasattr(sd, "_chat") else (lambda x: x)
    )

    rows = []
    site_detail = None
    for L in args.lengths:
        prompt, real_L = calibrate(sd.tokenizer, template_fn, L)
        max_len = int(getattr(sd, "max_text_length", 4096))
        if real_L > max_len:
            real_L = max_len  # encode truncates at the model's max
        if real_L != L:
            print(f"(target L={L}: achieved {real_L})", flush=True)
        for B in args.batches:
            if B * real_L > args.max_total_tokens:
                continue
            prompts = [prompt] * B
            probe_list = []
            pe = None
            torch.cuda.synchronize()
            try:
                # warmup (allocator/cuBLAS setup) with accounting off
                sd.get_prompt_embeds(prompts)
                torch.cuda.synchronize()
                account.records.clear()
                account.enabled = True
                qwen35_fast.ACTIVITY_PROBE = probe_list
                base_bytes = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.time()
                pe = sd.get_prompt_embeds(prompts)
                torch.cuda.synchronize()
                dt = time.time() - t0
                # measured BEFORE the failed-alloc leftovers are dropped, so
                # the OOM cells are excluded rather than this one
                live_peak = torch.cuda.max_memory_allocated(device) - base_bytes
                out_bytes = sum(
                    t.numel() * t.element_size()
                    for k in pe.keys()
                    for t in pe[k]
                )
                act = account.total() + sum(n for _, n in probe_list)
                per_site = {}
                for site, n in account.records + probe_list:
                    per_site[site] = per_site.get(site, 0) + n
            except torch.cuda.OutOfMemoryError:
                # the only place a flush earns its keep: a half-finished
                # forward can leave the allocator holding blocks the next
                # (smaller) cell would otherwise fail on
                qwen35_fast.ACTIVITY_PROBE = None
                account.enabled = False
                pe = None
                probe_list = None
                account.records.clear()
                torch.cuda.empty_cache()
                print(f"B={B:3d} L={real_L:5d}: OOM", flush=True)
                rows.append({"batch": B, "tokens": real_L, "oom": True})
                continue
            qwen35_fast.ACTIVITY_PROBE = None
            account.enabled = False
            pe = None  # drop the embeds before sizing anything derived
            row = {
                "batch": B,
                "tokens": real_L,
                "total_tokens": B * real_L,
                "activation_bytes": act,
                "live_peak_bytes": live_peak,
                "output_bytes": out_bytes,
                "seconds": dt,
                "mb_per_token": act / (B * real_L) / 2**20,
                "live_mb_per_token": live_peak / (B * real_L) / 2**20,
                "sec_per_caption": dt / B,
            }
            if B == 1 and real_L == 256:
                site_detail = per_site
            rows.append(row)
            print(
                f"B={B:3d} L={real_L:5d}  act={act / 2**20:8.1f} MB"
                f"  live={live_peak / 2**20:8.1f} MB"
                f"  live_rate={live_peak / (B * real_L):7.0f} B/tok"
                f"  out={out_bytes / 2**20:7.1f} MB"
                f"  {dt:6.2f} s ({dt / B:.3f} s/caption)",
                flush=True,
            )
            del prompts, probe_list, per_site, row

    # processed output: bytes = intercept + rate * total_tokens, for BOTH
    # metrics. total object-sums describe the model; the live-peak fit is the
    # one the batched-cache scheduler sizes against.
    fit_rows = [r for r in rows if not r.get("oom")]
    import statistics

    def fit(key):
        xs = [r["total_tokens"] for r in fit_rows]
        ys = [r[key] for r in fit_rows]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        rate = sxy / sxx if sxx else 0.0
        intercept = my - rate * mx
        ss_res = sum((y - (intercept + rate * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys) or 1.0
        return intercept, rate, 1.0 - ss_res / ss_tot

    print("\n=== processed ===")
    fits = {}
    for key, label in (("activation_bytes", "object-sums"), ("live_peak_bytes", "live peak")):
        intercept, rate, r2 = fit(key)
        fits[key] = {"intercept_bytes": intercept, "rate_bytes_per_token": rate, "r2": r2}
        print(f"{label:>12}: bytes = {intercept / 2**20:7.1f} MB + {rate:9.0f} B/token x total_tokens   R^2={r2:.4f}")
    rates = [r["live_mb_per_token"] for r in fit_rows]
    print(f"live rate spread: min={min(rates):.3f} median={statistics.median(rates):.3f} max={max(rates):.3f} MB/token  ({len(fit_rows)} cells)")
    headroom = free_b * 0.9
    rate = fits["live_peak_bytes"]["rate_bytes_per_token"]
    intercept = fits["live_peak_bytes"]["intercept_bytes"]
    if rate > 0:
        print(f"scheduler math: budget_tokens = (headroom - intercept) / rate; headroom now {headroom / 2**30:.1f} GiB")
        budget = max(0.0, (headroom - intercept) / rate)
        print(f"  token budget at current free vram: {int(budget)} tokens")
        for L in (256, 1024, 4096):
            print(f"  uniform-batch max bs @ L={L}: {int(budget // L)}")

    out = {
        "arch": args.arch,
        "resident_bytes": resident,
        "vram_total_bytes": total_b,
        "vram_free_bytes": free_b,
        "rows": rows,
        "fits": fits,
        "site_detail_b1_l256": site_detail,
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"raw values saved to {args.json}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(TOOLKIT_ROOT, "testing"))
    main()
