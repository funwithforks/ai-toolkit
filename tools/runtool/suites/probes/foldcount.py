"""LoRA fold accounting: how many LoRA forwards run, how many
up-projections actually ride the convrot epilogue (cr8_lora= passed to
OstrisLinear.forward) vs run as a separate mm, grouped by (org out, org
in, qdata?) and (in_features, rank, input dim). Counters only; flush on
atexit so early trainer exits still write.
"""

import atexit
import gc
from collections import Counter

_stats = Counter()
_done = {"inst": False, "steps": 0}
_OUT = None


def _key_list(counter):
    return [
        {"key": [str(p) for p in k], "count": int(v)}
        for k, v in sorted(counter.items(), key=lambda x: -x[1])
    ]


def _flush():
    if _OUT is None:
        return
    data = {
        "probe": "foldcount",
        "steps_done": _done["steps"],
        "counters": {
            "|".join(e["key"]): e["count"] for e in _key_list(_stats)
        },
    }
    tmp = _OUT + ".tmp"
    with open(tmp, "w") as f:
        json_dump(data, f)
    import os

    os.replace(tmp, _OUT)


def json_dump(data, f):
    import json

    json.dump(data, f, indent=1)


def INSTALL(artifacts_path):
    global _OUT
    _OUT = artifacts_path
    atexit.register(_flush)


def WRAP(hook):
    def wrapped(self, batch):
        _instrument_once()
        out = hook(self, batch)
        _done["steps"] += 1
        return out

    return wrapped


def _instrument_once():
    if _done["inst"]:
        return
    _done["inst"] = True
    from toolkit.util.ostris_quant import OstrisLinear

    for m in gc.get_objects():
        if type(m).__name__ != "LoRAModule" or not hasattr(m, "lora_down"):
            continue
        _wrap_lora(m)
        # lora_special.py:133 captured the ORIGINAL org forward on the lora
        # module and swapped the org instance .forward to lora's; the
        # convrot STE entry (the only place a cr8_lora= kwarg can arrive)
        # is that captured m.org_forward
        of = getattr(m, "org_forward", None)
        rec = getattr(of, "__self__", None)
        if rec is not None and isinstance(rec, OstrisLinear):
            _wrap_org(m, of, rec)


def _wrap_org(lmod, of, rec):
    key = (int(rec.weight.shape[0]), int(rec.weight.shape[1]),
           getattr(rec, "cr8_qdata", None) is not None)

    def fwd(*a, **kw):
        _stats[("ste_entry",) + key
               + ("cr8_lora" if "cr8_lora" in kw else "plain",)] += 1
        return of(*a, **kw)

    lmod.org_forward = fwd


def _wrap_lora(m):
    lf = m.forward
    r = int(m.lora_down.weight.shape[0])
    in_f = int(m.lora_down.weight.shape[1])

    def fwd(x, *a, **kw):
        _stats[("lora_fwd", in_f, r, int(x.dim()))] += 1
        return lf(x, *a, **kw)

    m.forward = fwd
