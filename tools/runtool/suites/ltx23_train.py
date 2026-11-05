"""ltx23_train suite: LTX-2.3 training-step measurement.

All generic machinery lives in tools/runtool/trainer_kit.py (harness,
loud merge, dataset guards, metrics, watchers, sampling option, phase
markers). This module keeps ONLY what is specific to LTX-2.3.

The quantization axis is param-owned config (loud-merged with full
provenance, refused on template disagreement, unreachable to
config_patch) so the comparison runs are four param sets on one suite:

* qfloat8            - qtype=qfloat8, ship excludes: the upstream-
                       equivalent qfloat8 pass on the bf16 checkpoint
* convrot_ship       - qtype=convrot8, ship excludes: the shipped int8
                       layer scheme quantized fresh from bf16
* convrot_all        - qtype=convrot8, quantize_exclude=none: the same
                       layers the qfloat8 pass covers, in convrot
* prequant_int8      - dit_path=<int8 convrot file>, quantize=false:
                       the pre-quantized checkpoint, zero casting
                       (the TE still quantizes per quantize_te)

The bf16 mono checkpoint stays name_or_path in the owner template; the
pre-quant run overrides only the transformer via model_kwargs.dit_path
(the documented per-component override), so one template serves all four.

Job naming (kit mechanism, approved format):
LTX23_<kind>_<dataset>_bs<N>[_<flags>][_purpose]_v<N> -- flags mark the
quant arm (convrot/noexcl/prequant/noquant). Dataset keys resolve from
the machine-local registry only (gitignored).
"""

from ..registry import Param
from ..trainer_kit import TrainerKit

_MK = "config.process[0].model.model_kwargs"

_kit = TrainerKit(
    namespace="LTX23",
    module="tools.runtool.suites.ltx23_train",
    extra_params={
        "qtype": Param(
            "enum", default="qfloat8", choices=("qfloat8", "convrot8"),
            desc="transformer quantization (model.qtype); convrot8 with the "
                 "ship excludes reproduces the shipped int8 layer scheme",
        ),
        "quantize": Param(
            "bool", default=True,
            desc="model.quantize: false runs the transformer exactly as "
                 "shipped (the pre-quant int8 artifact); the text encoder "
                 "still follows quantize_te from the template",
        ),
        "quantize_exclude": Param(
            "enum", default="ship", choices=("ship", "none"),
            desc="convrot layer scheme: ship = the shipped int8 excludes; "
                 "none = the same layers the qfloat8 pass covers",
        ),
        "dit_path": Param(
            "str", default="",
            desc="model_kwargs.dit_path override: the pre-quantized int8 "
                 "convrot transformer file for the as-shipped run (empty = "
                 "resolve from name_or_path)",
        ),
    },
    switches={
        "qtype": "config.process[0].model.qtype",
        "quantize": "config.process[0].model.quantize",
        "quantize_exclude": _MK + ".quantize_exclude",
        "dit_path": _MK + ".dit_path",
    },
    extra_flags=lambda params: [
        *([] if params.get("qtype", "qfloat8") == "qfloat8" else ["convrot"]),
        *([] if params.get("quantize_exclude", "ship") == "ship" else ["noexcl"]),
        *([] if params.get("quantize", True) else ["noquant"]),
        *(["prequant"] if params.get("dit_path") else []),
    ],
)


SUITE = _kit.suite(
    suite_id="ltx23_train",
    variants=("v1",),
    timeout_s=24 * 3600,  # full training runs on a 22B model
    desc="LTX-2.3 training measurement (run.py, sqlite-structured metrics)",
)


if __name__ == "__main__":
    import sys

    sys.exit(_kit.main())
