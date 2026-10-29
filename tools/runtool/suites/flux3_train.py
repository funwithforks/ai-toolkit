"""flux3_train suite: FLUX.3 Image training-step measurement.

All generic machinery lives in tools/runtool/trainer_kit.py. This
module keeps ONLY what is specific to FLUX.3.

Switch routes (both fully recorded -- see h3_train's docstring for the
route contract):
* env route: rope/norm (FLUX3_ROPE_FUSE / FLUX3_NORM_FUSE fused Triton
  cores; false restores the eager chain for A/B controls).
* model_kwargs route: int8_bwd and attention_backend ride the loud
  merge into config.process[0].model.model_kwargs.* with provenance.
  The pre-kit version's silent "cudnn" fallback (contradicting the
  declared native default) is gone: the switch value comes from the
  param, the only source.

Job naming (kit mechanism, approved format):
F3_<kind>_<dataset>_bs<N>[_<flags>][_purpose]_v<N> (F3 kept from the
pre-kit records for continuity). Dataset keys resolve from the
machine-local registry only.
"""

from ..registry import Param
from ..trainer_kit import TrainerKit

_MK = "config.process[0].model.model_kwargs"

_kit = TrainerKit(
    namespace="F3",
    module="tools.runtool.suites.flux3_train",
    extra_params={
        "int8_bwd": Param(
            "bool", default=True,
            desc="model_kwargs int8_bwd: dequant-free int8 STE input-gradient "
                 "backward on convrot linears (quantizes upstream grads). "
                 "Suite default ON matches the measured-best stack; pass "
                 "false to isolate the eager backward",
        ),
        "rope": Param(
            "bool", default=True,
            desc="FLUX3_ROPE_FUSE: fused bitwise Triton rope (false = eager "
                 "chain, for A/B controls)",
        ),
        "norm": Param(
            "bool", default=True,
            desc="FLUX3_NORM_FUSE: fused Triton QK RMSNorm core (false = "
                 "eager fp32 chain, for A/B controls)",
        ),
        "attn": Param(
            "enum", default="native", choices=("native", "cudnn", "flash"),
            desc="model_kwargs attention_backend (cudnn: 1.7x in isolation, "
                 "-5.8% in the live checkpointed step -- native stays)",
        ),
    },
    switches={
        "int8_bwd": _MK + ".int8_bwd",
        "attn": _MK + ".attention_backend",
    },
    extra_flags=lambda params: [
        *(["no8bwd"] if not params.get("int8_bwd", True) else []),
        *([] if params.get("rope", True) else ["norope"]),
        *([] if params.get("norm", True) else ["nonorm"]),
        *([] if params.get("attn", "native") == "native"
          else ["attn_" + params["attn"]]),
    ],
    build_env=lambda params, paths: {
        "FLUX3_ROPE_FUSE": "1" if params.get("rope", True) else "0",
        "FLUX3_NORM_FUSE": "1" if params.get("norm", True) else "0",
    },
)


SUITE = _kit.suite(
    suite_id="flux3_train",
    variants=("v1",),
    timeout_s=6 * 3600,
    desc="FLUX.3 Image training-step measurement (run.py, sqlite-structured metrics)",
)


if __name__ == "__main__":
    import sys

    sys.exit(_kit.main())
