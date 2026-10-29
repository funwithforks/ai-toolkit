"""h3_train suite: MiniMax-H3 training-step measurement.

All generic machinery lives in tools/runtool/trainer_kit.py (harness,
loud merge, dataset guards, metrics, watchers, sampling option, phase
markers, reservation, placement). This module keeps ONLY what is
specific to H3.

Model switches, tracked in two routes (both recorded, nothing hidden):
* env route (build_env): varlen/rope/adaln/liger -- read by the model's
  fusion modules at import (flash_varlen, adaln_fusion); recorded in
  the manifest env. adaln is the H3_ADALN_FUSE switch the toolkit
  honors (the pre-kit suite declared it but never set the env -- wired
  at the kit migration).
* model_kwargs route (switches=): cutlass_bwd/cutlass_fwd land at
  config.process[0].model.model_kwargs.* where the model reads them.
  They are PARAM-OWNED config fields: loud-merged with full
  provenance (submit echo + config_provenance.json + derived config),
  refused on template disagreement, unreachable to config_patch. These
  values measure the run -- they are never side-channel.

Job naming (kit mechanism, approved format):
H3_<kind>_<dataset>_bs<N>[_<flags>][_purpose]_v<N>; flags mark
deviation from the measured-best stack (novarlen, norope, noadaln,
liger, cutlass_<arm>, cutlassfwd). Dataset keys resolve from the
machine-local registry only.
"""

from ..registry import Param
from ..trainer_kit import TrainerKit

_MK = "config.process[0].model.model_kwargs"

_kit = TrainerKit(
    namespace="H3",
    module="tools.runtool.suites.h3_train",
    extra_params={
        "varlen": Param(
            "bool", default=True, desc="H3_VARLEN_ATTN (packed flash-varlen)",
        ),
        "rope": Param(
            "bool", default=True, desc="H3_ROPE_FUSE",
        ),
        "adaln": Param(
            "bool", default=True, desc="H3_ADALN_FUSE (adaln_fusion.py)",
        ),
        "liger": Param(
            "bool", default=False, desc="H3_LIGER_FUSIONS",
        ),
        "cutlass_bwd": Param(
            "enum", default="fc2", choices=("off", "fc2", "full"),
            desc="int8_cutlass_bwd model kwarg: CUTLASS NT dX arm fed by a "
                 "per-step tiled transpose. fc2 is the code default and the "
                 "measured-best arm (~2ms/layer at production m, washes at "
                 "m~1.4k); full also tags fc1-class, which loses; off is the "
                 "escape hatch / A-B control",
        ),
        "cutlass_fwd": Param(
            "bool", default=False,
            desc="int8_cutlass_gemm model kwarg: CUTLASS fused-dequant int8 "
                 "forward on quantized linears (no resident copies; the "
                 "weight buffer is already the operand layout)",
        ),
    },
    switches={
        "cutlass_bwd": _MK + ".int8_cutlass_bwd",
        "cutlass_fwd": _MK + ".int8_cutlass_gemm",
    },
    extra_flags=lambda params: [
        *([] if params.get("varlen", True) else ["novarlen"]),
        *([] if params.get("rope", True) else ["norope"]),
        *([] if params.get("adaln", True) else ["noadaln"]),
        *(["liger"] if params.get("liger") else []),
        *([] if params.get("cutlass_bwd", "fc2") == "fc2"
          else ["cutlass_" + params["cutlass_bwd"]]),
        *(["cutlassfwd"] if params.get("cutlass_fwd") else []),
    ],
    build_env=lambda params, paths: {
        "H3_VARLEN_ATTN": "1" if params.get("varlen", True) else "0",
        "H3_ROPE_FUSE": "1" if params.get("rope", True) else "0",
        "H3_ADALN_FUSE": "1" if params.get("adaln", True) else "0",
        "H3_LIGER_FUSIONS": "1" if params.get("liger") else "0",
    },
)


SUITE = _kit.suite(
    suite_id="h3_train",
    variants=("v2",),  # carried from the pre-kit registry (records cite it)
    timeout_s=4 * 3600,
    desc="H3 training-step measurement (run.py, sqlite-structured metrics)",
)


if __name__ == "__main__":
    import sys

    sys.exit(_kit.main())
