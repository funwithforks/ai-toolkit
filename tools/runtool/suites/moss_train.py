"""moss_train suite: MOSS-Image training-step measurement.

All generic machinery lives in tools/runtool/trainer_kit.py (harness,
loud merge, dataset guards, metrics, watchers, sampling option, phase
markers). This module keeps ONLY what is specific to MOSS-Image; it is
the template for future model support: what does not fit here belongs
in the kit, not copied here.

Owner-template rule: the tool changes name/steps/batch/sampling/ckpt/
save only (loud-merged); everything else ships as the owner set it.

One performance switch is exposed: varlen (model switch
MOSS_VARLEN_ATTN, read by flash_varlen.varlen_ok) -- packed flash-varlen
attention is the DEFAULT for padded bs>1 t2i batches; varlen=false
restores the legacy masked mem-efficient path for A/B measurement. Any
further param appears here only once its model-level switch exists and
is measured on THIS model (per-model rollout rule -- krea2/H3/LTX/flux3
verdicts are not inherited, only their measurement discipline).

Job naming: MOSS_<dataset>_bs<batch>_<steps>[_<tag>] (kit mechanism,
owner rule). Dataset folder paths never live in tracked code: they
resolve at run time from the machine-local registry (gitignored).
"""

from ..registry import Param
from ..trainer_kit import TrainerKit

_kit = TrainerKit(
    namespace="MOSS",
    module="tools.runtool.suites.moss_train",
    extra_params={
        "varlen": Param(
            "bool", default=True,
            desc="packed flash-varlen attention for padded batches (model "
                 "switch MOSS_VARLEN_ATTN, the standard for bs>1); false "
                 "restores the legacy masked mem-efficient path for A/B",
        ),
    },
    build_env=lambda params, paths: {
        # the packed-varlen model switch (flash_varlen.varlen_ok reads
        # the env; default on). A/B: varlen=false reruns the masked
        # reference.
        "MOSS_VARLEN_ATTN": "1" if params.get("varlen", True) else "0",
    },
)


SUITE = _kit.suite(
    suite_id="moss_train",
    variants=("v1",),
    timeout_s=12 * 3600,  # quality runs (1400 steps at bs8 + sampling) pass 6 h
    desc="MOSS-Image training-step measurement (run.py, sqlite-structured metrics)",
)


if __name__ == "__main__":
    import sys

    sys.exit(_kit.main())
