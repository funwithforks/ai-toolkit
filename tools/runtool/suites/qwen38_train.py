"""qwen38_train suite: Qwen3.8-27B LoRA training-step measurement.

All generic machinery lives in tools/runtool/trainer_kit.py (harness,
loud merge, dataset guards, metrics, watchers, sampling option, phase
markers). This module intentionally exposes no model switches yet: per
the per-model rollout rule, a param appears here only once its
model-level switch exists and is measured on THIS model.

Owner-template rule: the tool changes name/steps/batch/sampling/ckpt/
save only (loud-merged); everything else ships as the owner set it.

Job naming (kit mechanism, approved format):
QWEN38_<kind>_<dataset>_bs<N>[_<flags>][_purpose]_v<N> -- dataset keys
resolve from the machine-local registry at submit; folder paths never
live in tracked code.
"""

from ..trainer_kit import TrainerKit

_kit = TrainerKit(
    namespace="QWEN38",
    module="tools.runtool.suites.qwen38_train",
)


SUITE = _kit.suite(
    suite_id="qwen38_train",
    variants=("v1",),
    timeout_s=4 * 3600,
    desc="Qwen3.8-27B LoRA training-step measurement (run.py, sqlite-structured metrics)",
)


if __name__ == "__main__":
    import sys

    sys.exit(_kit.main())
