"""ltx2_train suite: LTX-2.5 training-step measurement.

All generic machinery lives in tools/runtool/trainer_kit.py. LTX-2.5
carries NO model switches of its own: the measured stack ships in the
owner template, so this declaration is pure base_params (that is also
the proof that a zero-switch model costs zero suite code).

Job naming (kit mechanism, approved format):
LTX_<kind>_<dataset>_bs<N>[_<flags>][_purpose]_v<N>. Dataset keys
resolve from the machine-local registry only (gitignored).
"""

from ..trainer_kit import TrainerKit

_kit = TrainerKit(
    namespace="LTX",
    module="tools.runtool.suites.ltx2_train",
)


SUITE = _kit.suite(
    suite_id="ltx2_train",
    variants=("v1",),
    timeout_s=6 * 3600,  # video steps are slow; long measurements fit
    desc="LTX-2.5 training-step measurement (run.py, sqlite-structured metrics)",
)


if __name__ == "__main__":
    import sys

    sys.exit(_kit.main())
