"""Probe runner: the training-process entry for the h3_probe suite.

Installs a named probe (tools.runtool.suites.probes.<name>) before
run.main(), then trains normally. Probe contract:

    INSTALL(artifacts_path)
        called once before training starts; may patch classes and must
        register its own flush (atexit) so data survives early exits.
    WRAP(hook_train_loop) -> callable
        optional; returns the replacement SDTrainer.hook_train_loop.

Nothing here (and nothing a probe may do) edits the training tree; all
instrumentation is process-local monkeypatching, inert outside this
runner. Usage: python h3_probe_runner.py --config C --probe NAME --out F
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = os.getcwd()
    sys.path.insert(0, repo)

    from tools.runtool.suites import probes as probe_pkg

    probe = probe_pkg.load(args.probe)
    probe.INSTALL(args.out)

    import run as runmod
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    if hasattr(probe, "WRAP"):
        SDTrainer.hook_train_loop = probe.WRAP(SDTrainer.hook_train_loop)

    sys.argv = ["run.py", args.config]
    runmod.main()


if __name__ == "__main__":
    main()
