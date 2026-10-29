"""h3_probe suite: short instrumented training runs for call-site
measurement (mm/fold accounting, module census). The training process
is launched through h3_probe_runner, which installs a named probe
before run.main() -- no base-code edits. Output is structured:
artifacts/probe.json (one entry per observation key) plus the normal
run.log; metrics are reproduced from the record alone.

Kit-shaped (migration 2026-10-28): all generic machinery is the kit's;
this module declares only the probe vocabulary and the two things that
make it probe-shaped --
* parse: metrics come from artifacts/probe.json, not the loss db
  (tag_metrics states the source; it lands in result.json);
* spawn_job: the inner command is the probe runner with --probe/--out,
  not run.py.
The old per-launch timestamp name (fresh-name-anti-resume trick, only
second-resolution) is superseded by the kit's versioned reservation:
a reserved dir is never shared, and the reuse-trap guard refuses a
resume of a claimed job dir. Probe runs now land in the jobs subtree,
get the no-weights cleanup, and are gc-reapable -- the pre-kit probes
accumulated save dirs unbounded.

Probe runs are phase-1/2 / call-site checks: 1 step is legal here
(steps min 1), where the measurement suites keep the floor at 10.
"""

import json
import os
import subprocess
import sys

from ..registry import Param, RunPaths
from ..trainer_kit import TrainerKit, tag_metrics

_PROBES = ("foldcount", "batchshapes")
_MK = "config.process[0].model.model_kwargs"

METRICS = {
    "probe_ok": "1 if artifacts/probe.json was written by the probe",
    "steps_done": "training steps that ran (probe ran to config steps)",
    "probe_keys": "number of distinct observation keys",
    "probe_total": "sum of all observation counts",
}


@tag_metrics("artifacts/probe.json")
def _parse(paths: RunPaths, params):
    pj = os.path.join(paths.artifacts, "probe.json")
    out = {"probe_ok": 0, "steps_done": None, "probe_keys": 0,
           "probe_total": 0}
    if os.path.exists(pj):
        with open(pj) as f:
            data = json.load(f)
        entries = data.get("counters", {})
        out["probe_ok"] = 1
        out["probe_keys"] = len(entries)
        out["probe_total"] = sum(int(v) for v in entries.values())
        out["steps_done"] = data.get("steps_done")
    return out


def _spawn(paths, params, manifest, cfg_path, job_dir):
    # the probe runner installs the named probe in-process, then calls
    # run.main(); --out lands the observation json in the RECORD
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.call(
        [sys.executable, os.path.join(here, "h3_probe_runner.py"),
         "--config", cfg_path,
         "--probe", params["probe"],
         "--out", os.path.join(paths.artifacts, "probe.json")],
        cwd=manifest["cwd"],
    )


_kit = TrainerKit(
    namespace="H3PROBE",
    module="tools.runtool.suites.h3_probe",
    extra_params={
        "probe": Param(
            "enum", required=True, choices=_PROBES,
            desc="instrumentation to install (tools.runtool.suites.probes)",
        ),
        "steps": Param(
            "int", default=3, min=1, max=50,
            desc="call-site checks need 1-3 steps; the floor is 1 here "
                 "because probes measure phases, not throughput",
        ),
        "batch_size": Param("int", default=4, min=1, max=8),
        "ckpt_stride": Param(
            "int", default=None, min=0, max=16,
            desc="NOT WIRED: reserved for the checkpoint-stride experiment "
                 "planned on the second machine. The value is recorded in "
                 "the manifest/params but has no config effect until a "
                 "wired run exists -- it must not look live before it is.",
        ),
        "cutlass_bwd": Param(
            "enum", default="fc2", choices=("off", "fc2", "full"),
            desc="model_kwargs int8_cutlass_bwd; defaults to the code "
                 "default (fc2) so probes see the production call sites; "
                 "pass off explicitly to probe the eager-dX path",
        ),
    },
    switches={
        "cutlass_bwd": _MK + ".int8_cutlass_bwd",
    },
    extra_flags=lambda params: [
        "probe_" + params["probe"],
        *([] if params.get("cutlass_bwd", "fc2") == "fc2"
          else ["cutlass_" + params["cutlass_bwd"]]),
    ],
    parse=_parse,
    spawn_job=_spawn,
)


SUITE = _kit.suite(
    suite_id="h3_probe",
    variants=("v2",),
    metrics=METRICS,
    timeout_s=45 * 60,
    desc="instrumented short training run for call-site measurement",
)


if __name__ == "__main__":
    sys.exit(_kit.main())
