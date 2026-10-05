"""h3_probe suite: short instrumented training runs for call-site
measurement (mm/fold accounting, module census). The training process is
launched through h3_probe_runner, which installs a named probe before
run.main() — no base-code edits. Output is structured: artifacts/probe.json
(one entry per observation key) plus the normal run.log; metrics are
reproduced from the record alone.

Job names are fresh per launch (timestamp tag) — probes must never resume
a previous job: resumed weights/shapes make the inventory meaningless.
"""

import argparse
import json
import os
import sys
import threading
import time

from ..registry import Param, RunPaths, Suite, safe_run_component
from .h3_train import _dataset_keys, _derive_config, _watch_db

HERE = os.path.dirname(os.path.abspath(__file__))

_PROBES = ("foldcount", "batchshapes")


def _params():
    return {
        "base": Param(
            "str", required=True,
            desc="path to the base config (v2-lineage template)",
        ),
        "dataset": Param(
            "enum", required=True, choices=_dataset_keys(),
            desc="dataset abbrev (resolved via datasets.json)",
        ),
        "probe": Param(
            "enum", required=True, choices=_PROBES,
            desc="instrumentation to install (tools.runtool.suites.probes)",
        ),
        "steps": Param("int", default=3, min=2, max=50),
        "batch_size": Param("int", default=4, min=1, max=8),
        "seed": Param("int", default=42, min=0, max=2**31 - 1),
        "ckpt_stride": Param("int", default=None, min=0, max=16),
        "cutlass_bwd": Param(
            "enum", default="off", choices=("off", "fc2", "full"),
            desc="model_kwargs int8_cutlass_bwd (keep default off unless "
                 "the probe targets it)",
        ),
    }


METRICS = {
    "probe_ok": "1 if artifacts/probe.json was written by the probe",
    "steps_done": "training steps that ran (probe ran to config steps)",
    "probe_keys": "number of distinct observation keys",
    "probe_total": "sum of all observation counts",
}


def _build(params, paths: RunPaths):
    argv = [
        sys.executable, "-m", "tools.runtool.suites.h3_probe",
        "--run-dir", paths.root,
    ]
    env = {
        "SEED": params["seed"],
        "H3_VARLEN_ATTN": "1",
        "H3_ROPE_FUSE": "1",
        "H3_LIGER_FUSIONS": "0",
    }
    return argv, env


def _job_name(params):
    parts = [
        "H3PROBE",
        safe_run_component(params["dataset"]),
        safe_run_component(f"bs{params['batch_size']}"),
        safe_run_component(str(params["steps"])),
        safe_run_component(params["probe"]),
        time.strftime("%m%d_%H%M%S"),
    ]
    return "_".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.environ.get("RUNTOOL_REPO", os.getcwd()))
    from ..runner import emit

    paths = RunPaths(run_id=os.path.basename(args.run_dir), root=args.run_dir)
    with open(paths.manifest) as f:
        manifest = json.load(f)
    params = manifest["params"]

    name = _job_name(params)
    cfg_path = os.path.join(paths.artifacts, "config.yaml")
    # h3_train._derive_config reads the shared keys; fill the rest from its
    # registered defaults so the derivation stays one code path
    from .h3_train import _params as h3_params

    derived = {k: params.get(k, spec.default) for k, spec in h3_params().items()}
    derived.update(
        base=params["base"], dataset=params["dataset"],
        steps=params["steps"], batch_size=params["batch_size"],
        seed=params["seed"], ckpt_stride=params["ckpt_stride"],
        cutlass_bwd=params["cutlass_bwd"], tag=f"probe_{params['probe']}",
    )
    cfg = _derive_config(derived, name, cfg_path)
    folder = "output"
    for proc in cfg["config"].get("process", []):
        folder = proc.get("training_folder", folder)
    job_dir = os.path.join(folder, name)
    emit(paths, "harness_start", job=name, job_dir=job_dir, cfg=cfg_path)

    log_path = os.path.join(paths.artifacts, "run.log")
    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch_db, args=(job_dir, paths, stop), daemon=True
    )
    watcher.start()

    import subprocess

    rc = subprocess.call(
        [
            sys.executable, os.path.join(HERE, "h3_probe_runner.py"),
            "--config", cfg_path,
            "--probe", params["probe"],
            "--out", os.path.join(paths.artifacts, "probe.json"),
        ],
        cwd=manifest["cwd"],
    )
    stop.set()
    if rc != 0:
        emit(paths, "stage_failed", rc=rc)
        return rc
    metrics = SUITE.parse(paths, params)
    with open(paths.result + ".tmp", "w") as f:
        json.dump(
            {"ok": True, "metrics": metrics, "job_name": name,
             "job_dir": job_dir, "metric_source": "artifacts/probe.json"},
            f, indent=1,
        )
    os.replace(paths.result + ".tmp", paths.result)
    emit(paths, "harness_done", metrics=metrics)
    return 0


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


SUITE = Suite(
    id="h3_probe",
    variants=("v2",),
    params=_params(),
    metrics=METRICS,
    build=_build,
    parse=_parse,
    timeout_s=45 * 60,
    desc="instrumented short training run for call-site measurement",
)


if __name__ == "__main__":
    sys.exit(main())
