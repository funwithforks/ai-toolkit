"""h3_train suite: full training-step measurement via run.py.

The stage command is fixed: derive a job config from the (caller-supplied)
v2-lineage base template, run run.py with structured per-step timing on,
and watch the job's sqlite loss_log.db (steps.wall_time) for progress.
tqdm output stays in the logs; every metric comes from the db.

Job naming follows the owner rule and always carries the dataset
abbreviation: H3_<dataset>_bs<batch>_<steps>[_<tag>].

Dataset folder paths never live in tracked code: they are resolved at run
time from datasets.json (gitignored, output/runtool/) (untracked) via the dataset abbrev param.
"""

import argparse
import json
import os
import sqlite3
import sys
import threading
import time

from ..registry import Param, RunPaths, Suite, safe_run_component

HERE = os.path.dirname(os.path.abspath(__file__))
# dataset abbrev -> real folder paths. Kept OUT of the tracked tree
# (informal dataset names must never enter git). output/ is gitignored.
DATASET_MAP = os.environ.get(
    "RUNTOOL_DATASETS",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
        "output", "runtool", "datasets.json",
    ),
)


def _dataset_keys():
    """Dataset abbrevs offered to the caller, resolved from the local
    (gitignored) map at import time; empty when unconfigured."""
    try:
        with open(DATASET_MAP) as f:
            return tuple(json.load(f))
    except (OSError, ValueError):
        return ()


def _params():
    return {
        "base": Param(
            "str", required=True,
            desc="path to the base config (v2-lineage template)",
        ),
        "dataset": Param(
            "enum", required=True, choices=_dataset_keys(),
            desc="dataset abbrev (names the job; maps to folders in "
                 "datasets.json (gitignored, output/runtool/))",
        ),
        "steps": Param("int", default=120, min=10, max=5000),
        "batch_size": Param("int", default=1, min=1, max=8),
        "warmup": Param(
            "int", default=20, min=0, max=2000,
            desc="metrics window starts after this step",
        ),
        "seed": Param("int", default=42, min=0, max=2**31 - 1),
        "varlen": Param("bool", default=True, desc="H3_VARLEN_ATTN"),
        "rope": Param("bool", default=True, desc="H3_ROPE_FUSE"),
        "adaln": Param("bool", default=True, desc="H3_ADALN_FUSE"),
        "liger": Param("bool", default=False, desc="H3_LIGER_FUSIONS"),
        "cutlass_bwd": Param(
            "enum", default="off", choices=("off", "fc2", "full"),
            desc="int8_cutlass_bwd model kwarg: CUTLASS NT dX arm on "
                 "resident K-major operand copies (fc2 = fc2-class layers; "
                 "full = also fc1-class, slower tile at live m). Untimed "
                 "experimental path; needs copy headroom (96 GB card).",
        ),
        "tag": Param("str", default="", desc="extra job-name tag"),
        "prof_start": Param(
            "int", default=None, min=5, max=5000,
            desc="enable the torch-profiler window (no-stack) starting at "
                 "this step; artifacts land in record/artifacts/prof/",
        ),
        "prof_active": Param(
            "int", default=2, min=1, max=3,
            desc="profiler window length in steps (hard cap 3; stack "
                 "profiling is deliberately not exposed)",
        ),
    }


METRICS = {
    "step_ms_window": f"mean ms/step over (warmup, last]",
    "step_ms_last100": "mean ms/step over the final <=100 steps",
    "loss_final": "last logged total loss",
    "loss_mean_last10": "mean total loss over the last 10 logged steps",
    "steps_done": "last logged step",
    "wall_s": "db wall-time span first->last step",
}


def _job_name(params):
    parts = [
        "H3",
        safe_run_component(params["dataset"]),
        safe_run_component(f"bs{params['batch_size']}"),
        safe_run_component(str(params["steps"])),
    ]
    if params.get("tag"):
        parts.append(safe_run_component(params["tag"]))
    return "_".join(parts)


def _build(params, paths: RunPaths):
    argv = [
        sys.executable, "-m", "tools.runtool.suites.h3_train",
        "--run-dir", paths.root,
    ]
    env = {
        "SEED": params["seed"],
        "AITK_STEP_TIME": "1",
        "H3_VARLEN_ATTN": "1" if params["varlen"] else "0",
        "H3_ROPE_FUSE": "1" if params["rope"] else "0",
        "H3_LIGER_FUSIONS": "1" if params["liger"] else "0",
    }
    if params.get("prof_start") is not None:
        # the SDTrainer profiler window (no-stack) exports a chrome trace +
        # kernel-avg tables into the record; AITK_PROF_STACK is never set
        prof_dir = os.path.join(paths.artifacts, "prof")
        env["AITK_PROF_DIR"] = prof_dir
        env["AITK_PROF_START"] = params["prof_start"]
        env["AITK_PROF_ACTIVE"] = params["prof_active"]
        os.makedirs(prof_dir, exist_ok=True)
    return argv, env


# ------------------------------------------------------------- harness

def _load_map(dataset):
    with open(DATASET_MAP) as f:
        data = json.load(f)
    if dataset not in data:
        raise SystemExit(
            f"dataset {dataset!r} not in {DATASET_MAP}; add the folder list"
        )
    return data[dataset]["folders"]


def _derive_config(params, name, out_path):
    import yaml

    with open(params["base"]) as f:
        cfg = yaml.safe_load(f)
    # template shape: job: <str>; config: {name, process: [{training_folder,
    # network, datasets, train, model, sample}]}
    cfg["config"]["name"] = name
    cfg["name"] = name
    proc = cfg["config"]["process"][0]
    train = proc["train"]
    train["batch_size"] = int(params["batch_size"])
    train["steps"] = int(params["steps"])
    # toolkit sampling is never used for benchmarking (owner rule); the
    # sample block stays inert
    train["disable_sampling"] = True
    mm = proc["model"].setdefault("model_kwargs", {})
    mm["int8_cutlass_bwd"] = params["cutlass_bwd"]
    folders = _load_map(params["dataset"])
    ds = proc["datasets"]
    if len(folders) != len(ds):
        raise SystemExit(
            f"datasets.json gives {len(folders)} folders for "
            f"{params['dataset']!r}, template has {len(ds)} dataset entries"
        )
    for entry, folder in zip(ds, folders):
        entry["folder_path"] = folder
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg


def _watch_db(job_dir, paths: RunPaths, stop: threading.Event):
    db = os.path.join(job_dir, "loss_log.db")
    while not stop.wait(20.0):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1)
            row = con.execute("select max(step) from steps").fetchone()
            con.close()
            if row and row[0]:
                from ..runner import emit

                emit(paths, "progress", step=row[0])
        except Exception:
            pass  # db not created / mid-transaction; retry next tick


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.environ.get("RUNTOOL_REPO", os.getcwd()))
    from ..runner import emit  # noqa: E402  (repo root on path via -m)

    paths = RunPaths(run_id=os.path.basename(args.run_dir), root=args.run_dir)
    with open(paths.manifest) as f:
        manifest = json.load(f)
    params = manifest["params"]

    name = _job_name(params)
    cfg_path = os.path.join(paths.artifacts, "config.yaml")
    cfg = _derive_config(params, name, cfg_path)
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
        [sys.executable, "run.py", cfg_path, "--log", log_path],
        cwd=manifest["cwd"],
    )
    stop.set()
    if rc != 0:
        # no result file -> the runner reports crash with the exit code
        emit(paths, "stage_failed", rc=rc)
        return rc

    # snapshot the structured source into the record so metrics are
    # recomputable from the record alone (contract)
    src_db = os.path.join(job_dir, "loss_log.db")
    if os.path.exists(src_db):
        import shutil

        shutil.copy2(src_db, os.path.join(paths.artifacts, "metrics.db"))

    metrics = SUITE.parse(paths, params)
    with open(paths.result + ".tmp", "w") as f:
        json.dump(
            {"ok": True, "metrics": metrics, "job_name": name,
             "job_dir": job_dir, "metric_source": "steps.wall_time (db) + loss/loss"},
            f, indent=1,
        )
    os.replace(paths.result + ".tmp", paths.result)
    emit(paths, "harness_done", metrics=metrics)
    return 0


# -------------------------------------------------------------- parser

def _parse(paths: RunPaths, params):
    db = os.path.join(paths.artifacts, "metrics.db")
    if not os.path.exists(db):
        with open(paths.result) as f:
            job_dir = json.load(f).get("job_dir")
        db = os.path.join(job_dir, "loss_log.db")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        wall = dict(con.execute("select step, wall_time from steps"))
        w = int(params["warmup"])
        last = max(wall) if wall else 0

        def win(a, b):
            if a in wall and b in wall and b > a:
                return round((wall[b] - wall[a]) / (b - a) * 1000, 1)
            return None

        first = min(wall) if wall else 0
        losses = con.execute(
            "select step, value_real from metrics where key='loss/loss' "
            "and value_real is not null order by step"
        ).fetchall()
        out = {
            "step_ms_window": win(w, last),
            "step_ms_last100": win(max(w, last - 100), last),
            "loss_final": losses[-1][1] if losses else None,
            "loss_mean_last10": (
                round(sum(v for _, v in losses[-10:]) / min(10, len(losses)), 5)
                if losses else None
            ),
            "steps_done": last,
            "wall_s": round(wall[last] - wall[first], 1) if wall else None,
        }
        return out
    finally:
        con.close()


SUITE = Suite(
    id="h3_train",
    variants=("v2",),
    params=_params(),
    metrics=METRICS,
    build=_build,
    parse=_parse,
    timeout_s=4 * 3600,
    desc="H3 training-step measurement (run.py, sqlite-structured metrics)",
)


if __name__ == "__main__":
    sys.exit(main())
