"""ltx2_train suite: LTX-2.5 training-step measurement via run.py.

Same contract as h3_train: derive a job config from the (caller-supplied)
owner base template changing ONLY name/steps/batch/sampling/ckpt, run
run.py, and watch the job's sqlite loss_log.db (steps.wall_time) for
progress. Every metric comes from the db; tqdm output stays in the logs.

No performance switches are exposed: LTX-2.5 has none wired yet in
ltx2.py. A param appears here only once its model-level switch exists
and is measured on this model (per-model rollout rule -- H3/krea2
verdicts are not inherited).

Job naming carries the dataset abbreviation (owner rule):
LTX_<dataset>_bs<batch>_<steps>[_<tag>].

Dataset folder paths never live in tracked code: they are resolved at
run time from datasets.json (gitignored, output/runtool/).
"""

import argparse
import json
import os
import sqlite3
import sys
import threading

from ..registry import Param, RunPaths, Suite, safe_run_component

HERE = os.path.dirname(os.path.abspath(__file__))
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
            desc="path to the base config (owner template, stored verbatim)",
        ),
        "dataset": Param(
            "enum", required=True, choices=_dataset_keys(),
            desc="dataset abbrev (names the job; maps to folders in "
                 "datasets.json (gitignored, output/runtool/))",
        ),
        "steps": Param("int", default=200, min=10, max=5000),
        "batch_size": Param("int", default=1, min=1, max=8),
        "warmup": Param(
            "int", default=20, min=0, max=2000,
            desc="metrics window starts after this step",
        ),
        "seed": Param("int", default=42, min=0, max=2**31 - 1),
        "ckpt": Param(
            "bool", default=True,
            desc="train.gradient_checkpointing (transformer recompute; "
                 "off saves the recompute but raises activation memory)",
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
        "LTX",
        safe_run_component(params["dataset"]),
        safe_run_component(f"bs{params['batch_size']}"),
        safe_run_component(str(params["steps"])),
    ]
    if params.get("tag"):
        parts.append(safe_run_component(params["tag"]))
    return "_".join(parts)


def _build(params, paths: RunPaths):
    argv = [
        sys.executable, "-m", "tools.runtool.suites.ltx2_train",
        "--run-dir", paths.root,
    ]
    env = {
        "SEED": params["seed"],
        "AITK_STEP_TIME": "1",
    }
    if params.get("prof_start") is not None:
        # the SDTrainer profiler window (no-stack) exports a chrome trace +
        # kernel-avg tables into the record; AITK_PROF_STACK is never set
        # (host OOM incident rule: stacks capped at 1 step, not exposed)
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
    # Owner-template rule: change ONLY name, steps, batch_size, sampling
    # off, ckpt. Anything else requires owner sign-off.
    cfg["config"]["name"] = name
    cfg["name"] = name
    proc = cfg["config"]["process"][0]
    train = proc["train"]
    train["batch_size"] = int(params["batch_size"])
    train["steps"] = int(params["steps"])
    # toolkit sampling is never used for benchmarking (owner rule); the
    # sample block stays inert
    train["disable_sampling"] = True
    train["gradient_checkpointing"] = bool(params["ckpt"])
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


def _watch_mem(paths: RunPaths, stop: threading.Event, interval=1.0):
    """1 Hz used-mem sampler (nvidia-smi, external = zero in-process
    cost). Closes the runtool blind spot: no memory metric. Emits peak
    plus a compact histogram to events.jsonl on exit."""
    import subprocess

    peak = 0
    samples = []
    while not stop.wait(interval):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                text=True,
            )
            used = max(int(x) for x in out.split())
        except Exception:
            continue
        if used > peak:
            peak = used
        samples.append(used)
        if len(samples) >= 5 and len(samples) % 30 == 0:
            from ..runner import emit

            emit(paths, "mem", used=used, peak=peak)
    if samples:
        from ..runner import emit

        tail = samples[-60:]
        emit(
            paths, "mem_summary", peak=peak,
            tail_median=sorted(tail)[len(tail) // 2], tail_max=max(tail),
        )


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
    mem_watcher = threading.Thread(
        target=_watch_mem, args=(paths, stop), daemon=True
    )
    mem_watcher.start()

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
    id="ltx2_train",
    variants=("v1",),
    params=_params(),
    metrics=METRICS,
    build=_build,
    parse=_parse,
    timeout_s=6 * 3600,
    desc="LTX-2.5 training-step measurement (run.py, sqlite-structured metrics)",
)


if __name__ == "__main__":
    sys.exit(main())
