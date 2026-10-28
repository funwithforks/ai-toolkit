"""moss_train suite: MOSS-Image training-step measurement.

Same contract as flux3_train/ltx2_train: derive a job config from the
(caller-supplied) owner base template changing ONLY name/steps/batch/
sampling/ckpt/save, run run.py, and watch the job's loss_log.db
(steps.wall_time) for progress. Every metric comes from the db; tqdm
output stays in the logs. The trainer is extensions_built_in/sd_trainer
(uid diffusion_trainer), so loss_log.db + AITK_STEP_TIME + the
AITK_PROF_* window are live for this model too.

Sampling standard (owner, 2026-10-24): the owner template's sample block
is ComfyUI REMOTE sampling (comfy/comfy_url). It stays inert on every
measurement run (sample=false, the default) and is kept LIVE on the long
quality-test runs (sample=true) -- the runtool standard for this model:
speed data never pays for sampling, quality runs are judged by sampling.
With sample=true the template's save block is also kept as the owner set
it (step-saves + final LoRA: the artifacts the ComfyUI gate reads);
measurement runs never write safetensors.

One performance switch is exposed: varlen (model switch
MOSS_VARLEN_ATTN, read by flash_varlen.varlen_ok) -- packed flash-varlen
attention is the DEFAULT for padded bs>1 t2i batches; varlen=false
restores the legacy masked mem-efficient path for A/B measurement. Any
further param appears here only once its model-level switch exists and
is measured on THIS model (per-model rollout rule -- krea2/H3/LTX/flux3
verdicts are not inherited, only their measurement discipline).

Reuse-trap guard (H3 incident: a rerun under a finished job's name
resumed and "completed" instantly): the harness refuses to launch when
output/<job_name>/optimizer.pt already exists, failing the record with
a clear error instead of reporting a resumed run as fresh metrics.

Job naming carries the dataset abbreviation (owner rule):
MOSS_<dataset>_bs<batch>_<steps>[_<tag>].

Dataset folder paths never live in tracked code: they are resolved at
run time from datasets.json (gitignored, output/runtool/).
"""

import argparse
import json
import os
import sqlite3
import sys
import threading

from .. import datasets as dsets
from .. import markers as dmark
from ..registry import (
    Param, ParamError, RunPaths, Suite, safe_run_component,
)

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
        "steps": Param("int", default=60, min=10, max=5000),
        "batch_size": Param(
            "int", default=1, min=1, max=32,
            desc="train batch at 1024 res; fit is machine-dependent "
                 "(measured marginal ~3.3-3.7 GiB/sample over resident), "
                 "high values are deliberate fit probes",
        ),
        "warmup": Param(
            "int", default=20, min=0, max=2000,
            desc="metrics window starts after this step",
        ),
        "seed": Param("int", default=42, min=0, max=2**31 - 1),
        "ckpt": Param(
            "bool", default=True,
            desc="train.gradient_checkpointing (per-block recompute on the "
                 "42-block DiT; off saves the recompute, fit unmeasured)",
        ),
        "sample": Param(
            "bool", default=False,
            desc="keep the owner template's ComfyUI sampling block LIVE "
                 "(quality runs). Default false = sampling inert, the "
                 "measurement-run standard; also unlocks the template's "
                 "save block (samples + step-saves are the ComfyUI gate)",
        ),
        "varlen": Param(
            "bool", default=True,
            desc="packed flash-varlen attention for padded batches (model "
                 "switch MOSS_VARLEN_ATTN, the standard for bs>1); false "
                 "restores the legacy masked mem-efficient path for A/B",
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
                 "profiling is deliberately not exposed -- host-OOM rule)",
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
        "MOSS",
        safe_run_component(params["dataset"]),
        safe_run_component(f"bs{params['batch_size']}"),
        safe_run_component(str(params["steps"])),
    ]
    if params.get("tag"):
        parts.append(safe_run_component(params["tag"]))
    return "_".join(parts)


def _build(params, paths: RunPaths):
    argv = [
        sys.executable, "-m", "tools.runtool.suites.moss_train",
        "--run-dir", paths.root,
    ]
    env = {
        "SEED": params["seed"],
        "AITK_STEP_TIME": "1",
        # the packed-varlen model switch (flash_varlen.varlen_ok reads the
        # env; default on). A/B: varlen=false reruns the masked reference.
        "MOSS_VARLEN_ATTN": "1" if params.get("varlen", True) else "0",
    }
    if params.get("prof_start") is not None:
        # the SDTrainer profiler window (no-stack) exports a chrome trace;
        # AITK_PROF_STACK is never set (host OOM incident rule: stacks
        # capped at 1 step, not exposed)
        prof_dir = os.path.join(paths.artifacts, "prof")
        env["AITK_PROF_DIR"] = prof_dir
        env["AITK_PROF_START"] = params["prof_start"]
        env["AITK_PROF_ACTIVE"] = params["prof_active"]
        os.makedirs(prof_dir, exist_ok=True)
    return argv, env


# ------------------------------------------------------------- harness

def _load_map(dataset):
    try:
        with open(DATASET_MAP) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ParamError(
            f"dataset registry {DATASET_MAP} does not exist; register "
            f"folders with 'runtool datasets --add KEY --folder PATH' or "
            f"create a tool dataset with 'runtool datasets --create KEY "
            f"--synthetic N'"
        )
    if dataset not in data:
        raise ParamError(
            f"dataset {dataset!r} not in the registry; "
            f"keys: {sorted(data)}"
        )
    return data[dataset]["folders"]


def _tool_dataset_check(dataset, folders):
    # datasets are always tool-owned (owner policy 2026-10-28): every tool
    # run trains against a synthetic or COPIED rt_-prefixed folder, never a
    # registered real dataset folder -- real folders stay read-only to the
    # tool (copy source via 'datasets --create --from-key' is the one use).
    # The refusal names only the KEY: a real folder's basename must never
    # reach agent-facing output, errors included.
    root = os.path.realpath(dsets.DATASETS_ROOT) + os.sep
    for folder in folders:
        rp = os.path.realpath(folder)
        if not (rp.startswith(root)
                and os.path.basename(rp).startswith(dsets.TOOL_PREFIX)):
            raise ParamError(
                f"dataset '{dataset}' points outside the tool dataset "
                f"namespace; make a tool dataset with 'runtool datasets "
                f"--create K --synthetic N' (synthetic) or '--create K "
                f"--from-key {dataset}' (copy, caches optional via "
                f"--copy-cache)"
            )


def _datasets_precheck(params):
    # submit-process rejection (runner.submit -> Suite.precheck): a bad
    # dataset never becomes a doomed launch. The harness re-runs the same
    # check in _derive_config so a direct harness call cannot skip it.
    _tool_dataset_check(
        params["dataset"], _load_map(params["dataset"]))


def _derive_config(params, name, out_path):
    import yaml

    with open(params["base"]) as f:
        cfg = yaml.safe_load(f)
    # Owner-template rule: change ONLY name, steps, batch_size, sampling
    # off, ckpt, save pin. Anything else requires owner sign-off.
    cfg["config"]["name"] = name
    cfg["name"] = name
    proc = cfg["config"]["process"][0]
    train = proc["train"]
    train["batch_size"] = int(params["batch_size"])
    train["steps"] = int(params["steps"])
    # Sampling/save policy follows the owner's 2026-10-24 standard:
    # measurement runs (sample=false, default) run the sample block inert
    # (comfy remote included) and never write safetensors -- save pinned
    # past the last step (save params live in the save: block,
    # toolkit/config_modules.py:25). Quality runs (sample=true) keep BOTH
    # blocks exactly as the owner template has them.
    sample = bool(params.get("sample", False))
    train["disable_sampling"] = not sample
    train["gradient_checkpointing"] = bool(params["ckpt"])
    # no model_kwargs switches yet: MOSS wires no perf knobs (per-model
    # rollout rule); the block ships as the owner set it
    if not sample:
        save = proc.setdefault("save", {})
        save["save_every"] = int(params["steps"]) + 1
        save["max_step_saves_to_keep"] = 1
    folders = _load_map(params["dataset"])
    ds = proc["datasets"]
    if len(folders) != len(ds):
        raise SystemExit(
            f"datasets.json gives {len(folders)} folders for "
            f"{params['dataset']!r}, template has {len(ds)} dataset entries"
        )
    for entry, folder in zip(ds, folders):
        entry["folder_path"] = folder
    # datasets are always tool-owned (owner policy 2026-10-28). The same
    # check runs in the submit process via SUITE.precheck; this copy guards
    # the harness path so nothing can train on a non-tool folder
    _tool_dataset_check(params["dataset"], folders)
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
    cost). Emits peak plus a compact summary to events.jsonl on exit.
    The bs8 fit boundary makes peak the headline metric on this model."""
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

    # reuse-trap guard: a finished/partial job under this name makes the
    # trainer RESUME (optimizer.pt, BaseSDTrainProcess), i.e. a bogus
    # fast "done" -- fail the record loudly instead
    if os.path.exists(os.path.join(job_dir, "optimizer.pt")):
        emit(paths, "reuse_trap", job=name, job_dir=job_dir)
        with open(paths.result + ".tmp", "w") as f:
            json.dump(
                {"ok": False,
                 "error": f"reuse trap: {job_dir} holds a previous run "
                          f"(optimizer.pt); the trainer would resume it. "
                          f"Delete the job dir or change the tag."},
                f, indent=1,
            )
        os.replace(paths.result + ".tmp", paths.result)
        emit(paths, "stage_failed", rc=1, reason="reuse_trap")
        return 1

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

    # measurement runs must leave no weights on disk (owner hard disk
    # rule): the toolkit's end-of-training save is UNCONDITIONAL
    # (BaseSDTrainProcess.py:3055 -- the save_every pin only defers
    # in-run step saves), so the harness deletes what it just made.
    # Metrics live in the db snapshot above; quality runs (sample=true)
    # keep their saves because the saves ARE the deliverable.
    removed = []
    if not params.get("sample", False) and os.path.isdir(job_dir):
        for fname in sorted(os.listdir(job_dir)):
            if fname.endswith(".safetensors") or fname == "optimizer.pt":
                os.remove(os.path.join(job_dir, fname))
                removed.append(fname)
        if removed:
            emit(paths, "artifact_cleanup", removed=removed)

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


def _moss_phases(paths, suite, full):
    # phase fold lives with the marker machinery (markers.py); this is the
    # suite's declaration that it wants the phased-loading summary
    return dmark.moss_phases(paths, suite, full)


SUITE = Suite(
    id="moss_train",
    variants=("v1",),
    params=_params(),
    metrics=METRICS,
    build=_build,
    parse=_parse,
    precheck=_datasets_precheck,
    # marker lines live in the trainer's prints (SDTrainer.py,
    # dataloader_mixins.py, model phase code); these patterns were taken
    # from real records, do not "prettify" them without re-checking
    markers={
        "te_skip": r"Skipping text encoder load",
        "te_load": r"Loading \S+ text encoder from",
        "te_reason": r" - text encoder needed: (?P<reason>.+)",
        "fixed_cache": r"Loading fixed prompt embeddings from cache",
        "fixed_cache_now": r"Caching fixed prompt embeddings",
        "plan": (r"Batched caption cache: (?P<planned>\d+) captions "
                 r"-> (?P<groups>\d+) groups"),
        "vae_round": r"VAE unloaded until next sample round",
    },
    phases=_moss_phases,
    timeout_s=12 * 3600,  # quality runs (1400 steps at bs8 + sampling) pass 6 h
    desc="MOSS-Image training-step measurement (run.py, sqlite-structured metrics)",
)


if __name__ == "__main__":
    sys.exit(main())
