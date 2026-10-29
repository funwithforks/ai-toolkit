"""trainer_kit: the generic machinery every ai-toolkit trainer suite shares.

Owner ruling (2026-10-28): pull the generic items out of suites. The
option of sampling is generic; the run.py harness, the loss_log.db
metrics, the loud merge over the trainer config layout, the dataset
guards, the watchers, the reuse-trap and the measurement-never-saves
cleanup are generic. A suite module keeps ONLY what is specific to the
model it supports: namespace, model switches/params/env, model-specific
markers, timeout, description.

moss_train is the reference consumer (and for now the only one; the
older suites keep their private copies until they are migrated after
the tool settles -- no backward-compat obligation to them here).

Config layout assumption (the ai-toolkit trainer schema): a job config
of config.process[N].{train,save,sample,datasets, model}, the process
entry addressed by `prefix` (default config.process[0]).

Harness contract (generic main): manifest -> derive (loud merge) ->
reuse-trap guard -> run.py -> db snapshot -> measurement cleanup ->
result.json. The suite module's __main__ is one call: kit.main(argv).
"""

import json
import os
import sqlite3
import subprocess
import sys
import threading

from . import datasets as dsets
from . import dotpath
from . import markers as dmark
from .registry import Param, ParamError, RunPaths, Suite, safe_run_component

# ---------------------------------------------------------------- datasets

def dataset_keys():
    """Dataset keys offered to the caller, from the local registry at
    import time (empty when unconfigured; registry is machine-local)."""
    try:
        return tuple(dsets._load())
    except ParamError:
        return ()


def load_map(dataset):
    try:
        data = dsets._load()
    except FileNotFoundError:
        raise ParamError(
            f"dataset registry {dsets.REGISTRY_PATH} does not exist; register "
            f"folders with 'runtool datasets --add KEY --folder PATH' or "
            f"create a tool dataset with 'runtool datasets --create KEY "
            f"--synthetic N'"
        )
    if dataset not in data:
        raise ParamError(f"dataset {dataset!r} not in the registry; "
                         f"keys: {sorted(data)}")
    return data[dataset]["folders"]


def tool_dataset_check(dataset, folders):
    # datasets are always tool-owned (owner policy 2026-10-28): every tool
    # run trains against a synthetic or COPIED rt_-prefixed folder, never a
    # registered real dataset folder -- real folders stay read-only to the
    # tool (copy source via 'datasets --create --from-key' is the one use).
    # The refusal names only the KEY: a real folder's basename must never
    # reach agent-facing output, errors included.
    root = os.path.realpath(dsets.DATASETS_ROOT) + os.sep
    for folder in folders:
        rp = os.path.realpath(folder)
        # ownership = first path segment under the tool root is rt_-
        # prefixed (create() may keep folder structure inside it);
        # a manually-placed real folder never qualifies.
        owned = rp.startswith(root) and os.path.relpath(
            rp, os.path.realpath(dsets.DATASETS_ROOT)).split(
                os.sep)[0].startswith(dsets.TOOL_PREFIX)
        if not owned:
            raise ParamError(
                f"dataset '{dataset}' points outside the tool dataset "
                f"namespace; make a tool dataset with 'runtool datasets "
                f"--create K --synthetic N' (synthetic) or '--create K "
                f"--from-key {dataset}' (copy, caches optional via "
                f"--copy-cache)"
            )


# ------------------------------------------------------------------ params

def base_params(**extra):
    """Params every trainer suite carries. Model switches are extras."""
    p = {
        "base": Param(
            "str", required=True,
            desc="path to the base config (owner template, stored verbatim)",
        ),
        "dataset": Param(
            "enum", required=True, choices=dataset_keys(),
            desc="dataset key (names the job; maps to folders in the "
                 "machine-local registry)",
        ),
        "steps": Param("int", default=60, min=10, max=5000),
        "batch_size": Param(
            "int", default=1, min=1, max=32,
            desc="train batch; fit is machine-dependent, high values are "
                 "deliberate fit probes",
        ),
        "warmup": Param(
            "int", default=20, min=0, max=2000,
            desc="metrics window starts after this step",
        ),
        "seed": Param("int", default=42, min=0, max=2**31 - 1),
        "kind": Param(
            "enum", default="lora", choices=("lora", "full"),
            desc="training kind carried in the job name",
        ),
        "ckpt": Param(
            "bool", default=True,
            desc="train.gradient_checkpointing (recompute saves memory, "
                 "costs time)",
        ),
        "sample": Param(
            "bool", default=False,
            desc="keep the owner template's sampling block LIVE (quality "
                 "runs). Default false = sampling inert, the measurement-"
                 "run standard; also unlocks the template's save block",
        ),
        "tag": Param("str", default="", desc="extra job-name tag"),
        "prof_start": Param(
            "int", default=None, min=5, max=5000,
            desc="enable the SDTrainer profiler window (no-stack) starting "
                 "at this step; artifacts land in record/artifacts/prof/",
        ),
        "prof_active": Param(
            "int", default=2, min=1, max=3,
            desc="profiler window length in steps (hard cap 3; stack "
                 "profiling is deliberately not exposed -- host-OOM rule)",
        ),
    }
    p.update(extra)
    return p


METRICS = {
    "step_ms_window": "mean ms/step over (warmup, last]",
    "step_ms_last100": "mean ms/step over the final <=100 steps",
    "loss_final": "last logged total loss",
    "loss_mean_last10": "mean total loss over the last 10 logged steps",
    "steps_done": "last logged step",
    "wall_s": "db wall-time span first->last step",
}

# marker lines the SHARED trainer code prints (SDTrainer.py,
# dataloader_mixins.py); patterns were taken from real records -- do not
# "prettify" them without re-checking. Model-specific markers are extras.
def tag_metrics(src):
    """Decorator: a parse function states where its numbers come from;
    result.json carries the label (a custom parser that skips this is
    silently mislabeled -- the label is the contract)."""
    def deco(fn):
        fn.metric_source = src
        return fn
    return deco


TRAINER_MARKERS = {
    "te_skip": r"Skipping text encoder load",
    "te_load": r"Loading \S+ text encoder from",
    "te_reason": r" - text encoder needed: (?P<reason>.+)",
    "fixed_cache": r"Loading fixed prompt embeddings from cache",
    "fixed_cache_now": r"Caching fixed prompt embeddings",
    "plan": (r"Batched caption cache: (?P<planned>\d+) captions "
             r"-> (?P<groups>\d+) groups"),
    "vae_round": r"VAE unloaded until next sample round",
}


def job_name_base(namespace):
    """Approved naming format (owner 2026-10-28; agent-utility is the
    bar -- the name carries what the AGENT varied, not what the
    template implies):

      <NAMESPACE>_<kind>_<dataset>_bs<N>[_<flags>...][_purpose]_v<N>

    flags are derived: nockpt (ckpt off), samp (sample on), dop (a
    caption-dropout patch > 0), plus suite extras (e.g. novarlen).
    purpose is the tag. The version is appended by reservation, never
    guessed here."""
    def build(params, ctx=None):
        ctx = ctx or {}
        parts = [
            namespace,
            safe_run_component(params.get("kind") or "lora"),
            safe_run_component(params["dataset"]),
            safe_run_component(f"bs{params['batch_size']}"),
        ]
        if not bool(params.get("ckpt", True)):
            parts.append("nockpt")
        if bool(params.get("sample", False)):
            parts.append("samp")
        for path, value in (ctx.get("config_patch") or {}).items():
            if "caption_dropout_rate" in path and float(value or 0) > 0:
                parts.append("dop")
                break
        for flag in (ctx.get("extra_flags") or ()):
            parts.append(safe_run_component(flag))
        if params.get("tag"):
            parts.append(safe_run_component(params["tag"]))
        return "_".join(parts)
    return build


# ------------------------------------------------------------- loud merge

def param_fields(prefix):
    """params that own trainer-config fields (the ai-toolkit layout).
    Fields listed here are PARAM-OWNED: config_patch may never claim
    them -- patch-vs-param claims on one field are impossible by
    construction."""
    train = f"{prefix}.train"
    return {
        "batch_size": (train + ".batch_size", lambda p: int(p["batch_size"])),
        "steps": (train + ".steps", lambda p: int(p["steps"])),
        "ckpt": (train + ".gradient_checkpointing", lambda p: bool(p["ckpt"])),
        "sample": (train + ".disable_sampling", lambda p: not bool(p["sample"])),
    }


def patchable(prefix):
    """the config vocabulary every trainer suite exposes to
    config_patch: knobs no suite param owns"""
    train = f"{prefix}.train"
    return (
        train + ".lr",
        train + ".optimizer",
        train + ".gradient_accumulation",
        train + ".cache_text_embeddings",
        train + ".unload_text_encoder",
        train + ".dtype",
        f"{prefix}.sample.sample_every",
        f"{prefix}.sample.comfy",
        f"{prefix}.sample.seed",
        f"{prefix}.datasets[0].caption_dropout_rate",
        f"{prefix}.datasets[0].num_repeats",
    )


def merge_plan(params, overrides, config_patch, name, *, prefix,
               patchable_paths, fields=None, jobs_root=None, switches=None):
    """Derive the run config from base template + suite params +
    config_patch, with a provenance row for every field any layer
    touched. Raises ParamError listing EVERY unresolved conflict:
    a defaulted param disagreeing with an explicit template value is
    refused -- the silent sample-disable incident is structurally
    impossible. Explicit params win over templates and are logged.

    switches maps suite param -> dotted config path (e.g. the
    model_kwargs entries the model actually reads). Switch fields are
    PARAM-OWNED like the train.* fields: they carry full loud-merge
    semantics (default vs template conflict refused) and config_patch
    can never claim them. This is how model kwargs become trackable:
    every value lands in artifacts/config.yaml, the provenance file
    and the submit echo -- never in a side channel."""
    import yaml

    try:
        with open(params["base"]) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise ParamError(f"base template {params['base']!r} does not exist")
    except yaml.YAMLError as e:
        raise ParamError(
            f"base template {params['base']!r} is not valid YAML: {e}")

    fields = fields or param_fields(prefix)
    switches = dict(switches or {})
    owned = {path: pname for pname, (path, _) in fields.items()}
    owned.update({path: pname for pname, path in switches.items()})
    overrides = set(overrides or ())
    config_patch = dict(config_patch or {})
    prov, conflicts = [], []

    # -- param layer (owned fields + model switches) ----------------------
    items = [(pn, path, pick, "param")
             for pn, (path, pick) in fields.items()]
    items += [(pn, path, (lambda p, _n=pn: p[_n]), "param-switch")
              for pn, path in sorted(switches.items())]
    for pname, path, pick, tag in items:
        segs = dotpath.parse(path)
        value = pick(params)
        template_v = dotpath.get(cfg, segs)
        if pname in overrides:
            note = {"path": path, "source": tag, "value": value}
            if template_v is not dotpath.MISSING and template_v != value:
                note["against_template"] = template_v
            dotpath.setval(cfg, segs, value)
            prov.append(note)
        else:
            if template_v is not dotpath.MISSING and template_v != value:
                if pname == "sample":
                    opts = (f"pass sample={bool(params['sample'])} to "
                            f"override the template, or "
                            f"sample={not bool(params['sample'])} to keep it")
                else:
                    opts = (f"pass {pname}={params.get(pname)!r} "
                            f"explicitly to override the template")
                conflicts.append(
                    f"{path}: template sets {template_v!r} but suite param "
                    f"{pname!r} (default) would set {value!r}. Defaults do "
                    f"not silently override templates -- {opts}"
                )
                continue
            dotpath.setval(cfg, segs, value)
            prov.append({"path": path, "source": f"{tag}(default)",
                         "value": value})

    # -- patch layer -------------------------------------------------------
    for path, value in sorted(config_patch.items()):
        if path in owned:
            conflicts.append(
                f"config_patch {path!r} claims a field owned by suite "
                f"param {owned[path]!r}; use the param, not a patch"
            )
            continue
        if path not in patchable_paths:
            conflicts.append(
                f"config_patch {path!r} is not patchable for this suite; "
                f"allowed: {list(patchable_paths)}"
            )
            continue
        try:
            segs = dotpath.parse(path)
            before = dotpath.get(cfg, segs)
            dotpath.setval(cfg, segs, value, allow_new=False)
        except (ValueError, KeyError) as e:
            conflicts.append(f"config_patch {path!r}: {e}")
            continue
        prov.append({"path": path, "source": "patch", "value": value,
                     "was": None if before is dotpath.MISSING else before})

    # -- suite policy pins (not layerable; owner 2026-10-24 standard) -----
    cfg["config"]["name"] = name
    cfg["name"] = name
    prov.append({"path": "config.name", "source": "suite-policy(name)",
                 "value": name})
    if jobs_root:
        # F3 placement: tool jobs never land in output/ proper; one
        # reapable subtree the mount already reads
        for i, proc in enumerate(cfg["config"].get("process", [])):
            old = proc.get("training_folder")
            proc["training_folder"] = jobs_root
            row = {"path": f"config.process[{i}].training_folder",
                   "source": "suite-policy(job-placement)",
                   "value": jobs_root}
            if old is not None and old != jobs_root:
                row["against_template"] = old
            prov.append(row)
    if not bool(params.get("sample", False)):
        proc = cfg["config"]["process"][0]
        save = proc.setdefault("save", {})
        save["save_every"] = int(params["steps"]) + 1
        save["max_step_saves_to_keep"] = 1
        prov.append({"path": f"{prefix}.save",
                     "source": "suite-policy(measurement-never-saves)",
                     "value": {"save_every": save["save_every"],
                               "max_step_saves_to_keep": 1}})

    if conflicts:
        raise ParamError(
            "config merge refused (nothing launched):\n  - "
            + "\n  - ".join(conflicts)
        )
    return cfg, prov


# -------------------------------------------------------------- harness

def watch_db(job_dir, paths: RunPaths, stop: threading.Event):
    db = os.path.join(job_dir, "loss_log.db")
    while not stop.wait(20.0):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1)
            row = con.execute("select max(step) from steps").fetchone()
            con.close()
            if row and row[0]:
                from .runner import emit

                emit(paths, "progress", step=row[0])
        except Exception:
            pass  # db not created / mid-transaction; retry next tick


def _tree_rss_mb():
    """Sum RSS (MB) of every process in THIS process group -- the run's
    real host-memory footprint (supervisor, harness, run.py and their
    children share the group). This is the 'RSS delta during load'
    attribution: system-wide memory can't say who ate it; the group
    can (e.g. weights staging in host RAM instead of going direct to
    GPU -- the legacy-load-code failure mode)."""
    total = 0
    try:
        pgid = os.getpgid(0)
    except OSError:
        return 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:  # ppid... field 3 = pgrp after ')'
                continue
            with open(f"/proc/{pid}/statm") as f:
                total += int(f.read().split()[1])
        except (FileNotFoundError, ProcessLookupError, IndexError,
                PermissionError, ValueError):
            continue
    return total * 4096 // (1024 * 1024)


def watch_mem(paths: RunPaths, stop: threading.Event, interval=1.0):
    """1 Hz sampler: GPU memory (nvidia-smi, external = zero in-process
    cost) AND this run's host RSS (own process group). Emits peak plus a
    compact summary to events.jsonl on exit. gpu fields keep their old
    names; rss/rss_peak are the per-run host footprint."""
    peak = 0
    rss_peak = 0
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
        rss = _tree_rss_mb()
        peak = max(peak, used)
        rss_peak = max(rss_peak, rss)
        samples.append(used)
        if len(samples) >= 5 and len(samples) % 30 == 0:
            from .runner import emit

            emit(paths, "mem", used=used, peak=peak, rss=rss,
                 rss_peak=rss_peak)
    if samples:
        from .runner import emit

        tail = samples[-60:]
        emit(
            paths, "mem_summary", peak=peak,
            tail_median=sorted(tail)[len(tail) // 2], tail_max=max(tail),
            rss_peak=rss_peak,
        )


@tag_metrics("steps.wall_time (db) + loss/loss")
def parse_metrics(paths: RunPaths, params):
    """All metrics from the job's loss_log.db (steps.wall_time +
    loss/loss); prefers the record's own snapshot (recompute contract:
    reproducible from the record alone)."""
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
        return {
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
    finally:
        con.close()


class TrainerKit:
    """Per-suite wiring: pass the model-specific bits, get the suite.

    namespace   job-name namespace (e.g. "MOSS")
    module      harness module path (python -m <module> --run-dir DIR)
    prefix      config path to the process entry ("config.process[0]")
    extra_params     model switches added to base_params
    extra_markers    model-specific markers added to TRAINER_MARKERS
    build_env(params, paths) -> env additions (model switches etc.)
    """

    def __init__(self, *, namespace, module, prefix="config.process[0]",
                 extra_params=None, extra_markers=None, build_env=None,
                 extra_flags=None, jobs_root=None, switches=None,
                 parse=None, spawn_job=None):
        from .runner import RUNS_ROOT  # no cycle: runner never imports kit

        self.namespace = namespace
        self.module = module
        self.prefix = prefix
        self.extra_params = extra_params or {}
        self.extra_markers = dict(extra_markers or {})
        self.build_env = build_env or (lambda p, paths: {})
        self.extra_flags = extra_flags or (lambda p: [])
        # param -> dotted config path for model switches (model_kwargs
        # and anything else the model reads from config); tracked with
        # full provenance, param-owned, never patchable
        self.switches = dict(switches or {})
        # custom metric parsing (probe-shaped suites); default loss-db
        self.parse = parse or parse_metrics
        # custom inner job spawn for probe-shaped harnesses; default
        # spawns run.py with the derived config
        self.spawn_job = spawn_job
        self.jobs_root = jobs_root or os.path.join(RUNS_ROOT, "jobs")
        self.job_name = job_name_base(namespace)
        self.patchable = patchable(prefix)

    # -- derivation (used by precheck, resolve and the harness) ----------

    def _named_ctx(self, params, ctx):
        ctx = dict(ctx or {})
        ctx.setdefault("extra_flags", tuple(self.extra_flags(params)))
        return ctx

    def plan(self, params, ctx):
        ctx = self._named_ctx(params, ctx)
        name = ctx.get("job_name") or self.job_name(params, ctx)
        return merge_plan(
            params, ctx.get("overrides"), ctx.get("config_patch"),
            name, prefix=self.prefix, patchable_paths=self.patchable,
            jobs_root=self.jobs_root, switches=self.switches,
        )

    def reserve(self, params, ctx=None):
        """Claim the versioned job dir (F3): creating the dir IS the
        reservation; a same-name race bumps the version, never shares."""
        ctx = self._named_ctx(params, ctx)
        base = self.job_name(params, ctx)
        os.makedirs(self.jobs_root, exist_ok=True)
        for v in range(1, 1000):
            name = f"{base}_v{v}"
            d = os.path.join(self.jobs_root, name)
            try:
                os.mkdir(d)
                return name, d
            except FileExistsError:
                continue
        raise ParamError(f"cannot reserve a job dir for {base!r} in 999 tries")

    def derive_config(self, params, name, out_path, overrides=None,
                      config_patch=None):
        import yaml

        cfg, prov = merge_plan(
            params, overrides, config_patch, name, prefix=self.prefix,
            patchable_paths=self.patchable, jobs_root=self.jobs_root,
            switches=self.switches,
        )
        folders = load_map(params["dataset"])
        ds = cfg["config"]["process"][0]["datasets"]
        if len(folders) != len(ds):
            raise ParamError(
                f"dataset {params['dataset']!r} maps {len(folders)} folders, "
                f"template has {len(ds)} dataset entries"
            )
        for entry, folder in zip(ds, folders):
            entry["folder_path"] = folder
        # the same check runs in the submit process via SUITE.precheck;
        # this copy guards the harness path so nothing can train on a
        # non-tool folder
        tool_dataset_check(params["dataset"], folders)
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f)
        with open(out_path.replace("config.yaml", "config_provenance.json"),
                  "w") as f:
            json.dump(prov, f, indent=1)
        return cfg

    # -- submit-process hooks ---------------------------------------------

    def precheck(self, params, ctx=None):
        # bad dataset or conflicted merge never becomes a doomed launch;
        # the harness re-runs the same code so a direct call can't skip
        tool_dataset_check(params["dataset"], load_map(params["dataset"]))
        self.plan(params, ctx)

    def resolve(self, params, ctx=None):
        _, prov = self.plan(params, ctx)
        return {"provenance": prov,
                "config_file": "artifacts/config.yaml (written by harness)"}

    # -- launch ------------------------------------------------------------

    def build(self, params, paths: RunPaths):
        argv = [sys.executable, "-m", self.module, "--run-dir", paths.root]
        env = {"SEED": params["seed"], "AITK_STEP_TIME": "1"}
        if params.get("prof_start") is not None:
            # the SDTrainer profiler window (no-stack); AITK_PROF_STACK is
            # never set (host OOM incident rule: stacks not exposed)
            prof_dir = os.path.join(paths.artifacts, "prof")
            env["AITK_PROF_DIR"] = prof_dir
            env["AITK_PROF_START"] = params["prof_start"]
            env["AITK_PROF_ACTIVE"] = params["prof_active"]
            os.makedirs(prof_dir, exist_ok=True)
        env.update({k: v for k, v in self.build_env(params, paths).items()})
        return argv, env

    # -- harness ------------------------------------------------------------

    def main(self, argv=None):
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--run-dir", required=True)
        args = ap.parse_args(argv)

        sys.path.insert(0, os.environ.get("RUNTOOL_REPO", os.getcwd()))
        from .runner import emit

        paths = RunPaths(run_id=os.path.basename(args.run_dir),
                        root=args.run_dir)
        with open(paths.manifest) as f:
            manifest = json.load(f)
        params = manifest["params"]

        # name/dir were reserved in the submit process (F3); the harness
        # uses them -- the run dir already exists, an empty dir is not a
        # reuse trap (only optimizer.pt is)
        name = manifest.get("job_name") or self.job_name(params)
        cfg_path = os.path.join(paths.artifacts, "config.yaml")
        cfg = self.derive_config(
            params, name, cfg_path,
            overrides=manifest.get("overrides"),
            config_patch=manifest.get("config_patch"),
        )
        folder = "output"
        for proc in cfg["config"].get("process", []):
            folder = proc.get("training_folder", folder)
        job_dir = os.path.join(folder, name)

        # divergence guard: the submit process reserved the exact job
        # dir; if the harness's independent derivation disagrees, the
        # layers drifted -- fail loudly rather than write somewhere else
        reserved = manifest.get("job_dir")
        if reserved and os.path.realpath(reserved) != os.path.realpath(job_dir):
            emit(paths, "job_dir_divergence", reserved=reserved,
                 derived=job_dir)
            with open(paths.result + ".tmp", "w") as f:
                json.dump(
                    {"ok": False,
                     "error": f"job dir divergence: reserved {reserved!r} "
                              f"but config derives {job_dir!r}; refusing "
                              f"to run outside the reservation"},
                    f, indent=1,
                )
            os.replace(paths.result + ".tmp", paths.result)
            return 1

        # reuse-trap guard: a finished/partial job under this name makes
        # the trainer RESUME (optimizer.pt), i.e. a bogus fast "done" --
        # fail the record loudly instead
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
        threading.Thread(target=watch_db, args=(job_dir, paths, stop),
                         daemon=True).start()
        threading.Thread(target=watch_mem, args=(paths, stop),
                         daemon=True).start()

        if self.spawn_job is not None:
            # probe-shaped suite: its own inner command
            rc = self.spawn_job(paths, params, manifest, cfg_path, job_dir)
        else:
            rc = subprocess.call(
                [sys.executable, "run.py", cfg_path, "--log", log_path],
                cwd=manifest["cwd"],
            )
        stop.set()
        if rc != 0:
            emit(paths, "stage_failed", rc=rc)
            return rc

        # snapshot the structured source into the record so metrics are
        # recomputable from the record alone (contract)
        src_db = os.path.join(job_dir, "loss_log.db")
        if os.path.exists(src_db):
            import shutil

            shutil.copy2(src_db, os.path.join(paths.artifacts, "metrics.db"))

        # measurement runs must leave no weights on disk (owner hard disk
        # rule): the toolkit's end-of-training save is UNCONDITIONAL, so
        # the harness deletes what it just made. Metrics live in the db
        # snapshot; quality runs (sample=true) keep their saves because
        # the saves ARE the deliverable.
        removed = []
        if not params.get("sample", False) and os.path.isdir(job_dir):
            for fname in sorted(os.listdir(job_dir)):
                if fname.endswith(".safetensors") or fname == "optimizer.pt":
                    os.remove(os.path.join(job_dir, fname))
                    removed.append(fname)
            if removed:
                emit(paths, "artifact_cleanup", removed=removed)

        metrics = self.parse(paths, params)
        with open(paths.result + ".tmp", "w") as f:
            json.dump(
                {"ok": True, "metrics": metrics, "job_name": name,
                 "job_dir": job_dir,
                 "metric_source": getattr(self.parse, "metric_source",
                                          "steps.wall_time (db) + loss/loss")},
                f, indent=1,
            )
        os.replace(paths.result + ".tmp", paths.result)
        emit(paths, "harness_done", metrics=metrics)
        return 0

    # -- assembly -----------------------------------------------------------

    def suite(self, *, suite_id, variants, timeout_s, desc, metrics=None,
              retention="disposable"):
        markers = dict(TRAINER_MARKERS)
        markers.update(self.extra_markers)
        return Suite(
            id=suite_id,
            variants=variants,
            params=base_params(**self.extra_params),
            metrics=metrics or METRICS,
            build=self.build,
            parse=self.parse,
            precheck=self.precheck,
            resolve=self.resolve,
            reserve=self.reserve,
            patchable=self.patchable,
            retention=retention,
            markers=markers,
            phases=dmark.trainer_phases,
            timeout_s=timeout_s,
            desc=desc,
        )
