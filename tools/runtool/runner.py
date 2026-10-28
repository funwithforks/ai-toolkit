"""Process control for runtool: submit, supervise, wait, status, cancel.

The runner owns launch, process group, pid, timeout, cancel and the wait.
A tool call never passes a command; suites do (see registry.py).

    submit    validates suite/variant/params, captures the working-tree
              patch, writes the manifest, detaches a supervisor (own
              session => recorded pid == process-group id) and returns.
    supervise launches the suite harness as a child in the same group,
              blocks on waitpid (so exit codes are captured even for hard
              child crashes), writes exit.code.
    wait      blocks on the RECORDED pid (never ps) until exit / cancel
              marker / timeout, then classifies and returns metrics.
    status    one-shot read: recorded-pid liveness + classification.
    cancel    writes the cancel marker, then signals the recorded pgid.

Classification contract:
    done       harness exited 0 AND wrote result.json (metrics)
    failed     harness exited non-zero AND wrote result.json (ok:false)
    cancelled  cancel marker present and the group is gone
    crash      process gone without the result file (exit code attached
               when the supervisor survived to record it)
    running    recorded pid is alive
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

from . import markers as marker_mod
from .registry import ParamError, RunPaths, get_suite

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
RUNS_ROOT = os.environ.get("RUNTOOL_HOME") or os.path.join(
    REPO_ROOT, "output", "runtool"
)


# ----------------------------------------------------------------- records

def _now():
    return time.time()


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def emit(paths: RunPaths, event: str, **fields):
    """Append one structured progress event (jsonl). Safe from any process."""
    rec = {"t": round(_now(), 3), "event": event}
    rec.update(fields)
    with open(paths.events, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()


def _run_dir(run_id):
    return RunPaths(run_id=run_id, root=os.path.join(RUNS_ROOT, run_id))


def _git(*args):
    try:
        out = subprocess.run(
            ("git",) + args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


# ------------------------------------------------------------------ submit

def _all_record_ids():
    try:
        names = sorted(os.listdir(RUNS_ROOT))
    except FileNotFoundError:
        return []
    return [n for n in names
            if n not in ("jobs", "datasets")
            and os.path.isdir(os.path.join(RUNS_ROOT, n))]


def _live_state(run_id):
    """Cheap liveness triage WITHOUT classify's side effects:
    'running' | 'queued' | 'finished'. Only used by the exclusivity
    guard and the queue cap."""
    paths = _run_dir(run_id)
    manifest = _read_json(paths.manifest)
    if manifest is None:
        return "finished"
    if os.path.exists(paths.pid):
        if os.path.exists(paths.result) or os.path.exists(paths.exit_code) \
                and not _pid_alive(paths, manifest):
            # exited: final status doesn't matter, it's not live
            return "finished"
        return "running" if _pid_alive(paths, manifest) else "finished"
    if os.path.exists(paths.cancel):
        return "finished"
    if manifest.get("after") and not os.path.exists(paths.result):
        return "queued"
    return "finished"


def _live_job_runs():
    """Live runs in the jobs-subtree (F3-era, have job_dir) — exactly
    what the exclusivity guard and queue cap count."""
    out = []
    for rid in _all_record_ids():
        manifest = _read_json(_run_dir(rid).manifest)
        if manifest is None or not manifest.get("job_dir"):
            continue
        st = _live_state(rid)
        if st in ("running", "queued"):
            out.append({"run_id": rid, "state": st,
                        "job_name": manifest.get("job_name", rid)})
    return out


_SUBMIT_LOCK_TIMEOUT_S = 120


def _acquire_submit_lock():
    """Serialise exclusivity-check + spawn so two same-instant submits
    cannot both pass the guard. Stale locks (holder died mid-submit)
    are stealable after a window."""
    lock = os.path.join(RUNS_ROOT, ".submit.lock")
    os.makedirs(RUNS_ROOT, exist_ok=True)
    deadline = time.time() + 10
    while True:
        try:
            os.mkdir(lock)
            return lock
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > _SUBMIT_LOCK_TIMEOUT_S:
                    os.rmdir(lock)
                    continue
            except FileNotFoundError:
                continue
            if time.time() > deadline:
                raise ParamError(
                    "another submit is in flight and the submit lock did "
                    f"not clear in 10s ({lock}); check `runtool list`")
            time.sleep(0.1)


def _spawn_supervisor(run_id, paths):
    sup = [
        sys.executable,
        "-m", "tools.runtool.runner",
        "--supervise", run_id,
    ]
    # supervisor's own output goes to the record (DEVNULL would swallow
    # supervisor-side failures whole)
    with open(os.path.join(paths.root, "supervisor.log"), "ab") as suplog:
        proc = subprocess.Popen(
            sup,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=suplog,
            stderr=suplog,
            start_new_session=True,  # detached; this pid is the pgid to signal
        )
    with open(paths.pid, "w") as f:
        f.write(str(proc.pid))
    return proc


def submit(suite_id, variant, overrides, note="", expect=None,
           config_patch=None, after=None):
    suite = get_suite(suite_id)
    params = suite.validate(variant, overrides or {})
    # F2: the loud-merge layers need to know which params the submitter
    # EXPLICITLY passed (those may override the template; a silent
    # default may not). Record the explicit set, not just the merged one.
    explicit = sorted(k for k in (overrides or {}) if k in suite.params)
    if config_patch is not None and not isinstance(config_patch, dict):
        raise ParamError("config_patch must be an object of dotted paths")
    ctx = {"overrides": explicit, "config_patch": config_patch}
    if suite.precheck is not None:
        suite.precheck(params, ctx)  # reject here, not in a doomed launch
    if expect is not None:
        try:
            marker_mod.validate_expect(suite, expect)
        except ValueError as e:
            raise ParamError(str(e)) from None
    # F6: exclusivity + queue. A jobs-subtree run reserves a dir and
    # measures timing/memory that only means something on a quiet GPU;
    # a second live one poisons both sets of numbers. --after queues
    # behind the holder (small chains, cap 3) instead of refusing.
    jobs_subtree = suite.reserve is not None
    if after is not None:
        if not jobs_subtree:
            raise ParamError("after is only valid for suites that "
                             "reserve job dirs")
        if not os.path.isdir(os.path.join(RUNS_ROOT, after)):
            raise ParamError(f"after: no run record for {after!r}")
        if _live_state(after) == "finished":
            raise ParamError(
                f"after: {after!r} is not live (already finished) — "
                "submit without --after to run directly")
    retention = "judged" if (params or {}).get("sample") else suite.retention
    # The guard runs BEFORE any reservation (a refusal leaves no record
    # and no claimed job dir); the lock spans check -> spawn so two
    # same-instant submits cannot both pass it.
    lock = _acquire_submit_lock() if jobs_subtree else None
    try:
        queued = after is not None
        job_name = job_dir = None
        if jobs_subtree:
            live = _live_job_runs()
            if queued:
                if len(live) + 1 > 3:
                    holders = ", ".join(
                        f"{r['run_id']} ({r['state']})" for r in live)
                    raise ParamError(
                        f"queue is full (max 3 live) — {holders}. "
                        "Wait for one to finish.")
            elif live:
                holders = ", ".join(
                    f"{r['run_id']} ({r['state']}, {r['job_name']})"
                    for r in live)
                raise ParamError(
                    f"the jobs subtree is busy: {holders}. Two live "
                    "jobs-subtree runs poison each other's timing and "
                    "memory numbers. Queue behind one with "
                    "after=<run_id>, or cancel it first.")
            # only now claim the dir: a refusal must not burn a version
            job_name, job_dir = suite.reserve(params, ctx)
            ctx["job_name"] = job_name
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        run_id = f"{stamp}_{suite.id}_{variant}"
        paths = _run_dir(run_id)
        base = run_id
        n = 1
        while os.path.exists(paths.root):  # same-second collision guard
            n += 1
            run_id = f"{base}_{n}"
            paths = _run_dir(run_id)
        os.makedirs(paths.artifacts, exist_ok=True)

        argv, env_over = suite.build(params, paths)
        env = dict(os.environ)
        env.update({k: str(v) for k, v in env_over.items()})

        patch = _git("diff", "HEAD")
        if patch:
            with open(paths.patch, "w") as f:
                f.write(patch)

        manifest = {
            "run_id": run_id,
            "suite": suite.id,
            "variant": variant,
            "params": params,
            "note": note,
            "argv": argv,
            "env": {k: v for k, v in env_over.items()},
            "cwd": REPO_ROOT,
            "record": paths.root,
            "git": {
                "head": _git("rev-parse", "--short", "HEAD").strip(),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
                "dirty_tracked": bool(patch),
                "porcelain": _git("status", "--porcelain").strip().splitlines()[:200],
            },
            "submitted_at": _now(),
            "timeout_s": suite.timeout_s,
            "python": sys.executable,
        }
        if expect is not None:
            manifest["expect"] = expect
        manifest["overrides"] = explicit
        manifest["retention"] = retention
        if job_name is not None:
            manifest["job_name"] = job_name
            manifest["job_dir"] = job_dir
        if config_patch is not None:
            manifest["config_patch"] = config_patch
        if after is not None:
            manifest["after"] = after
        _write_json(paths.manifest, manifest)
        emit(paths, "submitted", suite=suite.id, variant=variant,
             params=params, after=after)

        proc = None
        if not queued:
            proc = _spawn_supervisor(run_id, paths)
    finally:
        if lock is not None:
            try:
                os.rmdir(lock)
            except OSError:
                pass

    result = {"run_id": run_id, "record": paths.root, "pid": None,
              "retention": retention,
              "status": "queued" if queued else "running"}
    if after is not None:
        result["after"] = after
    if proc is not None:
        result["pid"] = proc.pid
    if job_dir is not None:
        result["job_name"] = job_name
        result["job_dir"] = job_dir
    if expect is not None:
        result["expect"] = expect  # echo the contract the verdict will grade
    if suite.resolve is not None:
        # loud merge: the layer decisions are known NOW (precheck proved
        # the merge clean; ctx carries the reserved versioned job_name);
        # echo them so the caller sees what the job will run without
        # opening a file
        result["config_resolved"] = suite.resolve(params, ctx)
    return result


# --------------------------------------------------------------- supervise

def supervise(run_id):
    paths = _run_dir(run_id)
    manifest = _read_json(paths.manifest)
    if manifest is None:
        return 2
    env = dict(os.environ)
    env.update({k: str(v) for k, v in manifest.get("env", {}).items()})
    emit(paths, "supervise_start", argv=manifest["argv"])
    with open(paths.stdout, "ab") as out, open(paths.stderr, "ab") as err:
        proc = subprocess.Popen(
            manifest["argv"],
            cwd=manifest["cwd"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            # same process group as the supervisor => one killpg ends the
            # harness plus every stage child it spawns
        )
    emit(paths, "harness_spawned", pid=proc.pid)
    rc = proc.wait()
    with open(paths.exit_code, "w") as f:
        f.write(str(rc))
    emit(paths, "harness_exit", rc=rc)
    return 0


# ------------------------------------------------------------ liveness

def _proc_start_ticks(pid):
    """/proc starttime ticks, or None when the pid is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, ValueError):
        return None


_SUBMIT_MARK_KEY = "_submit_start_ticks"


def _pid_alive(paths, manifest):
    try:
        pid = int(open(paths.pid).read().strip())
    except (FileNotFoundError, ValueError):
        return False
    ticks = _proc_start_ticks(pid)
    if ticks is None:
        return False
    mark = manifest.get(_SUBMIT_MARK_KEY)
    if mark is not None and ticks != mark:  # pid recycled since submit
        return False
    return True


def _attach_start_mark(paths, manifest):
    # called lazily right after submit while the supervisor is certainly
    # alive; pins the identity of the recorded pid against reuse
    if manifest.get(_SUBMIT_MARK_KEY) is not None:
        return manifest
    try:
        pid = int(open(paths.pid).read().strip())
        ticks = _proc_start_ticks(pid)
    except (FileNotFoundError, ValueError):
        return manifest
    if ticks is None:
        return manifest
    manifest[_SUBMIT_MARK_KEY] = ticks
    _write_json(paths.manifest, manifest)
    return manifest


# -------------------------------------------------------------- classify

def _start_queued(paths, manifest):
    """Queue pump (F6): start a queued run once its predecessor reaches
    a non-blocking terminal state. Called from classify(), so ANY
    status/wait/list call advances the queue -- no daemon process.
    Returns 'started' | 'waiting' | 'cancelled' | 'failed'."""
    after = manifest["after"]
    if os.path.exists(paths.cancel):
        return "cancelled"
    try:
        pred = classify(after)["status"]
    except FileNotFoundError:
        pred = "gone"
    if pred == "cancelled":
        # cancellation propagates down the chain (spec F6)
        with open(paths.cancel, "w") as f:
            f.write(str(_now()))
        emit(paths, "cancel_propagated", after=after)
        return "cancelled"
    if pred == "gone":
        with open(paths.exit_code, "w") as f:
            f.write("1")
        with open(paths.result + ".tmp", "w") as f:
            json.dump({"ok": False,
                       "error": f"queued predecessor {after!r} record is "
                                "gone; refusing to start unattended"}, f)
        os.replace(paths.result + ".tmp", paths.result)
        emit(paths, "queued_predecessor_gone", after=after)
        return "failed"
    if pred in ("running", "queued"):
        return "waiting"
    # predecessor finished (done/failed/crash): the chain must not
    # stall on a failed link -- start now. Claim atomically so two
    # concurrent pumps cannot double-spawn.
    claim = os.path.join(paths.root, ".start_claim")
    try:
        os.mkdir(claim)
    except FileExistsError:
        if os.path.exists(paths.pid):
            return "started"
        if time.time() - os.path.getmtime(claim) > 120:
            # a previous claimer died before spawning: retire its claim
            os.replace(claim, claim + f".dead.{int(time.time())}")
            os.mkdir(claim)
        else:
            return "waiting"
    proc = _spawn_supervisor(manifest["run_id"], paths)
    emit(paths, "queued_start", after=after, pid=proc.pid)
    return "started"


def _tail(path, n_lines=12):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            return "".join(f.read().decode("utf8", "replace").splitlines(True)[-n_lines:])
    except FileNotFoundError:
        return ""


def classify(run_id):
    """One-shot status: {run_id, status, exit_code, metrics, progress, ...}."""
    paths = _run_dir(run_id)
    manifest = _read_json(paths.manifest)
    if manifest is None:
        raise FileNotFoundError(f"no run record for {run_id!r} under {RUNS_ROOT}")
    manifest = _attach_start_mark(paths, manifest)
    # F6 pump: an unstarted queued run advances on any status touch
    queued_hold = False
    if manifest.get("after") and not os.path.exists(paths.pid):
        if _start_queued(paths, manifest) == "waiting":
            queued_hold = True
    result = _read_json(paths.result)
    exit_code = None
    if os.path.exists(paths.exit_code):
        try:
            exit_code = int(open(paths.exit_code).read().strip())
        except ValueError:
            exit_code = None
    alive = _pid_alive(paths, manifest)
    cancelled = os.path.exists(paths.cancel)
    progress = None
    try:
        with open(paths.events, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8192))
            lines = [ln for ln in f.read().decode("utf8", "replace").splitlines() if ln.strip()]
            for ln in reversed(lines):
                try:
                    ev = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "progress":
                    progress = ev
                    break
            if progress is None and lines:
                try:
                    progress = json.loads(lines[-1])
                except json.JSONDecodeError:
                    progress = {"raw": lines[-1][-200:]}
    except FileNotFoundError:
        pass

    if queued_hold:
        status = "queued"
    elif alive:
        status = "running"
    elif cancelled:
        status = "cancelled"
    elif exit_code is None:
        status = "crash"
    elif result is not None and result.get("ok") and exit_code == 0:
        status = "done"
    elif result is not None and not result.get("ok"):
        status = "failed"
    else:
        # exited 0 without the result file (contract violation by the run)
        # or crashed non-zero: either way there is no metrics file -> crash
        status = "crash"

    out = {
        "run_id": run_id,
        "suite": manifest["suite"],
        "variant": manifest["variant"],
        "status": status,
        "exit_code": exit_code,
        "pid": _pid_of(paths),
        "alive": alive,
        "progress": progress,
        "record": paths.root,
        "started_at": manifest.get("submitted_at"),
    }
    if result is not None:
        out["metrics"] = result.get("metrics", {})
        if not result.get("ok") and result.get("error"):
            out["error"] = result["error"]
    if status in ("crash", "failed") and "error" not in out:
        err_tail = _tail(paths.stderr) or _tail(paths.stdout)
        if err_tail:
            out["error_tail"] = err_tail

    # F1: marker extraction + expectation verdict, derived from the record
    # alone. Suites that declare nothing cost nothing. The scan runs once
    # per call (status/wait both flow through here); the live window is
    # bounded and finished records are small.
    try:
        suite = get_suite(manifest["suite"])
    except ParamError:
        suite = None  # record references a removed suite: metrics still serve
    if suite is not None and (suite.markers or suite.phases) \
            and status != "queued":
        full = not alive
        hits = marker_mod.scan(paths, suite, full)
        if hits:
            out["markers"] = hits
        if suite.phases is not None:
            phases = suite.phases(paths, suite, full)
            if phases:
                out["phases"] = phases
        expect = manifest.get("expect")
        if expect is not None:
            verdict = marker_mod.evaluate(expect, hits, out.get("metrics", {}))
            if alive:
                # mid-run: checks are informational only -- an absence may
                # still fill in. PASS/FAIL is a post-expectation verdict,
                # never a live guess.
                verdict = {"verdict": "PENDING", "checks": verdict["checks"]}
            out["expect"] = verdict
            out["expect"] = verdict
            if not alive:
                # persist so recompute must reproduce exactly this verdict
                _write_json(os.path.join(paths.root, "markers.json"), {
                    "markers": hits,
                    "phases": out.get("phases"),
                    "expect": verdict,
                })
    return out


def _pid_of(paths):
    try:
        return int(open(paths.pid).read().strip())
    except (FileNotFoundError, ValueError):
        return None


# ------------------------------------------------------------------ wait

def wait(run_id, timeout_s=None, poll_s=0.75):
    manifest = _read_json(_run_dir(run_id).manifest)
    if manifest is None:
        raise FileNotFoundError(f"no run record for {run_id!r} under {RUNS_ROOT}")
    if timeout_s is None:
        timeout_s = manifest.get("timeout_s", 3600)
    deadline = _now() + timeout_s
    while True:
        st = classify(run_id)
        if st["status"] not in ("running", "queued"):
            if st["status"] in ("crash", "failed", "timeout"):
                # the moment of pain is where the channel gets used
                from .complain import complaint_hint
                st["you_may_file"] = complaint_hint(run_id)
            return st
        if _now() >= deadline:
            st["status"] = "timeout"
            emit(_run_dir(run_id), "wait_timeout", timeout_s=timeout_s)
            from .complain import complaint_hint
            st["you_may_file"] = complaint_hint(run_id)
            return st
        time.sleep(poll_s)


# ---------------------------------------------------------------- cancel

def cancel(run_id, grace_s=8.0):
    paths = _run_dir(run_id)
    st = classify(run_id)
    if st["status"] == "queued":
        with open(paths.cancel, "w") as f:
            f.write(str(_now()))
        emit(paths, "cancel_requested", queued=True)
        return classify(run_id)
    if st["status"] != "running":
        return {"run_id": run_id, "status": st["status"], "detail": "not running"}
    with open(paths.cancel, "w") as f:
        f.write(str(_now()))
    emit(paths, "cancel_requested")
    pgid = st["pid"]
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = _now() + grace_s
    while _now() < deadline:
        if not _pid_alive(paths, _read_json(paths.manifest)):
            break
        time.sleep(0.2)
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    emit(paths, "cancel_signalled")
    return classify(run_id)


# ------------------------------------------------------------------- list

def list_runs(limit=15):
    try:
        names = sorted(os.listdir(RUNS_ROOT), reverse=True)
    except FileNotFoundError:
        return []
    out = []
    for name in names:
        # the runs root also hosts subtrees (jobs/, datasets/, the
        # registry file): only real run records qualify
        if name in ("jobs", "datasets") or not os.path.isdir(
                os.path.join(RUNS_ROOT, name)):
            continue
        try:
            out.append(classify(name))
        except (FileNotFoundError, OSError):
            continue
        if len(out) >= limit:
            break
    return out


def recompute(run_id):
    """Re-run the suite parser (and marker scan) over the record; must
    match result.json metrics and any stored markers.json verdicts."""
    paths = _run_dir(run_id)
    manifest = _read_json(paths.manifest)
    if manifest is None:
        raise FileNotFoundError(run_id)
    suite = get_suite(manifest["suite"])
    params = manifest["params"]
    if not os.path.exists(paths.result):
        # never started (cancelled in queue) or died before writing:
        # there is no result to reproduce -- say so, don't traceback
        return {"run_id": run_id, "match": None,
                "detail": "no result.json: the run never produced a "
                          "result (cancelled before start?)"}
    metrics = suite.parse(paths, params)
    stored = (_read_json(paths.result) or {}).get("metrics")
    out = {
        "run_id": run_id,
        "recomputed": metrics,
        "stored": stored,
        "match": stored == metrics,
    }
    if suite.markers or suite.phases:
        hits = marker_mod.scan(paths, suite, full=True)
        recomputed = {"markers": hits}
        if suite.phases is not None:
            recomputed["phases"] = suite.phases(paths, suite, True)
        expect = manifest.get("expect")
        if expect is not None:
            recomputed["expect"] = marker_mod.evaluate(expect, hits, metrics)
        out["markers_recomputed"] = recomputed
        stored_mk = _read_json(os.path.join(paths.root, "markers.json"))
        if stored_mk is not None:
            out["markers_match"] = {
                "markers": stored_mk.get("markers") == hits,
                "phases": stored_mk.get("phases") == recomputed.get("phases"),
                "expect": (stored_mk.get("expect")
                           == recomputed.get("expect")),
            }
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--supervise", required=True)
    a = ap.parse_args()
    sys.exit(supervise(a.supervise))
