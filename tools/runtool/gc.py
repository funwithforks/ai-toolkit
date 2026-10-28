"""gc: reap the runtool jobs subtree (F3).

Owner ruling 2026-10-29: tool-made training runs are DISPOSABLE by
default -- checkpoints, weights and embed caches under the jobs root
are NOT precious and must not accumulate; agents may apply the cleanup.
The judged retention (owner 2026-10-24 quality standard: sample=true
runs are judged by their samples) still needs --include-judged, and
judged deletions PRINT what they kill first.

Deletion targets live strictly inside <runs_root>/jobs/:
  * job dirs referenced by a finished run record (done/failed/crash/
    cancelled), older than --days;
  * jobs-dir orphans (no record references them);
  * jobs_root/.prompt_embed_cache/<hash> entries older than --days
    (fixed-embed caches regenerate on demand: deleting is a TE reload
    cost, never a data loss).
Run RECORDS (output/runtool/<run_id>) are never touched: they are
small, permanent, and the mount reader depends on them. Nothing outside
the jobs subtree can be reached -- the path guard is absolute.
"""

import json
import os
import shutil
import time

from .runner import RUNS_ROOT, _read_json

JOBS_ROOT = os.path.join(RUNS_ROOT, "jobs")
CACHE_DIR = ".prompt_embed_cache"
FINAL = ("done", "failed", "crash", "cancelled")


def _size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _inside_jobs(path):
    real = os.path.realpath(path)
    try:
        return os.path.commonpath([os.path.realpath(JOBS_ROOT), real]) \
            == os.path.realpath(JOBS_ROOT) and real != os.path.realpath(
                JOBS_ROOT)
    except ValueError:
        return False


def candidates(days=7.0):
    """Everything gc would reap, with reasons; nothing is deleted here."""
    if not os.path.isdir(JOBS_ROOT):
        return {"jobs_root": JOBS_ROOT, "candidates": [], "orphans": [],
                "embed_caches": []}
    cutoff = time.time() - days * 86400

    # index job dirs -> their run record
    referenced = {}
    for name in sorted(os.listdir(RUNS_ROOT)):
        rec = os.path.join(RUNS_ROOT, name)
        if not os.path.isdir(rec) or name == "jobs" or name == "datasets":
            continue
        manifest = _read_json(os.path.join(rec, "manifest.json"))
        if manifest and manifest.get("job_dir"):
            referenced[os.path.realpath(manifest["job_dir"])] = (
                name, manifest)

    cands, orphans, caches = [], [], []
    # an active run anywhere may be reading a shared embed cache right
    # now -> the cache sweep is inert while anything runs (conservative:
    # these are per-model-hash dirs shared across concurrent jobs)
    any_active = False
    for run_id in referenced:
        from .runner import classify
        try:
            if classify(run_id)["status"] == "running":
                any_active = True
                break
        except Exception:
            pass
    for entry in sorted(os.listdir(JOBS_ROOT)):
        path = os.path.join(JOBS_ROOT, entry)
        if not os.path.isdir(path):
            continue
        real = os.path.realpath(path)
        mtime = os.path.getmtime(path)
        if entry == CACHE_DIR:
            for h in sorted(os.listdir(path)):
                hp = os.path.join(path, h)
                if not (os.path.isdir(hp) and os.path.getmtime(hp) <= cutoff):
                    continue
                if any_active:
                    continue  # a live run may be reading this cache
                caches.append({
                        "path": hp, "kind": "embed_cache", "hash": h,
                        "size_bytes": _size(hp),
                        "age_days": round((time.time()
                                           - os.path.getmtime(hp)) / 86400,
                                          1),
                        "note": "regenerates on demand (TE reload cost)",
                    })
            continue
        ref = referenced.get(real)
        if ref is None:
            if mtime <= cutoff:
                orphans.append({"path": path, "kind": "orphan",
                                "size_bytes": _size(path),
                                "age_days": round((time.time() - mtime)
                                                  / 86400, 1)})
            continue
        run_id, manifest = ref
        if mtime > cutoff:
            continue
        from .runner import classify  # lazy: needs registry

        try:
            status = classify(run_id)["status"]
        except Exception:  # record unreadable: not a candidate
            status = "unknown"
        if status not in FINAL:
            continue  # running/queued jobs are never candidates
        retention = manifest.get("retention", "disposable")
        cands.append({
            "path": path, "kind": "job", "run_id": run_id,
            "status": status, "retention": retention,
            "size_bytes": _size(path),
            "age_days": round((time.time() - mtime) / 86400, 1),
        })
    return {"jobs_root": JOBS_ROOT, "days": days, "candidates": cands,
            "orphans": orphans, "embed_caches": caches,
            "total_bytes": sum(c["size_bytes"] for c in cands)
            + sum(o["size_bytes"] for o in orphans)
            + sum(c["size_bytes"] for c in caches)}


def apply_removal(days=7.0, include_judged=False):
    """Delete exactly what candidates() listed (minus judged unless
    explicitly included). The path guard is re-checked per entry."""
    rep = candidates(days=days)
    removed, kept_judged, freed = [], [], 0
    groups = ([dict(c, _kill=True) for c in rep["candidates"]]
              + rep["orphans"] + rep["embed_caches"])
    for c in groups:
        if c.get("kind") == "job" and c.get("retention") == "judged" \
                and not include_judged:
            kept_judged.append({"path": c["path"], "run_id":
                                c.get("run_id"),
                                "detail": "judged; pass include_judged "
                                          "after the owner judges"})
            continue
        path = c["path"]
        if not _inside_jobs(path):
            kept_judged.append({"path": path,
                                "detail": "REFUSED: outside jobs root"})
            continue
        size = c.get("size_bytes") or _size(path)
        shutil.rmtree(path)
        freed += size
        removed.append({k: v for k, v in c.items() if k != "_kill"})
    rep["removed"] = removed
    rep["kept"] = kept_judged
    rep["freed_bytes"] = freed
    for k in ("candidates", "orphans", "embed_caches"):
        rep.pop(k, None)
    return rep
