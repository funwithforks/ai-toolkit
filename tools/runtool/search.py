"""search: query the run history (and the job dirs that exist).

The records are the database: every fact a search could want is
already structured JSON (manifest params/switches/note/git, result
metrics, persisted markers + expect verdict). Reading those is faster
and steadier than any grep over logs, and it keeps agent questions
inside the tool (AGENTS: grepping record dirs is the exception that
must file a complaint).

READ-ONLY by design: unlike status/list this never classifies with
side effects -- queued runs are reported as queued, not started. The
search must answer questions about history without touching it.

Rows are compact by intent: run_id, time, suite/variant, status,
expect verdict, job name, dataset, headline metrics. Full detail is
one status/inspect call away for whichever rows matter.
"""

import json
import os
import re
import time

from .runner import (RUNS_ROOT, _all_record_ids, _last_judgment, _pid_alive,
                     _read_json, _run_dir)

JOBS_ROOT = os.path.join(RUNS_ROOT, "jobs")
FINAL = ("done", "failed", "crash", "cancelled")

# what every row carries when the record has it
HEADLINE_METRICS = ("steps_done", "step_ms_window", "loss_final")


def _status_light(paths, manifest):
    """Terminal-ish status without classify's side effects (no queue
    pump, no marker scans -- those are fine for 20 rows, not for 2000
    records)."""
    if os.path.exists(paths.pid) and _pid_alive(paths, manifest):
        return "running"
    if manifest.get("after") and not os.path.exists(paths.pid):
        return "queued" if not os.path.exists(paths.cancel) else "cancelled"
    if os.path.exists(paths.cancel):
        return "cancelled"
    result = _read_json(paths.result)
    exit_code = None
    if os.path.exists(paths.exit_code):
        try:
            exit_code = int(open(paths.exit_code).read().strip())
        except ValueError:
            exit_code = None
    if result is not None and result.get("ok") and exit_code == 0:
        return "done"
    if result is not None and not result.get("ok"):
        return "failed"
    if exit_code is None and not os.path.exists(paths.exit_code):
        return "crash"
    return "crash"


_METRIC_RE = re.compile(r"^\s*(\w+)\s*(<=|>=|==|<|>)\s*(-?[\d.]+)\s*$")


def _metric_ok(metrics, clause):
    m = _METRIC_RE.match(clause)
    if not m:
        raise ValueError(f"bad --metric {clause!r} (want name<op><number>)")
    name, op, num = m.group(1), m.group(2), float(m.group(3))
    if not isinstance(metrics, dict) or metrics.get(name) is None:
        return False
    v = float(metrics[name])
    return {"<": v < num, "<=": v <= num, ">": v > num,
            ">=": v >= num, "==": v == num}[op]


def search(suite=None, status=None, dataset=None, tag=None, note=None,
           metric=None, marker=None, no_marker=None, param=None,
           expect=None, judged=None, since=None, variant=None, limit=25):
    """metric: ["loss_final<0.31", ...]; param: ["cutlass_bwd=off", ...];
    since: float days; expect: 'PASS'|'FAIL' (the persisted verdict);
    judged: verdict the owner attached via `judge`."""
    metric = metric or []
    param = param or []
    cutoff = time.time() - since * 86400 if since else None
    rows, matched, scanned = [], 0, 0
    for rid in sorted(_all_record_ids(), reverse=True):  # newest first
        paths = _run_dir(rid)
        manifest = _read_json(paths.manifest)
        if manifest is None:
            continue
        scanned += 1
        if suite and manifest.get("suite") != suite:
            continue
        if variant and manifest.get("variant") != variant:
            continue
        if cutoff and (manifest.get("submitted_at") or 0) < cutoff:
            continue
        if dataset and (manifest.get("params") or {}).get("dataset") != dataset:
            continue
        if tag and tag.lower() not in str(
                (manifest.get("params") or {}).get("tag", "")).lower():
            continue
        if note and note.lower() not in str(manifest.get("note", "")).lower():
            continue
        p = manifest.get("params") or {}
        bad_param = False
        for clause in param:
            k, _, want = clause.partition("=")
            if str(p.get(k)).lower() != want.strip().lower():
                bad_param = True
                break
        if bad_param:
            continue
        st = _status_light(paths, manifest)
        if status and st != status:
            continue

        result = _read_json(paths.result)
        metrics = (result or {}).get("metrics") or {}
        if any(not _metric_ok(metrics, c) for c in metric):
            continue

        mk = _read_json(os.path.join(paths.root, "markers.json")) or {}
        found = {k for k, v in (mk.get("markers") or {}).items()
                 if v.get("found")}
        if any(m not in found for m in (marker or [])):
            continue
        if any(m in found for m in (no_marker or [])):
            continue
        verdict = (mk.get("expect") or {}).get("verdict")
        if expect and verdict != expect:
            continue
        judgment = _last_judgment(paths)
        if judged and (judgment or {}).get("verdict") != judged:
            continue

        matched += 1
        if len(rows) < limit:
            row = {
                "run_id": rid,
                "when": time.strftime(
                    "%Y-%m-%d %H:%M",
                    time.localtime(manifest.get("submitted_at") or 0)),
                "suite": manifest.get("suite"),
                "variant": manifest.get("variant"),
                "status": st,
                "job_name": manifest.get("job_name"),
            }
            if verdict:
                row["verdict"] = verdict
            if judgment and judgment.get("verdict"):
                row["judged"] = judgment["verdict"]
            if p.get("dataset"):
                row["dataset"] = p["dataset"]
            if p.get("tag"):
                row["tag"] = p["tag"]
            head = {k: metrics[k] for k in HEADLINE_METRICS
                    if k in metrics}
            if head:
                row["metrics"] = head
            rows.append(row)
    return {"scanned": scanned, "matched": matched, "rows": rows,
            "truncated": matched > len(rows)}


def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def jobs():
    """The jobs-subtree question 'which job dirs exist, who owns them,
    what state': gc answers what it would REAP; this answers what IS."""
    owner = {}
    for rid in _all_record_ids():
        manifest = _read_json(_run_dir(rid).manifest)
        jd = (manifest or {}).get("job_dir")
        if jd:
            owner[os.path.realpath(jd)] = rid
    out = []
    if os.path.isdir(JOBS_ROOT):
        for entry in sorted(os.listdir(JOBS_ROOT)):
            path = os.path.join(JOBS_ROOT, entry)
            if not os.path.isdir(path):
                continue
            real = os.path.realpath(path)
            rid = owner.get(real)
            row = {"dir": entry,
                   "size_mb": round(_dir_size(path) / 2**20, 1),
                   "age_days": round((time.time() - os.path.getmtime(path))
                                     / 86400, 1)}
            if rid is None:
                row["owner"] = "unclaimed"
            else:
                manifest = _read_json(_run_dir(rid).manifest)
                row["owner"] = rid
                row["run_status"] = _status_light(_run_dir(rid), manifest)
                row["retention"] = manifest.get("retention")
            out.append(row)
    total = sum(r["size_mb"] for r in out)
    return {"jobs_root": JOBS_ROOT, "count": len(out),
            "total_size_mb": round(total, 1), "dirs": out}
