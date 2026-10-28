"""The complaint channel (F7): agents file what the tool made harder.

The channel only matters if it is cheaper than the workaround it
reports, so filing is one line and the failure-path hands the agent
the exact command at the moment of pain (runner.wait attaches
COMPLAINT_HINT to crash/failed/timeout returns).

Storage: one append-only JSONL under the runs root. Each line:
{t, run_id?, suite, variant?, category, text, cost?, agent?}.
`complaints` aggregates by category x suite and shows the newest --
that aggregate is the agenda for tool sessions.
"""

import json
import os
import time

from .runner import RUNS_ROOT, _read_json  # noqa: F401 (RUNS_ROOT for path)

COMPLAINTS_PATH = os.path.join(RUNS_ROOT, "complaints.jsonl")

CATEGORIES = ("api-confusion", "misleading-output", "missing-measurement",
              "ergonomics", "doc-gap", "other")


def complaint_hint(run_id):
    """The imperative sentence, offered where the tool hurt."""
    return ("If this made your job harder, file it BEFORE working around "
            f'it: runtool complain --run-id {run_id} --category one of '
            f'{"|".join(CATEGORIES)} --text "what hurt and the workaround '
            'you paid to find"')


def file_complaint(text, category, run_id=None, cost=None, agent=None):
    from .registry import ParamError
    if not text or not text.strip():
        raise ParamError("complain: --text is required")
    if category not in CATEGORIES:
        raise ParamError(f"complain: unknown category {category!r}; use "
                         f"one of {' | '.join(CATEGORIES)}")
    suite = variant = None
    if run_id is not None:
        manifest = _read_json(os.path.join(RUNS_ROOT, run_id,
                                           "manifest.json"))
        if manifest is None:
            raise ParamError(f"complain: no run record for {run_id!r}")
        suite = manifest.get("suite")
        variant = manifest.get("variant")
    entry = {"t": time.time(), "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
             "run_id": run_id, "suite": suite, "variant": variant,
             "category": category, "text": text.strip()}
    if cost:
        entry["cost"] = cost
    if agent:
        entry["agent"] = agent
    os.makedirs(RUNS_ROOT, exist_ok=True)
    with open(COMPLAINTS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"filed": True, "path": COMPLAINTS_PATH, "entry": entry}


def _all():
    try:
        with open(COMPLAINTS_PATH) as f:
            return [json.loads(ln) for ln in f if ln.strip()]
    except FileNotFoundError:
        return []


def complaints(stats=False, limit=20):
    rows = _all()
    newest = sorted(rows, key=lambda r: r.get("t", 0), reverse=True)
    out = {"path": COMPLAINTS_PATH, "total": len(rows),
           "newest": newest[:limit]}
    if stats:
        agg = {}
        for r in rows:
            key = (r.get("category", "?"), r.get("suite") or "-")
            agg[key] = agg.get(key, 0) + 1
        out["by_category_x_suite"] = [
            {"category": c, "suite": s, "count": n}
            for (c, s), n in sorted(agg.items(), key=lambda kv: -kv[1])]
        by_cat = {}
        for r in rows:
            by_cat[r.get("category", "?")] = by_cat.get(
                r.get("category", "?"), 0) + 1
        out["by_category"] = dict(sorted(by_cat.items(),
                                         key=lambda kv: -kv[1]))
    return out
