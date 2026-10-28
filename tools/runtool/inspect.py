"""Diagnostic digest for the unknown-unknown case (spec F1b increment).

expect grades what you PREDICTED. inspect answers what you DIDN'T:
something looks strange, nobody knew what to look for. One call, built
from the record alone:

  * generic ugly-line signals (traceback / OOM / NaN / Killed...) with
    counts and the last occurrence -- no expectation required;
  * idle/stall numbers for live runs (seconds since last event / last
    log byte) -- "hung at 100% GPU" becomes a number;
  * repeated-line spam (retry loops surface themselves);
  * the raw logs: --tail N lines (0 = the whole file), and --grep RX
    over the full record with line numbers and context. The sanctioned
    raw-text path lives IN the tool, so the agent never needs a
    free-range grep to read what its own run said;
  * --show-config inlines the derived config with one mechanical wall:
    a path belonging to a registered REAL dataset is replaced by its
    registry key. Tool (rt_) folders pass through untouched -- the
    guard means current runs can only reference them anyway.
  * last events, marker positions (pct through stdout), record sizes.

Power-tool contract: nothing is clamped or hidden by default beyond the
dataset-name wall enforced mechanically from the registry; tails are
uncapped, greps are the caller's regex. Read buffers use a generous cap
purely to bound memory; when a file exceeds it the response SAYS so
(truncated) rather than silently clipping.
"""

import json
import os
import re
import time
from collections import Counter

# conservative on purpose: noisy patterns inflate counts until the digest
# reads as wallpaper. Suites can extend via Suite.augment_signals later;
# this set is the generic floor.
UGLY = {
    "traceback": r"Traceback \(most recent call last\)",
    "error": r"(?:^|\s)(?:ERROR|Error:)",
    "exception": r"\w+(?:Error|Exception):",
    "killed": r"Killed|SIGKILL|Signal 9",
    "oom": r"out of memory|OOM|CUDA error",
    "nan": r"\bnan\b.*loss|loss.*\bnan\b",
    "segfault": r"Segmentation fault|core dumped",
    "warning": r"(?:^|\s)(?:WARNING|Warning:)",
}

# memory bound for reads, not an intelligence clamp: exceeding it is
# REPORTED ("truncated": true), never silently applied
FULL_CAP = 64 * 1024 * 1024
GREP_MAX_HITS = 200


def _read(path):
    """(text, size, truncated) for the whole file up to FULL_CAP."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - FULL_CAP)
            f.seek(start)
            return f.read().decode("utf-8", "replace"), size, start > 0
    except FileNotFoundError:
        return "", 0, False


def _tail(path, n):
    """Last n lines; n <= 0 means the whole file."""
    text, _, _ = _read(path)
    lines = text.splitlines()
    return "\n".join(lines if n <= 0 else lines[-n:])


def signals(paths):
    out = {}
    text, _, _ = _read(paths.stdout)
    err, _, _ = _read(paths.stderr)
    combined = text + "\n" + err
    for name, rx in UGLY.items():
        try:
            hits = [ln for ln in combined.splitlines() if re.search(rx, ln)]
        except re.error:
            continue
        if not hits:
            continue  # healthy runs: say nothing about nothing
        out[name] = {"count": len(hits), "last_line": hits[-1].strip()[:300]}
    return out


def grep(paths, pattern, context=0, files=("stdout", "stderr")):
    """Caller's regex over the full record, with line numbers and
    optional surrounding lines. Capped at GREP_MAX_HITS with an honest
    'more' flag -- the cap is disclosed, not a silently short answer."""
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"--grep: not a valid regex: {e}")
    hits, more, truncated = [], False, False
    for which in files:
        path = getattr(paths, which)
        text, _, trunc = _read(path)
        truncated = truncated or trunc
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if not rx.search(ln):
                continue
            if len(hits) >= GREP_MAX_HITS:
                more = True
                break
            block = lines[max(0, i - context):i + context + 1]
            hits.append({
                "file": which, "line": i + 1,
                "text": "\n".join(block)[:600],
            })
    return {"pattern": pattern, "hits": hits, "more": more,
            "truncated": truncated}


def repeats(paths, top=3, min_count=5):
    text, _, _ = _read(paths.stdout)
    c = Counter(ln.strip() for ln in text.splitlines() if ln.strip())
    spam = [(n, ln) for ln, n in c.items() if n >= min_count
            and len(ln) < 300]
    spam.sort(reverse=True)
    return [{"count": n, "line": ln} for n, ln in spam[:top]]


def idle(paths):
    """Seconds since the record last moved (live stall signal)."""
    out = {}
    now = time.time()
    for key, path in (("events_s", paths.events),
                      ("stdout_s", paths.stdout),
                      ("stderr_s", paths.stderr)):
        try:
            out[key] = round(now - os.path.getmtime(path), 1)
        except FileNotFoundError:
            pass
    return out


def events_tail(paths, n=5):
    try:
        with open(paths.events, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            lines = [ln for ln in f.read().decode("utf-8", "replace")
                     .splitlines() if ln.strip()]
        out = []
        for ln in lines[-n:]:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                out.append({"raw": ln[:200]})
        return out
    except FileNotFoundError:
        return []


def marker_positions(paths, suite):
    """Where each found marker first appeared, as % through stdout.
    'Died at 80%' plus the phases fold is the missing-what answer."""
    declared = getattr(suite, "markers", None) or {}
    if not declared:
        return {}
    text, size, _ = _read(paths.stdout)
    if not size:
        return {}
    out = {}
    for name, pattern in declared.items():
        try:
            m = re.search(pattern, text)
        except re.error:
            continue
        if m:
            out[name] = {"pct": round(m.start() / size * 100, 1)}
    return out


def record_sizes(paths):
    out = {}
    try:
        names = sorted(os.listdir(paths.root))
    except FileNotFoundError:
        return out
    for name in names:
        p = os.path.join(paths.root, name)
        if os.path.isfile(p):
            out[name] = os.path.getsize(p)
    if os.path.isdir(paths.artifacts):
        for name in sorted(os.listdir(paths.artifacts)):
            p = os.path.join(paths.artifacts, name)
            if os.path.isfile(p):
                out[f"artifacts/{name}"] = os.path.getsize(p)
    return out


def config_text(paths):
    """The derived config the job ran, with registered REAL dataset
    folders mechanically replaced by their registry keys. Tool (rt_)
    paths pass through. This is the one hard wall, done mechanically:
    the tool cannot print a real dataset name because the substitution
    happens before output, driven by the machine-local registry."""
    path = os.path.join(paths.artifacts, "config.yaml")
    try:
        text = open(path).read()
    except FileNotFoundError:
        return None
    from . import datasets as dsets
    try:
        registry = dsets._load()
    except Exception:
        registry = {}
    for key, entry in registry.items():
        if entry.get("created_by") == "runtool":
            continue  # tool folders are clean vocabulary
        for folder in entry.get("folders", []):
            for variant in {folder, os.path.realpath(folder)}:
                if variant and variant in text:
                    text = text.replace(variant, f"<dataset:{key}>")
    return text


def build(paths, suite, alive, tail_lines=30, grep_pattern=None,
          grep_context=0, show_config=False):
    n = int(tail_lines)  # <= 0 means everything; no upper clamp
    out = {
        "signals": signals(paths),
        "repeats": repeats(paths),
        "events_tail": events_tail(paths),
        "marker_positions": marker_positions(paths, suite),
        "record_sizes": record_sizes(paths),
        "tail_lines": n if n > 0 else "all",
        "stdout_tail": _tail(paths.stdout, n),
        "stderr_tail": _tail(paths.stderr, n),
    }
    if grep_pattern is not None:
        out["grep"] = grep(paths, grep_pattern, grep_context)
    if show_config:
        cfg = config_text(paths)
        out["config"] = cfg if cfg is not None else "<no artifacts/config.yaml>"
    if alive:
        out["idle"] = idle(paths)
    return out
