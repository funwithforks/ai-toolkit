"""Marker extraction and expectation verdicts (spec F1).

A run's INTENT must be checkable from the tool's own return, not from
grep-slop over stdout. Suites declare markers (name -> regex) matched
against the harness-recorded stdout/stderr; submit accepts an `expect`
dict; every classify/wait/status return carries the extraction and, when
an expect was given, a PASS/FAIL verdict with one line per named check.

Semantics (spec §2):
  * a present-expected marker that never appears is a named FAIL, never
    a shrug -- "a silent no-op is still broken" applied to runs;
  * absent-expected markers FAIL when seen;
  * value expectations compare captured named groups ({"name": value});
  * verdict is PASS only when every check passes.

Derivation contract: everything here reads the record alone (stdout.log
/ stderr.log + the caller's params), so recompute() reproduces stored
verdicts exactly -- same contract as metrics.

Only the HARNESS-recorded logs are scanned (the supervisor launches the
harness with stdout/stderr redirected to the record). The supervisor's
own log is deliberately not scanned: a line that is not in the record
cannot be part of a verdict the record must reproduce.
"""

import json
import os
import re

# live runs scan a bounded tail window; finished runs get a full pass
# (records are ~100KB; the cap only bounds pathological logs)
TAIL_WINDOW = 2 * 1024 * 1024
FULL_CAP = 8 * 1024 * 1024

_EXPECT_TOP = ("markers", "metrics", "job_files")
_METRIC_BOUNDS = ("min", "max", "eq")


def _read(path, full):
    """Full contents (capped), or the last TAIL_WINDOW bytes for live runs."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if full:
                f.seek(max(0, size - FULL_CAP))
            else:
                f.seek(max(0, size - TAIL_WINDOW))
            return f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def scan(paths, suite, full):
    """Declared markers -> {name: {found, line, captures}} via one
    bounded pass over stdout.log (+stderr.log). Suites that declare
    nothing cost nothing."""
    markers = getattr(suite, "markers", None) or {}
    if not markers:
        return {}
    text = _read(paths.stdout, full) + _read(paths.stderr, full)
    compiled = {}
    for name, pattern in markers.items():
        try:
            compiled[name] = re.compile(pattern)
        except re.error as e:
            raise ValueError(
                f"suite {suite.id!r}: marker {name!r} is not a valid "
                f"regex: {e}"
            )
    out = {}
    for name, rx in compiled.items():
        hit = None
        for line in text.splitlines():
            m = rx.search(line)
            if m:
                hit = (m, line)
                break
        if hit is None:
            out[name] = {"found": False}
            continue
        m, line = hit
        entry = {"found": True, "line": line.strip()[:400]}
        if m.groupdict():
            entry["captures"] = {
                k: v for k, v in m.groupdict().items() if v is not None
            }
        out[name] = entry
    return out


def count(paths, suite, name, full):
    """How many times a marker line appears (e.g. VAE sample rounds).
    Cheap standalone count for suites' phase fns."""
    markers = getattr(suite, "markers", None) or {}
    if name not in markers:
        return 0
    text = _read(paths.stdout, full) + _read(paths.stderr, full)
    return len(re.findall(markers[name], text))


# ---------------------------------------------------------------- verdicts

def validate_expect(suite, expect):
    """Submit-process rejection of malformed expectations: every named
    marker/metric must exist, every shape must be one we can evaluate.
    A typo'd expectation must never silently degrade into no check."""
    if not isinstance(expect, dict):
        raise ValueError(f"expect must be an object, got {type(expect).__name__}")
    unknown = [k for k in expect if k not in _EXPECT_TOP]
    if unknown:
        raise ValueError(
            f"expect: unknown section(s) {unknown}; allowed: {list(_EXPECT_TOP)}"
        )
    declared = set(getattr(suite, "markers", None) or {})
    for name, want in (expect.get("markers") or {}).items():
        if name not in declared:
            raise ValueError(
                f"expect.markers: {name!r} is not declared by suite "
                f"{suite.id!r}; declared: {sorted(declared)}"
            )
        ok_shape = (
            (isinstance(want, str) and want in ("present", "absent"))
            or isinstance(want, dict)
        )
        if not ok_shape:
            raise ValueError(
                f"expect.markers.{name}: want 'present'/'absent' or an "
                f"object of named-group values, got {want!r}"
            )
        if isinstance(want, dict):
            caps = _capturable_names(suite, name)
            bad = [k for k in want if k not in caps]
            if bad:
                raise ValueError(
                    f"expect.markers.{name}: {bad} are not named groups of "
                    f"the marker's regex; capturable: {caps}"
                )
    for name, want in (expect.get("metrics") or {}).items():
        if name not in suite.metrics:
            raise ValueError(
                f"expect.metrics: {name!r} is not a declared metric; "
                f"declared: {sorted(suite.metrics)}"
            )
        if isinstance(want, dict):
            bad = [k for k in want if k not in _METRIC_BOUNDS]
            if bad:
                raise ValueError(
                    f"expect.metrics.{name}: {bad} not understood; "
                    f"allowed bounds: {list(_METRIC_BOUNDS)}"
                )
        elif not isinstance(want, (int, float)):
            raise ValueError(
                f"expect.metrics.{name}: number or bounds object, "
                f"got {want!r}"
            )
    if "job_files" in expect:
        raise ValueError(
            "expect.job_files is spec F4 (not built yet); drop it or wait "
            "for the increment that ships it"
        )


def _capturable_names(suite, marker):
    rx = getattr(suite, "markers", {}).get(marker, "")
    try:
        return sorted(re.compile(rx).groupindex)
    except re.error:
        return []


def _num(name, want, got, checks):
    if got is None:
        checks.append({"name": f"metrics.{name}", "want": want,
                       "got": "absent", "status": "fail"})
        return
    if isinstance(want, dict):
        bad = [k for k in want if k not in _METRIC_BOUNDS]
        if bad:  # validate_expect rejects these at submit; belt for direct calls
            checks.append({"name": f"metrics.{name}", "want": want,
                           "got": "invalid-expectation", "status": "fail"})
            return
        ok = (("min" not in want or got >= want["min"])
              and ("max" not in want or got <= want["max"])
              and ("eq" not in want or got == want["eq"]))
        checks.append({"name": f"metrics.{name}", "want": want,
                       "got": got, "status": "pass" if ok else "fail"})
    else:
        ok = got == want
        checks.append({"name": f"metrics.{name}", "want": want,
                       "got": got, "status": "pass" if ok else "fail"})


def evaluate(expect, marker_hits, metrics):
    """expect + extractions -> {verdict, checks:[{name,want,got,status}]}."""
    checks = []
    for name, want in (expect.get("markers") or {}).items():
        hit = marker_hits.get(name, {"found": False})
        if want == "present":
            checks.append({"name": f"markers.{name}", "want": want,
                           "got": "found" if hit.get("found") else "absent",
                           "status": "pass" if hit.get("found") else "fail"})
        elif want == "absent":
            checks.append({"name": f"markers.{name}", "want": want,
                           "got": "found" if hit.get("found") else "absent",
                           "status": "pass" if not hit.get("found") else "fail"})
        else:  # named-group value expectations
            if not hit.get("found"):
                checks.append({"name": f"markers.{name}", "want": want,
                               "got": "absent", "status": "fail"})
            else:
                caps = hit.get("captures", {})
                for gname, gval in want.items():
                    got = caps.get(gname)
                    ok = got is not None and str(gval) == str(got)
                    checks.append({
                        "name": f"markers.{name}.{gname}", "want": gval,
                        "got": got, "status": "pass" if ok else "fail"})
    for name, want in (expect.get("metrics") or {}).items():
        _num(name, want, metrics.get(name), checks)
    verdict = "PASS" if checks and all(
        c["status"] == "pass" for c in checks) else ("FAIL" if checks else "NONE")
    return {"verdict": verdict, "checks": checks}


# ------------------------------------------------------------ moss phases
# The phase fold for the phased-loading contract (TE gate / caption plan /
# fixed-embed source / VAE sample rounds). Generic helpers exist so other
# suites can write their own fold; moss is the reference.

def moss_phases(paths, suite, full):
    hits = scan(paths, suite, full)
    out = {}
    te = hits.get("te_skip", {})
    tl = hits.get("te_load", {})
    if te.get("found"):
        out["te"] = "skipped"
    elif tl.get("found"):
        reason = hits.get("te_reason", {}).get("captures", {}).get("reason")
        out["te"] = f"loaded: {reason}" if reason else "loaded"
    plan = hits.get("plan", {})
    if plan.get("found"):
        out["cache_plan"] = plan.get("captures", {})
    fixed = hits.get("fixed_cache", {})
    cache_now = hits.get("fixed_cache_now", {})
    if fixed.get("found"):
        out["fixed_embeds"] = "cache"
    elif cache_now.get("found"):
        out["fixed_embeds"] = "encoded+cached"
    rounds = count(paths, suite, "vae_round", full)
    if rounds:
        out["vae_sample_rounds"] = rounds
    return out
