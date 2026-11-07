"""HTTP transport for --machine: run an op on a registered machine.

`runtool <op> ... --machine NAME` forwards the op verbatim (the raw argv
tokens, --machine stripped) to that machine's web server (POST /api/runtool),
prints the response byte-identically to what the CLI would print locally,
and exits with the remote exit code. Without --machine nothing here runs.

Registry: <runs_root>/machines.json — hand-edited, gitignored, same
discipline as datasets.json (loud errors, never a silent empty fallback):

    {"big": {"url": "http://big-host:8675",
             "token": "<that machine's AI_TOOLKIT_AUTH>"}}

wait is a client-side poll loop of short `status` calls: silent while the
server is healthy, and if it dies the agent's turn regains control with the
failure, how long the wait ran, and the run's last known state. Each poll
is a real remote status, so queued chains advance exactly as they do
locally; the deadline (explicit --timeout-s or the manifest timeout the
remote classify reports) matches local wait semantics.

The client never re-wraps a remote verdict: remote stdout passes through
verbatim and the remote exit code becomes ours. New envelopes exist only
for transport-level failures (unreachable / auth), printed in the CLI's
error style: one compact JSON line, exit 2.
"""

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

from .registry import ParamError
from .runner import get_runs_root

POLL_INTERVAL_S = 2.0
CONNECT_TIMEOUT_S = 10.0
# read timeout per status poll: a healthy status call answers in seconds;
# exceeding it means the server is gone or wedged
_STATUS_READ_S = 30.0
# transport guards for one-shot ops, not semantics (gc size-walks and
# dataset copies are the slow ones)
_DEFAULT_READ_S = 600.0
_LONG_READ_S = 1800.0
_LONG_OPS = ("gc", "datasets")
# a dead server must reach the agent's turn in seconds, but one lost
# packet must not surrender a two-hour wait
_RETRY_ATTEMPTS = 3
_RETRY_GAP_S = 1.0


def registry_path():
    """The machine registry file, under the runs root; re-resolved on
    every call (the runs root follows the UI's TRAINING_FOLDER setting)."""
    return os.path.join(get_runs_root(), "machines.json")


def _load_registry():
    try:
        with open(registry_path()) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ParamError(
            f"no machine registry at {registry_path()!r}: add machines to "
            'it as {"name": {"url": "http://host:8675", "token": "..."}}'
        )
    except json.JSONDecodeError as e:
        raise ParamError(f"machine registry {registry_path()} is corrupt: {e}")
    if not isinstance(data, dict):
        raise ParamError(
            f"machine registry {registry_path()} must be an object mapping "
            "machine names to {url, token} entries"
        )
    return data


class _TransportFailure(Exception):
    """A transport-level failure: no remote verdict exists for it."""

    def __init__(self, message, kind="unreachable"):
        super().__init__(message)
        self.kind = kind


def _print_envelope(obj):
    """Transport failures print in the CLI's error style: one compact
    JSON line, exit 2 (cli.py prints its envelopes the same way)."""
    print(json.dumps(obj, default=str))
    return 2


def _call(entry, machine, tokens, read_timeout_s):
    """One POST to the machine's route; returns the parsed envelope
    {exit_code, stdout, stderr}. Raises _TransportFailure when no remote
    verdict could be obtained."""
    url = entry["url"].rstrip("/") + "/api/runtool"
    body = json.dumps({"argv": list(tokens)}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {entry.get('token') or ''}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=read_timeout_s) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise _TransportFailure(
                f"machine {machine!r} rejected the token (HTTP {e.code})",
                kind="auth",
            )
        raise _TransportFailure(
            f"machine {machine!r} returned HTTP {e.code} ({url})")
    except (urllib.error.URLError, TimeoutError, OSError, socket.timeout) as e:
        raise _TransportFailure(
            f"machine {machine!r} unreachable: {e} ({url})")
    try:
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or "exit_code" not in payload:
            raise ValueError("envelope has no exit_code")
    except (ValueError, UnicodeDecodeError) as e:
        raise _TransportFailure(
            f"machine {machine!r} returned non-JSON output "
            f"(HTTP {status}): {e}")
    return payload


def _emit(payload):
    """Print the remote CLI's output verbatim and return its exit code.
    stdout is the contract; stderr only surfaces when stdout is empty
    (argparse usage errors and tracebacks bypass the JSON contract)."""
    out = payload.get("stdout") or ""
    if not out.strip():
        out = payload.get("stderr") or ""
    if out and not out.endswith("\n"):
        out += "\n"
    sys.stdout.write(out)
    code = payload.get("exit_code")
    if isinstance(code, bool) or not isinstance(code, int):
        return 2
    return code


def list_machines():
    """Agent-facing registry view: names, urls, notes — never tokens.
    The registry path is included so the owner can be pointed at the
    exact file; a missing registry is a loud teaching error, matching
    the datasets registry discipline (the silent empty fallback is how a
    wiped registry once read as 'unconfigured')."""
    registry = _load_registry()  # ParamError propagates: teaching message
    machines = []
    for name in sorted(registry):
        entry = registry[name]
        if not isinstance(entry, dict):
            continue
        machines.append({
            "name": name,
            "url": entry.get("url"),
            "auth": "set" if entry.get("token") else "unset",
            **({"note": entry["note"]} if entry.get("note") else {}),
        })
    return {"registry": registry_path(), "machines": machines}


def _strip_machine_tokens(tokens):
    """The exact user tokens, minus --machine / --machine=... — the remote
    does its own json.loads on --params/--expect/--config-patch, so no
    value is ever re-serialized here."""
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--machine":
            i += 2  # flag and its value
            continue
        if t.startswith("--machine="):
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def delegate(machine, op, raw_tokens, args=None):
    """The --machine boundary: every failure here prints one envelope and
    returns 2; every remote verdict passes through verbatim with the
    remote's own exit code."""
    try:
        registry = _load_registry()
    except ParamError as e:
        return _print_envelope({"error": str(e), "type": "rejected"})
    entry = registry.get(machine)
    if not isinstance(entry, dict) or not entry.get("url"):
        known = ", ".join(sorted(registry)) or "none"
        return _print_envelope({
            "error": f"unknown machine {machine!r}; known: {known} — "
                     f"edit {registry_path()}",
            "type": "rejected",
        })
    tokens = _strip_machine_tokens(raw_tokens)
    if op == "wait":
        return _remote_wait(machine, entry, tokens, args)
    read_s = _LONG_READ_S if op in _LONG_OPS else _DEFAULT_READ_S
    try:
        payload = _call(entry, machine, tokens, read_s)
    except _TransportFailure as e:
        return _print_envelope(
            {"error": str(e), "type": e.kind, "machine": machine})
    return _emit(payload)


def _parse_wait_tokens(tokens):
    """run_id and --timeout-s from the raw wait tokens (already validated
    by the local argparse; values are read, never re-serialized)."""
    run_id = None
    timeout_s = None
    i = 1  # skip the op name
    while i < len(tokens):
        t = tokens[i]
        if t == "--timeout-s":
            timeout_s = float(tokens[i + 1])
            i += 2
        elif t.startswith("--timeout-s="):
            timeout_s = float(t.split("=", 1)[1])
            i += 1
        else:
            if run_id is None and not t.startswith("-"):
                run_id = t
            i += 1
    return run_id, timeout_s


def _remote_wait(machine, entry, tokens, args):
    """Poll remote `status` until terminal or deadline. Silent while
    healthy; a dead server returns the agent's turn in seconds with
    waited_s and the run's last known state."""
    from .complain import complaint_hint

    run_id, timeout_s = _parse_wait_tokens(tokens)
    if args is not None:
        # the locally-parsed values are authoritative (argparse already
        # validated them); the token scan is the fallback
        run_id = run_id or getattr(args, "run_id", None)
        if getattr(args, "timeout_s", None) is not None:
            timeout_s = args.timeout_s
    if run_id is None:
        return _print_envelope({
            "error": "wait needs a run_id", "type": "rejected"})

    start = time.time()
    deadline = start + timeout_s if timeout_s is not None else None
    last_status = None
    status_tokens = ["status", run_id]

    def surrender(err):
        env = {"error": str(err), "type": err.kind, "machine": machine,
               "waited_s": round(time.time() - start, 1)}
        if last_status is not None:
            env["last_status"] = last_status
        return _print_envelope(env)

    while True:
        try:
            payload = _call(entry, machine, status_tokens, _STATUS_READ_S)
        except _TransportFailure as e:
            for _ in range(_RETRY_ATTEMPTS):
                time.sleep(_RETRY_GAP_S)
                try:
                    payload = _call(entry, machine, status_tokens,
                                    _STATUS_READ_S)
                    break
                except _TransportFailure:
                    payload = None
            if payload is None:
                return surrender(e)
        try:
            st = json.loads(payload.get("stdout") or "")
        except ValueError:
            return _emit(payload)  # remote bug: show it like local would
        if not isinstance(st, dict) or "type" in st or "status" not in st:
            # not_found / rejected envelope: passthrough, byte-identical
            return _emit(payload)
        last_status = st
        if deadline is None:
            # the manifest deadline, reported by the remote classify —
            # the same default runner.wait uses
            deadline = start + (st.get("timeout_s") or 3600)
        status = st.get("status")
        if status not in ("running", "queued"):
            if status in ("crash", "failed"):
                st["you_may_file"] = complaint_hint(run_id)
            print(json.dumps(st, indent=1, default=str))
            return 0 if status not in ("crash", "failed", "timeout") else 1
        if time.time() >= deadline:
            st["status"] = "timeout"
            st["you_may_file"] = complaint_hint(run_id)
            print(json.dumps(st, indent=1, default=str))
            return 1
        time.sleep(POLL_INTERVAL_S)
