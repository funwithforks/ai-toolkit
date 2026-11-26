#!/usr/bin/env python3
"""Cold-path download probe for the comfy-candidates door (phased models).

The bug this guards: a component table declares a file under a key/repo
that does not host it, and a warm MODELS_PATH hides the broken download
branch forever (54 days for the H3 text encoder, wrong-org 404s). A warm
run can never catch it; this cold probe can, so it runs instead of on
trust.

Scope is phased loading. A class is in scope when its file lives in a
model package that contains a ``PhasedLoadMixin`` subclass, or (shared
``toolkit/models`` classes) any phased package source references the
class by name. Pre-phased models are not covered and not maintained
(owner ruling; AGENTS.md Content/scope rules) -- ``--all`` extends the
sweep to them for information, same verdicts.

Two verifications per declaring class:
  1. table-driven: every (repo, file) the declared entries can request,
     following the ``resolve_comfy_weights`` precedence (entry
     ``{"repo": ...}`` > hub-style key > class ``aitk_comfy_repo``) via
     the shared helpers.
  2. door-driven: the real classmethod is called per table key with the
     downloader patched to record-and-abort under an empty MODELS_PATH;
     the pair the door actually asks for is probed, and must match the
     table's declaration -- a door/table disagreement (DRIFT) means the
     door and this probe diverged, which is a failure. A local hit in
     cold mode is a failure (environment not cold). ``None``/raise on a
     declared key is a failure (the door stopped honoring the table).

Every pair gets a zero-byte ranged GET (redirects followed, Range
preserved): 200/206 serve; 401/403 without a token pass as GATED; with
HF_TOKEN they must be tree-confirmed. Negative canaries (the historical
wrong-org requests) must 404 -- if one serves, the hub layout changed or
the canary is stale, and that is a failure too. The TE weak spot gets
its own sweep: tokenizer/processor json assets (class
``aitk_tokenizer_repo``/``aitk_processor_repo`` links plus the H3
holder's hardcoded original-repo dirs) bypass the door entirely, so
they are listed via the tree API and ranged-probed with the same
verdicts.

Run after ANY change to an ``aitk_comfy_weight_names`` table, the door
(``toolkit/models/v2/_mixin.py``), or the resolver
(``toolkit/models/v2/resolver.py``).

Usage (repo root, project venv):
  python tools/comfy_probe.py [--endpoint URL] [--filter SUBSTR]
                              [--list] [--all]
Exit 0 = every in-scope pair serves/gates, no DRIFT, no cold hits, no
import errors, all canaries 404.
"""
import argparse
import ast
import gzip
import importlib
import io
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ATTR = "aitk_comfy_weight_names"
MIRROR = "http://10.1.1.83:3143"
# the scaffold model's table names a placeholder repo that hosts nothing
EXCLUDE_DIRS = {"example_model", "__pycache__", "tmp"}
SEARCH_ROOTS = [
    os.path.join(REPO, "toolkit", "models"),
    os.path.join(REPO, "extensions_built_in"),
]
EXT_ROOT = os.path.join(REPO, "extensions_built_in")

# historical wrong-org requests (the pre-fix declarations): the hub must
# NOT host these, or the check has lost its teeth
CANARIES = [
    ("Comfy-Org/Krea-2",
     "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    ("Comfy-Org/MiniMax-H3",
     "diffusion_models/minimax_h3_image_pruned_bf16.safetensors"),
    ("Comfy-Org/MiniMax-H3",
     "diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot"
     ".safetensors"),
]

# the TE weak spot: tokenizer/processor assets bypass the door entirely
# (transformers pulls them from the hub), so the safetensors sweep cannot
# see them. The H3 holder hardcodes its tokenizer/processor host:
TE_JSON_DIRS = [
    ("MiniMaxAI/MiniMax-H3", "FL2VA/tokenizer"),
    ("MiniMaxAI/MiniMax-H3", "FL2VA/processor"),
]
JSON_EXTS = (".json", ".txt", ".model", ".jinja", ".tiktoken")


class _Probe(Exception):
    """Raised by the patched downloader, carrying the requested pair."""

    def __init__(self, repo, filename):
        self.repo, self.filename = repo, filename


def _fetch(url, token, headers=None, timeout=30, _left=6):
    """GET with Range preserved across redirects (HF 307s /resolve/)."""
    req = urllib.request.Request(url, headers=dict(headers or {}),
                                 method="GET")
    req.add_header("User-Agent", "aitk-cold-probe")
    req.add_header("Accept-Encoding", "gzip")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                data = gzip.GzipFile(fileobj=io.BytesIO(data)).read()
            return resp.status, data, None
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308) and _left:
            loc = e.headers.get("Location")
            if loc:
                return _fetch(urllib.request.urljoin(url, loc), token,
                              headers, timeout, _left - 1)
        return e.code, None, None
    except Exception as e:  # network down etc.: report, never hide
        return None, None, f"{type(e).__name__}: {e}"


def http_status(url, token):
    status, _, err = _fetch(url, token, {"Range": "bytes=0-0"})
    return status, err


def tree_contains(endpoint, repo, file_path, token):
    dir_ = os.path.dirname(file_path)
    url = (f"{endpoint}/api/models/{repo}/tree/main/{dir_}" if dir_
           else f"{endpoint}/api/models/{repo}/tree/main")
    status, data, err = _fetch(url, token)
    if status != 200 or data is None:
        return f"TREE-ERR({status or err})"
    try:
        parsed = json.loads(data)
    except Exception:
        return "TREE-ERR(parse)"
    if isinstance(parsed, dict):
        return "TREE-MISS" if "error" in parsed else "TREE-ERR"
    return ("TREE-OK" if any(e.get("path") == file_path for e in parsed)
            else "TREE-MISS")


# ------------------------------------------------------------------
# discovery: text mention -> import -> runtime class attribute
# ------------------------------------------------------------------

def _iter_py(root, skip_pkgs=()):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        rel_dir = os.path.relpath(dirpath, REPO)
        if root == EXT_ROOT and rel_dir.split(os.sep)[1:2] and \
                rel_dir.split(os.sep)[1] in skip_pkgs:
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _reads(path):
    try:
        return open(path, encoding="utf-8").read()
    except Exception:
        return ""


CATEGORY_MULTI = {"diffusion_models"}  # models live in subdirs of these


def _pkg_dir_of(rel_parts):
    """Model-package dir name ('diffusion_models/minimax_h3', 'llm_models')
    from a file's path parts under extensions_built_in, or None for the
    registry glue files that sit directly in a multi-model category."""
    if not rel_parts:
        return None
    if len(rel_parts) == 1:
        return None if rel_parts[0] in CATEGORY_MULTI else rel_parts[0]
    if rel_parts[0] in CATEGORY_MULTI:
        return f"{rel_parts[0]}/{rel_parts[1]}"
    return rel_parts[0]


def phased_pkgs_and_corpus():
    """(set of phased model-package dirs, concatenated phased source).

    A package is phased when any file in its dir tree names
    ``PhasedLoadMixin``; the corpus for shared-class reference checks is
    those packages' sources only."""
    pkgs = set()
    for dirpath, dirnames, filenames in os.walk(EXT_ROOT):
        if "__pycache__" in dirpath:
            continue
        parts = os.path.relpath(dirpath, EXT_ROOT).split(os.sep)
        parts = [] if parts == ["."] else parts
        pkg = _pkg_dir_of(parts)
        if pkg is None or pkg in pkgs:
            continue
        if any("PhasedLoadMixin" in _reads(os.path.join(dirpath, fn))
               for fn in filenames if fn.endswith(".py")):
            pkgs.add(pkg)
    corpus = []
    for pkg in pkgs:
        for path in _iter_py(os.path.join(EXT_ROOT, *pkg.split("/"))):
            corpus.append(_reads(path))
    return pkgs, "\n".join(corpus)


def declaring_names(src):
    """Class names assigning the table (class body or module level)."""
    names = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == ATTR
                        for t in stmt.targets):
                    names.add(node.name)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == ATTR
                for t in node.targets):
            val = node.value
            # X.aitk_comfy_weight_names = ... -> X may be the class
            if isinstance(node.targets[0], ast.Attribute) and \
                    isinstance(node.targets[0].value, ast.Name):
                names.add(node.targets[0].value.id)
            del val
    return names


def discover():
    """[(rel_path, class_name)] in scope (default: phased only)."""
    pkgs, corpus = phased_pkgs_and_corpus()
    found, skipped = [], []
    for root in SEARCH_ROOTS:
        for path in _iter_py(root):
            rel = os.path.relpath(path, REPO)
            src = _reads(path)
            if ATTR not in src:
                continue
            names = declaring_names(src)
            if not names:
                # mentions the attribute without assigning it (docstring,
                # helper): import it anyway and let the runtime scan rule
                # it in if a class there carries a table
                names = {"*"}
            parts = rel.split(os.sep)
            if parts[0] == "extensions_built_in":
                pkg = _pkg_dir_of(parts[1:])
                if pkg is None or pkg not in pkgs:
                    skipped.append(f"{rel} (pre-phased or glue)")
                    continue
            else:  # shared toolkit/models class: name-referenced by a
                   # phased package?
                if not any(re.search(rf"\b{re.escape(n)}\b", corpus)
                           for n in names):
                    skipped.append(f"{rel} (class name unreferenced by "
                                   "phased pkgs)")
                    continue
            for n in names:
                found.append((rel, n))
    return found, skipped, pkgs


def load_classes(found):
    """[(label, cls, table)] de-duplicated by class, plus import errors."""
    classes, errors, seen = [], [], set()
    for rel, cls_name in found:
        dotted = rel[:-3].replace(os.sep, ".").replace("-", "_")
        try:
            mod = importlib.import_module(dotted)
        except Exception as e:
            errors.append(f"{rel}: {type(e).__name__}: {e}")
            continue
        if cls_name == "*":
            cands = [c for _, c in vars(mod).items()
                     if isinstance(c, type) and ATTR in getattr(
                         c, "__dict__", {})]
        else:
            c = getattr(mod, cls_name, None)
            cands = [c] if isinstance(c, type) else []
            if not cands:
                errors.append(f"{rel}::{cls_name}: class not found")
                continue
        hit = False
        for c in cands:
            table = getattr(c, "__dict__", {}).get(ATTR, None)
            if table is None:  # inherited-only: the declaring class owns it
                table = getattr(c, ATTR, None)
                if not isinstance(table, dict) or table is None:
                    errors.append(f"{rel}::{c.__name__}: no dict table")
                    continue
                # attribute inherited from a base in another module: the
                # base's declaring entry covers it; only probe when the
                # table literally lives on this class
                owner = next((b for b in c.__mro__ if ATTR in
                              getattr(b, "__dict__", {})), None)
                if owner is not c:
                    continue
            if not isinstance(table, dict):
                errors.append(f"{rel}::{c.__name__}: table is not a dict")
                continue
            key = (c.__module__, c.__qualname__, id(table))
            if key in seen:
                hit = True  # already probed via another discovery path
                continue
            seen.add(key)
            classes.append((f"{c.__module__.split('.')[-1]}.{c.__name__}",
                            c, table))
            hit = True
        if not hit and cls_name != "*":
            errors.append(f"{rel}::{cls_name}: declares nothing at runtime")
    return classes, errors


# ------------------------------------------------------------------
# expectations + door driving
# ------------------------------------------------------------------

def expected_pairs(cls, key):
    """(repo, file) set the door should request for this key."""
    from toolkit.models.v2.resolver import (
        candidate_file, repo_id_from_name_or_path)
    entry = cls.aitk_comfy_weight_names.get(key)
    repo = getattr(cls, "aitk_comfy_repo", None)
    # the holder calls the door with the repo id (str) plus subfolder for
    # tuple keys, so the hub-style key rule applies to key[0]
    key_str = key[0] if isinstance(key, tuple) else key
    if isinstance(key_str, str):
        repo = repo_id_from_name_or_path(key_str, repo)
    files = entry
    if isinstance(entry, dict):
        repo = entry.get("repo", repo)
        files = entry.get("files")
    out = set()
    for f in files or []:
        own = f[0] if isinstance(f, (list, tuple)) else repo
        fn = candidate_file(f)
        out.add((own, fn))
    return out


def drive_door(cls, key):
    """Call the real door cold; ('request', repo, file) | ('local', path)
    | ('none', None, None) | ('error', summary, None)."""
    import huggingface_hub

    def fake_download(repo_id, filename, **kw):
        raise _Probe(repo_id, filename)

    original = huggingface_hub.hf_hub_download
    huggingface_hub.hf_hub_download = fake_download
    # tuple keys ride the door the way holders pass them: repo id plus
    # subfolder, not the tuple itself
    if isinstance(key, tuple):
        name_or_path, subfolder = key[0], key[1]
    else:
        name_or_path, subfolder = key, None
    try:
        out = cls.resolve_comfy_weights(name_or_path, subfolder=subfolder)
    except _Probe as p:
        return ("request", p.repo, p.filename)
    except Exception as e:
        return ("error", f"{type(e).__name__}: {e}", None)
    finally:
        huggingface_hub.hf_hub_download = original
    return ("none", None, None) if out is None else ("local", str(out),
                                                     None)


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=None)
    ap.add_argument("--endpoint", default=os.environ.get(
        "AITK_PROBE_ENDPOINT", MIRROR))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="also sweep pre-phased models (same verdicts)")
    args = ap.parse_args()
    endpoint = args.endpoint.rstrip("/")
    token = os.environ.get("HF_TOKEN") or None

    # cold environment before any toolkit import: empty MODELS_PATH so the
    # door can only reach the downloader
    cold = tempfile.mkdtemp(prefix="aitk_cold_probe_")
    os.environ["MODELS_PATH"] = cold

    found, skipped, pkgs = discover()
    if args.all:
        # extend to pre-phased packages too
        all_found = []
        for root in SEARCH_ROOTS:
            for path in _iter_py(root):
                rel = os.path.relpath(path, REPO)
                src = _reads(path)
                if ATTR not in src:
                    continue
                for n in (declaring_names(src) or {"*"}):
                    all_found.append((rel, n))
        found, skipped = all_found, []

    classes, errors = load_classes(found)
    if not classes and not args.list:
        print("comfy_probe: no declaring classes discovered -- FAIL")
        for e in errors:
            print(f"  IMPORT-FAILED {e}")
        return 1

    print(f"comfy_probe: {len(pkgs)} phased pkgs, {len(classes)} declaring "
          f"classes, {len(skipped)} pre-phased files skipped")

    from toolkit.models.v2 import resolver
    resolver.MODELS_PATH = cold

    status_cache = {}
    failures, gated_n, ok_n = [], 0, 0

    def probe_pair(repo, file_path, origin):
        nonlocal gated_n, ok_n
        pair = f"{repo}/{file_path}" if repo else str(file_path)
        if args.filter and args.filter not in pair:
            return
        if args.list:
            print(f"LIST  {origin:44s} {pair}")
            return
        if repo is None or not isinstance(file_path, str) \
                or not file_path.endswith(".safetensors") or "<" in pair \
                or "://" in pair:
            failures.append(f"{origin}: MALFORMED {pair}")
            print(f"FAIL  {origin:44s} {pair} [MALFORMED]")
            return
        if pair in status_cache:
            verdict = status_cache[pair]
        else:
            status, err = http_status(
                f"{endpoint}/{repo}/resolve/main/{file_path}", token)
            if status in (200, 206):
                verdict = "OK"
                ok_n += 1
            elif status in (401, 403):
                if token:
                    v = tree_contains(endpoint, repo, file_path, token)
                    verdict = "GATED" if v == "TREE-OK" else f"FAIL:{v}"
                else:
                    verdict = "GATED"
                if verdict == "GATED":
                    gated_n += 1
            elif status is None:
                verdict = f"FAIL:NET {err}"
            else:
                verdict = f"FAIL:{status}"
            status_cache[pair] = verdict
        if verdict in ("OK", "GATED"):
            print(f"{verdict:5s} {origin:44s} {pair}")
        else:
            failures.append(f"{origin}: {pair} [{verdict}]")
            print(f"FAIL  {origin:44s} {pair} [{verdict}]")

    for label, cls, table in classes:
        for key in table:
            exp = expected_pairs(cls, key)
            for repo, fn in sorted(exp):
                probe_pair(repo, fn, f"{label} key={key!r}")
            kind, d_repo, d_file = drive_door(cls, key)
            if kind == "request":
                if (d_repo, d_file) not in exp:
                    failures.append(
                        f"{label} key={key!r}: DRIFT door requested "
                        f"{d_repo}/{d_file}, table declares "
                        f"{sorted(exp)}")
                    print(f"FAIL  {label + ' key=' + repr(key):44s} "
                          f"{d_repo}/{d_file} [DRIFT]")
                else:
                    probe_pair(d_repo, d_file, f"{label} door key={key!r}")
            elif kind == "local":
                failures.append(f"{label} key={key!r}: cold hit -- door "
                                f"returned local path {d_repo}")
                print(f"FAIL  {label:44s} key={key!r} [COLD-HIT "
                      f"{d_repo}]")
            elif kind == "none":
                failures.append(f"{label} key={key!r}: door returned None "
                                "for a declared key")
                print(f"FAIL  {label:44s} key={key!r} [DOOR-NONE]")
            else:
                failures.append(f"{label} key={key!r}: door raised "
                                f"{d_repo}")
                print(f"FAIL  {label:44s} key={key!r} [DOOR-RAISE "
                      f"{d_repo}]")

    # TE json side (tokenizer/processor): listed via the tree API and
    # ranged-probed, same verdicts; safetensors in those dirs are covered
    # by the table sweep above
    json_jobs = [(f"holder {repo}", repo, d) for repo, d in TE_JSON_DIRS]
    for label, cls, table in classes:
        for attr, sub in (("aitk_tokenizer_repo", "aitk_tokenizer_subfolder"),
                          ("aitk_processor_repo", "aitk_processor_subfolder")):
            r = getattr(cls, attr, None)
            if isinstance(r, str) and r:
                json_jobs.append((f"{label}.{attr}", r,
                                  getattr(cls, sub, None) or ""))
    for origin, repo, dir_ in json_jobs:
        url = (f"{endpoint}/api/models/{repo}/tree/main/{dir_}" if dir_
               else f"{endpoint}/api/models/{repo}/tree/main")
        status, data, err = _fetch(url, token)
        if status in (401, 403) and not token:
            gated_n += 1
            print(f"GATED {origin:44s} {repo}/{dir_} (listing)")
            continue
        if status != 200 or data is None:
            failures.append(f"{origin}: tree listing {repo}/{dir_} "
                            f"[{status or err}]")
            print(f"FAIL  {origin:44s} {repo}/{dir_} [LIST "
                  f"{status or err}]")
            continue
        try:
            entries = json.loads(data)
        except Exception:
            failures.append(f"{origin}: tree listing unparseable")
            continue
        if isinstance(entries, dict):
            failures.append(f"{origin}: tree listing error "
                            f"{entries.get('error')}")
            print(f"FAIL  {origin:44s} {repo}/{dir_} [LIST "
                  f"{entries.get('error')}]")
            continue
        for e in entries:
            if e.get("type") != "file" or not str(e.get("path", "")).lower().endswith(JSON_EXTS):
                continue
            f = e["path"]
            pair = f"{repo}/{f}"
            if args.filter and args.filter not in pair:
                continue
            if pair in status_cache:
                verdict = status_cache[pair]
            else:
                p_status, p_err = http_status(
                    f"{endpoint}/{repo}/resolve/main/{f}", token)
                if p_status in (200, 206):
                    verdict = "OK"
                    ok_n += 1
                elif p_status in (401, 403):
                    verdict = "FAIL:{p_status}" if token else "GATED"
                    if verdict == "GATED":
                        gated_n += 1
                elif p_status is None:
                    verdict = f"FAIL:NET {p_err}"
                else:
                    verdict = f"FAIL:{p_status}"
                status_cache[pair] = verdict
            if verdict in ("OK", "GATED"):
                print(f"{verdict:5s} {origin:44s} {pair}")
            else:
                failures.append(f"{origin}: {pair} [{verdict}]")
                print(f"FAIL  {origin:44s} {pair} [{verdict}]")

    for repo, file_path in CANARIES:
        pair = f"{repo}/{file_path}"
        if args.list:
            continue
        status, err = http_status(
            f"{endpoint}/{repo}/resolve/main/{file_path}", token)
        if status in (404, 410):
            print(f"CANARY {pair:70s} 404-as-expected")
        elif status is None:
            failures.append(f"canary {pair}: NET {err}")
            print(f"FAIL  canary {pair} [NET {err}]")
        else:
            failures.append(f"canary {pair}: serves [{status}] -- wrong-org "
                            "pair must NOT exist")
            print(f"FAIL  canary {pair} [{status} -- must 404]")

    print(f"\ncomfy_probe: {len(status_cache)} unique pairs, {ok_n} serve, "
          f"{gated_n} gated, {len(failures)} failed, "
          f"{len(errors)} import errors")
    for msg in failures:
        print(f"  FAILED {msg}")
    for msg in errors:
        print(f"  IMPORT-FAILED {msg}")
    for msg in skipped[:6]:
        print(f"  skipped {msg}")
    return 1 if (failures or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
