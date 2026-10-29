"""Machine-local dataset registry for runtool -- the ONLY dataset surface
an agent gets.

Policy this module enforces (owner, 2026-10-27/28):
  * The registry (datasets.json) is machine-local and gitignored, like
    every run record. Nothing dataset-related is ever versioned, so a
    fresh clone is never tied to this machine's datasets.
  * Keys are the vocabulary; real folder paths are not printed to
    agent-facing output (owner flag --show-paths only). A real dataset
    name that never enters tool output cannot leak into commits,
    comments or examples.
  * Agents run against TOOL-CREATED datasets: synthetic content, or a
    COPY of a real dataset (media/captions; the TE/latent/clip caches
    only on explicit request -- a silent cache copy would also silently
    carry stale entries). Real dataset folders stay read-only.
  * Deletion of files is possible ONLY for tool-created entries whose
    folders live under <runs_root>/datasets/ AND carry the tool prefix.
    Removing a registry entry is always possible; that is not deletion.

Registry shape:
  {key: {"folders": [str, ...], "created_by": "runtool"?, "created_at"?}}
"""

import json
import os
import shutil
import time

from .registry import ParamError, safe_run_component
from .runner import RUNS_ROOT

TOOL_PREFIX = "rt_"
CACHE_DIRNAMES = {"_t_e_cache", "_latent_cache", "_clip_vision_cache"}
DATASETS_ROOT = os.path.join(RUNS_ROOT, "datasets")
REGISTRY_PATH = os.environ.get(
    "RUNTOOL_DATASETS", os.path.join(RUNS_ROOT, "datasets.json")
)

# solid palette cycled by the synthetic generator -- content is generic by
# design (never dataset-derived)
_PALETTE = [
    (220, 40, 40), (40, 160, 60), (40, 80, 220),
    (140, 140, 140), (230, 200, 60),
]
_PALETTE_NAMES = ["red", "green", "blue", "gray", "yellow"]


def _load():
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise ParamError(f"registry {REGISTRY_PATH} is corrupt: {e}")


def _save(data):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, REGISTRY_PATH)


def _tool_folder(key):
    return os.path.join(DATASETS_ROOT, f"{TOOL_PREFIX}{key}")


def is_tool_folder(path):
    """A deletable folder: under the tool datasets root, prefixed."""
    real = os.path.realpath(path)
    return (os.path.commonpath([os.path.realpath(DATASETS_ROOT), real])
            == os.path.realpath(DATASETS_ROOT)
            and os.path.basename(real).startswith(TOOL_PREFIX))


def list_entries(show_paths=False):
    out = []
    for key, entry in sorted(_load().items()):
        row = {
            "key": key,
            "folders": len(entry.get("folders", [])),
            "created_by": entry.get("created_by", "external"),
        }
        if show_paths:
            row["paths"] = [
                p for p in entry.get("folders", [])
                if os.path.exists(p)
            ] or ["<missing>"]
        out.append(row)
    return out


def add(key, folders, tool_created=False):
    key = safe_run_component(key)
    data = _load()
    if key in data:
        raise ParamError(
            f"key {key!r} already maps {len(data[key].get('folders', []))} "
            f"folder(s); rm the entry first to re-point it"
        )
    missing = [p for p in folders if not os.path.isdir(p)]
    if missing:
        # count only -- the message must not echo real folder paths
        raise ParamError(f"{len(missing)} of {len(folders)} folder(s) do not exist")
    entry = {"folders": [os.path.abspath(p) for p in folders]}
    if tool_created:
        entry["created_by"] = "runtool"
        entry["created_at"] = round(time.time(), 1)
    data[key] = entry
    _save(data)
    return {"added": key, "folders": len(folders)}


def _copy_tree(src, dst, copy_cache):
    n = 0
    for root, dirs, files in os.walk(src):
        if not copy_cache:
            dirs[:] = [d for d in dirs if d not in CACHE_DIRNAMES]
        rel = os.path.relpath(root, src)
        target = os.path.normpath(os.path.join(dst, rel))
        os.makedirs(target, exist_ok=True)
        for name in files:
            shutil.copy2(os.path.join(root, name), os.path.join(target, name))
            n += 1
    return n


def _synthesize(dst, count, size):
    from PIL import Image
    for i in range(count):
        color = _PALETTE[i % len(_PALETTE)]
        img = Image.new("RGB", (size, size), color)
        base = f"synthetic_{i:03d}"
        img.save(os.path.join(dst, base + ".png"))
        with open(os.path.join(dst, base + ".txt"), "w") as f:
            f.write(f"a synthetic test image on a {_PALETTE_NAMES[i % len(_PALETTE)]} background\n")
    return count * 2


def create(key, from_key=None, from_path=None, synthetic=0,
           size=512, copy_cache=False):
    """Create a tool-owned dataset: synthetic media, or a copy of a
    dataset (registry key or explicit path). The source is never written."""
    key = safe_run_component(key)
    modes = [bool(from_key), bool(from_path), synthetic > 0]
    if sum(modes) != 1:
        raise ParamError(
            "exactly one source: --from-key | --from-path | --synthetic N"
        )
    data = _load()
    if key in data:
        raise ParamError(f"key {key!r} already exists")
    dst = _tool_folder(key)
    if os.path.exists(dst):
        raise ParamError(f"tool folder already exists: {dst}")

    src = from_path
    if from_key:
        if from_key not in data:
            raise ParamError(
                f"source key {from_key!r} not in the registry "
                f"(keys: {sorted(data)})"
            )
        srcs = data[from_key]["folders"]
    else:
        srcs = [src] if src else []

    os.makedirs(dst, exist_ok=True)
    copied = 0
    try:
        if synthetic > 0:
            copied = _synthesize(dst, synthetic, size)
            folders = [dst]
        else:
            # multi-folder sources KEEP their folder structure: for
            # multi-media models the folders ARE the semantics (image
            # set vs video set wire to different template entries, in
            # registry order). Subfolders are neutral (folder_N), never
            # the source names.
            folders = []
            for i, s in enumerate(srcs):
                if not os.path.isdir(s):
                    raise ParamError("source folder does not exist")
                sub = os.path.join(dst, f"folder_{i}")
                copied += _copy_tree(s, sub, copy_cache)
                folders.append(sub)
    except Exception:
        shutil.rmtree(dst, ignore_errors=True)
        raise

    data[key] = {
        "folders": folders,
        "created_by": "runtool",
        "created_at": round(time.time(), 1),
        "files": copied,
        "copy_cache": bool(copy_cache) if not synthetic else False,
    }
    _save(data)
    return {"created": key, "folder": dst, "files": copied}


def rm(key, delete_files=False):
    key = safe_run_component(key)
    data = _load()
    if key not in data:
        raise ParamError(f"key {key!r} not in the registry")
    entry = data[key]

    if not delete_files:
        del data[key]
        _save(data)
        return {"removed": key, "files_deleted": False,
                "detail": "registry entry only; folders untouched"}

    # file deletion: tool-created entries, tool root, tool prefix -- all
    # three, or nothing. Reasons printed, never a silent refusal.
    reasons = []
    if entry.get("created_by") != "runtool":
        reasons.append("entry is not tool-created (registered externally)")
    for p in entry.get("folders", []):
        if not is_tool_folder(p):
            reasons.append(
                f"folder outside tool datasets root or without the "
                f"'{TOOL_PREFIX}' prefix: {os.path.basename(p)}")
    if reasons:
        raise ParamError(
            f"refusing to delete files of {key!r}: " + "; ".join(reasons)
        )
    deleted = []
    for p in entry["folders"]:
        shutil.rmtree(p, ignore_errors=False)
        deleted.append(os.path.basename(p))
    del data[key]
    _save(data)
    return {"removed": key, "files_deleted": True, "folders": deleted}
