"""Config vocabulary for runtool, derived from the real code at query time.

Pure-AST scan (imports nothing, executes nothing) of every
config_modules.py in the repo: from each class __init__ it extracts the
kwargs.get surface -- key, literal default, inferred type. This is the
single source of *possible* config keys for the v2 diff-submit path;
when the trainer grows a config option, it appears here with no doc to
update.

Known limits, on purpose:
  * keys whose default is a computed expression (e.g. unet_lr <- self.lr)
    are reported with type "computed" and the source expression -- their
    real types/bounds come later as declarative per-class fields (the
    _FIELDS phase), not from guessing.
  * string-valued enums (optimizer, sampler, scheduler names) are NOT
    validated here. The name sets live in optimizer/kernel dispatch code
    and the standard for surfacing them is the kernel-opt session's to
    decide -- see docs/kernel_opt/PERF_FLAG_STANDARD.md. Inventing a
    parallel enum list here is exactly the rot this avoids.
"""

import ast
import glob
import os

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

CONFIG_SOURCES = [
    "toolkit/config_modules.py",
    "extensions_built_in/**/config_modules.py",
]

# config-section -> config class, for the sections runtool composes.
# Suites may extend this map (an extension with custom config classes
# declares its own sections); unknown sections are reported, never guessed.
DEFAULT_SECTIONS = {
    "train": "TrainConfig",
    "model": "ModelConfig",
    "network": "NetworkConfig",
    "save": "SaveConfig",
    "sample": "SampleConfig",
    "sample.samples[]": "SampleItem",
    "logging": "LoggingConfig",
    "datasets[]": "DatasetConfig",
    "ema": "EMAConfig",
    "validation": "ValidationConfig",
    "embeddings": "EmbeddingConfig",
    "guidance": "GuidanceConfig",
}


def _literal_or_src(node):
    if node is None:
        return None, "any", ""
    try:
        val = ast.literal_eval(node)
        return val, type(val).__name__, repr(val)
    except (ValueError, TypeError):
        try:
            src = ast.unparse(node)
        except AttributeError:  # pragma: no cover
            src = "<expr>"
        return None, "computed", src


def _scan_file(path):
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    out = {}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        fields = {}
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name == "__init__"):
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "kwargs"):
                    continue
                if not (node.args and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    continue
                key = node.args[0].value
                default, kind, src = _literal_or_src(
                    node.args[1] if len(node.args) > 1 else None)
                fields[key] = {"default": default, "type": kind, "src": src}
        if fields:
            out[cls.name] = fields
    return out


def schema():
    """{rel_path: {class_name: {key: {default, type, src}}}} -- derived fresh.

    The scan is milliseconds; caching would only add drift.
    """
    files = []
    for pat in CONFIG_SOURCES:
        files.extend(glob.glob(
            os.path.join(REPO_ROOT, pat), recursive=True))
    out = {}
    for path in sorted(set(files)):
        rel = os.path.relpath(path, REPO_ROOT)
        classes = _scan_file(path)
        if classes:
            out[rel] = classes
    return out


def _class_index(s):
    # class names are unique across config modules in practice; first
    # occurrence wins and the source file stays in the key record
    index = {}
    for rel, classes in s.items():
        for cls, fields in classes.items():
            index.setdefault(cls, {"source": rel, "fields": fields})
    return index


def summarize(section_map=None):
    s = schema()
    idx = _class_index(s)
    sections = dict(DEFAULT_SECTIONS)
    sections.update(section_map or {})
    resolved, unresolved = {}, []
    for section, cls in sections.items():
        if cls in idx:
            resolved[section] = {
                "class": cls,
                "keys": len(idx[cls]["fields"]),
            }
        else:
            unresolved.append(section)
    total = sum(len(e["fields"]) for e in idx.values())
    typed = sum(
        1 for e in idx.values()
        for f in e["fields"].values() if f["type"] != "computed"
    )
    return {
        "classes": {c: len(f["fields"]) for c, f in sorted(idx.items())},
        "sections": resolved,
        "sections_unresolved": unresolved,
        "total_keys": total,
        "literal_defaulted": typed,
    }


def query(section=None, key=None, cls=None):
    """One JSON-ready answer: a whole section's keys, or one key's spec."""
    s = schema()
    idx = _class_index(s)
    sections = dict(DEFAULT_SECTIONS)

    target_cls = cls
    if section and not cls:
        if section not in sections:
            return {"error": f"unknown section {section!r}",
                    "sections": sorted(sections)}
        target_cls = sections[section]
    if target_cls:
        if target_cls not in idx:
            return {"error": f"unknown config class {target_cls!r}",
                    "classes": sorted(idx)}
        fields = idx[target_cls]["fields"]
        if key:
            if key not in fields:
                # the teaching error: nearest real keys, not a shrug
                near = sorted(k for k in fields if key in k or k in key)[:8]
                return {"error": f"{target_cls} has no key {key!r}",
                        "similar": near}
            return {target_cls: {key: fields[key]}}
        return {target_cls: fields}
    if key:
        hits = {
            c: f[key] for c, e in idx.items()
            for f in [e["fields"]] if key in f
        }
        return hits or {"error": f"key {key!r} in no config class"}
    return summarize()
