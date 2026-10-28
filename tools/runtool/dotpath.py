"""Dotted config paths for the loud-merge layers (F2).

Syntax: "config.process[0].train.lr" -- dot-separated keys, [n] list
indices. Parse produces a segment list; get/set walk a parsed spec.
Missing intermediate containers are NOT created by set (a patch must
land on a real field; inventing structure silently is the same class
of bug the merge contract kills).
"""

import re

_SEG = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def parse(path):
    segs = []
    pos = 0
    for m in _SEG.finditer(path):
        if m.start() > pos and path[pos:m.start()].strip("."):
            raise ValueError(f"bad config path {path!r}")
        key, idx = m.group(1), m.group(2)
        if key is not None:
            segs.append(key)
        else:
            segs.append(int(idx))
        pos = m.end()
    if not segs or pos != len(path):
        raise ValueError(f"bad config path {path!r}")
    return segs


MISSING = object()


def get(obj, segs, default=MISSING):
    cur = obj
    for s in segs:
        try:
            cur = cur[s]
        except (KeyError, IndexError, TypeError):
            return default
    return cur


def setval(obj, segs, value, allow_new=True):
    """Set a leaf. allow_new=False (the patch layer) refuses any path
    whose parent or key does not already exist -- patches adjust real
    fields, they never scaffold structure silently. The param layer may
    create keys the template omitted (that fact lands in provenance)."""
    cur = obj
    for s in segs[:-1]:
        try:
            cur = cur[s]
        except (KeyError, IndexError, TypeError):
            raise KeyError("no such structure: " + ".".join(map(str, segs)))
    leaf = segs[-1]
    if isinstance(cur, dict):
        if leaf not in cur and not allow_new:
            raise KeyError("no such key: " + ".".join(map(str, segs)))
        cur[leaf] = value
    elif isinstance(cur, list):
        if not isinstance(leaf, int) or (leaf >= len(cur) and not allow_new):
            raise KeyError("no such index: " + ".".join(map(str, segs)))
        if leaf == len(cur) and allow_new:
            cur.append(value)
        else:
            cur[leaf] = value
    else:
        raise KeyError("no such structure: " + ".".join(map(str, segs)))
    return value


def fmt(segs):
    out = ""
    for s in segs:
        out += f"[{s}]" if isinstance(s, int) else (f".{s}" if out else s)
    return out
