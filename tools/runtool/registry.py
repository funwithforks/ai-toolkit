"""Suite registry and parameter validation for runtool.

A suite is a declared measurement stage: id, variants, typed params with
defaults/bounds, a fixed command builder, declared metric names, and a
parser that turns the stage's structured output into those metrics. The
command and the parser live here (versioned, checked once); a tool call
may only choose a suite/variant and vary registered params. Suites
without a parser or a done condition fail validation and cannot start.

Structured-output contract per run (see runner.py):
    events.jsonl   appended progress events (never tqdm/log scraping)
    result.json    written by the harness before it exits on success
    exit.code      written by the supervisor after the harness exits
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

RUNTOOL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ParamError(ValueError):
    """Validation failure; raised before launch, nothing starts."""


@dataclass
class Param:
    kind: str  # int | float | bool | str | enum
    default: Any = None
    required: bool = False
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[tuple] = None
    desc: str = ""

    def coerce(self, name: str, raw: Any) -> Any:
        if self.kind == "bool":
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.lower() in ("1", "true", "yes", "on"):
                return True
            if isinstance(raw, str) and raw.lower() in ("0", "false", "no", "off"):
                return False
            raise ParamError(f"param {name!r}: expected bool, got {raw!r}")
        if self.kind in ("int", "float"):
            try:
                val = int(raw) if self.kind == "int" else float(raw)
            except (TypeError, ValueError):
                raise ParamError(f"param {name!r}: expected {self.kind}, got {raw!r}")
            if self.min is not None and val < self.min:
                raise ParamError(f"param {name!r}: {val} below minimum {self.min}")
            if self.max is not None and val > self.max:
                raise ParamError(f"param {name!r}: {val} above maximum {self.max}")
            return val
        if self.kind == "enum":
            if raw not in self.choices:
                raise ParamError(
                    f"param {name!r}: {raw!r} not in choices {list(self.choices)}"
                )
            return raw
        if self.kind == "str":
            if not isinstance(raw, str):
                raise ParamError(f"param {name!r}: expected str, got {raw!r}")
            return raw
        raise ParamError(f"param {name!r}: unknown kind {self.kind!r}")


@dataclass
class RunPaths:
    run_id: str
    root: str  # record dir; manifest/events/result/stdout/stderr live here

    @property
    def manifest(self):
        return os.path.join(self.root, "manifest.json")

    @property
    def events(self):
        return os.path.join(self.root, "events.jsonl")

    @property
    def result(self):
        return os.path.join(self.root, "result.json")

    @property
    def exit_code(self):
        return os.path.join(self.root, "exit.code")

    @property
    def stdout(self):
        return os.path.join(self.root, "stdout.log")

    @property
    def stderr(self):
        return os.path.join(self.root, "stderr.log")

    @property
    def pid(self):
        return os.path.join(self.root, "pid")

    @property
    def cancel(self):
        return os.path.join(self.root, "cancel.flag")

    @property
    def patch(self):
        return os.path.join(self.root, "patch.diff")

    @property
    def artifacts(self):
        return os.path.join(self.root, "artifacts")


@dataclass
class Suite:
    id: str
    variants: tuple  # allowed variant names
    params: dict  # name -> Param
    metrics: dict  # declared metric name -> description
    # build(params: dict, paths: RunPaths) -> (argv: list[str], env: dict)
    # the argv is the HARNESS command; the harness is versioned code in
    # this package and is the only thing the runner ever launches.
    build: Callable = None
    # parse(paths: RunPaths, params: dict) -> metrics dict (must reproduce
    # result.json metrics from the record alone)
    parse: Callable = None
    done: str = "result.json"  # the contract value; validated below
    timeout_s: int = 3600
    desc: str = ""

    def check(self):
        if not self.id:
            raise ParamError(f"suite {self.id!r}: missing id")
        if not self.variants:
            raise ParamError(f"suite {self.id!r}: no variants declared")
        if self.build is None:
            raise ParamError(f"suite {self.id!r}: missing command builder")
        if self.parse is None:
            raise ParamError(f"suite {self.id!r}: missing parser")
        if self.done != "result.json":
            raise ParamError(f"suite {self.id!r}: unsupported done condition")
        if not self.metrics:
            raise ParamError(f"suite {self.id!r}: no metrics declared")

    def validate(self, variant: str, overrides: dict) -> dict:
        if variant not in self.variants:
            raise ParamError(
                f"suite {self.id!r}: unknown variant {variant!r}; "
                f"allowed: {list(self.variants)}"
            )
        unknown = [k for k in overrides if k not in self.params]
        if unknown:
            raise ParamError(
                f"suite {self.id!r}: unknown param(s) {unknown}; "
                f"allowed: {sorted(self.params)}"
            )
        merged = {}
        for name, spec in self.params.items():
            if name in overrides:
                merged[name] = spec.coerce(name, overrides[name])
            elif spec.required:
                raise ParamError(
                    f"suite {self.id!r}: param {name!r} is required"
                )
            else:
                merged[name] = spec.default
        return merged


_SUITES: Optional[dict] = None


def registry() -> dict:
    """Load the suite registry once (id -> Suite); every suite self-checks."""
    global _SUITES
    if _SUITES is None:
        from .suites import ALL
        loaded = {}
        for suite in ALL:
            suite.check()
            if suite.id in loaded:
                raise ParamError(f"duplicate suite id {suite.id!r}")
            loaded[suite.id] = suite
        _SUITES = loaded
    return _SUITES


def get_suite(suite_id: str) -> Suite:
    reg = registry()
    if suite_id not in reg:
        raise ParamError(
            f"unknown suite {suite_id!r}; known: {sorted(reg)}"
        )
    return reg[suite_id]


_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")


def safe_run_component(value: str) -> str:
    if not _SAFE_NAME.fullmatch(str(value)):
        raise ParamError(f"unsafe name component {value!r}")
    return str(value)
