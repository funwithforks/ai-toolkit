"""JSON CLI for runtool. Every subcommand prints one JSON object.

  suites                      registry (params, bounds, metrics)
  submit --suite S --variant V [--params JSON] [--note ...]
  wait   RUN_ID [--timeout-s N]
  status RUN_ID
  cancel RUN_ID
  list   [--limit N]
  recompute RUN_ID
"""

import argparse
import json
import sys

from . import runner
from .registry import ParamError, registry


def main(argv=None):
    ap = argparse.ArgumentParser(prog="runtool")
    sub = ap.add_subparsers(dest="op", required=True)

    sub.add_parser("suites")

    p = sub.add_parser("submit")
    p.add_argument("--suite", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--params", default="{}", help="JSON object of params")
    p.add_argument("--note", default="")

    for name in ("wait", "status", "cancel", "recompute"):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        if name == "wait":
            p.add_argument("--timeout-s", type=float, default=None)

    p = sub.add_parser("list")
    p.add_argument("--limit", type=int, default=15)

    a = ap.parse_args(argv)
    try:
        if a.op == "suites":
            reg = registry()
            out = {
                sid: {
                    "desc": s.desc,
                    "variants": list(s.variants),
                    "timeout_s": s.timeout_s,
                    "metrics": s.metrics,
                    "params": {
                        k: {
                            "kind": v.kind,
                            "default": v.default,
                            "required": v.required,
                            "min": v.min,
                            "max": v.max,
                            "choices": v.choices,
                            "desc": v.desc,
                        }
                        for k, v in s.params.items()
                    },
                }
                for sid, s in reg.items()
            }
            print(json.dumps(out, indent=1))
            return 0
        if a.op == "submit":
            params = json.loads(a.params)
            if not isinstance(params, dict):
                raise ParamError("--params must be a JSON object")
            res = runner.submit(a.suite, a.variant, params, note=a.note)
        elif a.op == "wait":
            res = runner.wait(a.run_id, timeout_s=a.timeout_s)
        elif a.op == "status":
            res = runner.classify(a.run_id)
        elif a.op == "cancel":
            res = runner.cancel(a.run_id)
        elif a.op == "recompute":
            res = runner.recompute(a.run_id)
        elif a.op == "list":
            res = {"runs": runner.list_runs(a.limit)}
        print(json.dumps(res, indent=1, default=str))
        status = res.get("status") if isinstance(res, dict) else None
        return 0 if status not in ("crash", "failed", "timeout") else 1
    except ParamError as e:
        print(json.dumps({"error": str(e), "type": "rejected"}))
        return 2
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e), "type": "not_found"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
