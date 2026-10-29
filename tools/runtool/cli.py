"""JSON CLI for runtool. Every subcommand prints one JSON object.

  suites                      registry (params, bounds, metrics)
  config-schema [--section S] [--key K] [--class C]
                              config vocabulary derived from the real
                              config code (summary | section | one key)
  datasets [--list | --add K --folder P... | --create K
            (--synthetic N | --from-key K2 | --from-path P) |
            --rm K [--delete-files]]
                              machine-local dataset registry; file
                              deletion only for tool-created rt_ folders
  submit --suite S --variant V [--params JSON] [--expect JSON] [--note ...]
                              --expect declares the run's intent; the
                              verdict grades it at wait (markers/metrics)
  wait   RUN_ID [--timeout-s N]
                              returns status + metrics + markers + phases
                              + expect verdict (when one was declared)
  status RUN_ID
  inspect RUN_ID [--tail N] [--grep RX] [--context C] [--show-config]
                              diagnostic digest for strange runs: ugly-line
                              signals, stall seconds, line spam, raw tails
                              (0 = whole file), caller regex over the record
  cancel RUN_ID
  list   [--limit N]
  search [--suite S] [--status ST] [--dataset K] [--tag T] [--note TEXT]
         [--metric NAME<OP>NUM]... [--marker M]... [--no-marker M]...
         [--param K=V]... [--expect PASS|FAIL] [--since DAYS] [--variant V]
         [--jobs] [--limit N]
                              query the history over record JSON (fast,
                              read-only -- never starts queued runs);
                              --jobs lists the job dirs that exist with
                              owner run, size, age
  recompute RUN_ID
  gc [--days N] [--apply] [--include-judged]
                              reap the jobs subtree (finished tool runs,
                              orphans, stale embed caches); listing is
                              always safe; --apply deletes inside jobs
                              only (runtool records are never touched)
  complain --category C --text T [--run-id R] [--cost TEXT] [--agent N]
                              file a complaint about the tool itself;
                              cheap on purpose -- file before working
                              around a tool gap
  complaints [--stats]        read the channel; --stats aggregates by
                              category x suite (the tool-session agenda)
"""

import argparse
import json
import sys

from . import datasets, inspect, runner, schema
from .registry import ParamError, RunPaths, get_suite, registry


def main(argv=None):
    ap = argparse.ArgumentParser(prog="runtool")
    sub = ap.add_subparsers(dest="op", required=True)

    sub.add_parser("suites")

    p = sub.add_parser("config-schema")
    p.add_argument("--section", default=None,
                   help="config section, e.g. train, model, datasets[]")
    p.add_argument("--key", default=None, help="one key, within --section")
    p.add_argument("--class", dest="cls", default=None,
                   help="config class name (overrides --section)")

    p = sub.add_parser("datasets")
    p.add_argument("--list", action="store_true")
    p.add_argument("--show-paths", action="store_true",
                   help="include folder paths (owner-side; kept out of "
                        "default agent-facing output)")
    p.add_argument("--add", default=None, metavar="KEY",
                   help="register existing folders under a key")
    p.add_argument("--folder", action="append", default=None,
                   help="folder for --add (repeatable)")
    p.add_argument("--tool-created", action="store_true",
                   help="mark an --add entry as tool-owned (deletable)")
    p.add_argument("--create", default=None, metavar="KEY",
                   help="create a tool dataset (rt_-prefixed folder)")
    p.add_argument("--from-key", default=None)
    p.add_argument("--from-path", default=None)
    p.add_argument("--synthetic", type=int, default=0,
                   help="N synthetic solid-colour images+captions")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--copy-cache", action="store_true",
                   help="with a copy source, also carry TE/latent caches")
    p.add_argument("--rm", default=None, metavar="KEY")
    p.add_argument("--delete-files", action="store_true")

    p = sub.add_parser("submit")
    p.add_argument("--suite", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--params", default="{}", help="JSON object of params")
    p.add_argument("--note", default="")
    p.add_argument("--expect", default=None, help=(
        'JSON intent, graded at wait: {"markers":{"te_skip":"present",'
        '"plan":{"planned":"19"}},"metrics":{"loss_final":{"max":1.5}}}'
    ))
    p.add_argument("--config-patch", default=None, help=(
        "JSON object of dotted config paths to override (suite allowlist "
        'applies), e.g. {"config.process[0].train.lr": 0.0001}'
    ))
    p.add_argument("--after", default=None, help=(
        "queue behind this run_id (serial comparison chains, max 3 live); "
        "plain submits are refused while any jobs-subtree run is live"
    ))

    for name in ("wait", "status", "cancel", "recompute"):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        if name == "wait":
            p.add_argument("--timeout-s", type=float, default=None)

    p = sub.add_parser("inspect")
    p.add_argument("run_id")
    p.add_argument("--tail", type=int, default=30,
                   help="tail lines from each log; 0 or negative = whole "
                        "file (uncapped)")
    p.add_argument("--grep", default=None, metavar="REGEX",
                   help="caller regex over full stdout+stderr, line-numbered")
    p.add_argument("--context", type=int, default=0,
                   help="surrounding lines per --grep hit")
    p.add_argument("--show-config", action="store_true",
                   help="inline the derived config (registered real dataset "
                        "paths mechanically replaced by their keys)")

    p = sub.add_parser("list")
    p.add_argument("--limit", type=int, default=15)

    p = sub.add_parser("search", help="query the run history (read-only)")
    p.add_argument("--suite", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--status", default=None,
                   help="done | failed | crash | cancelled | running | "
                        "queued")
    p.add_argument("--dataset", default=None, help="dataset key")
    p.add_argument("--tag", default=None, help="substring of the tag param")
    p.add_argument("--note", default=None, help="substring of the note")
    p.add_argument("--metric", action="append", metavar="NAME<OP>NUM",
                   help='repeatable, e.g. "loss_final<0.31" '
                        '"steps_done>=18" (ops: < <= == > >=)')
    p.add_argument("--marker", action="append",
                   help="repeatable: marker must be FOUND (persisted)")
    p.add_argument("--no-marker", action="append",
                   help="repeatable: marker must be ABSENT")
    p.add_argument("--param", action="append", metavar="K=V",
                   help="repeatable exact param match, e.g. "
                        "cutlass_bwd=off (switches are params)")
    p.add_argument("--expect", choices=("PASS", "FAIL"),
                   help="persisted expect verdict")
    p.add_argument("--since", type=float, default=None, metavar="DAYS",
                   help="only records newer than N days")
    p.add_argument("--jobs", action="store_true",
                   help="instead of records: the job dirs that exist "
                        "(owner, run status, size, age)")
    p.add_argument("--limit", type=int, default=25)

    p = sub.add_parser("gc")
    p.add_argument("--days", type=float, default=7.0)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--include-judged", action="store_true")

    p = sub.add_parser("complain", help="file a complaint about the tool")
    p.add_argument("--run-id", default=None)
    p.add_argument("--category", required=True,
                   help="api-confusion | misleading-output | "
                        "missing-measurement | ergonomics | doc-gap | other")
    p.add_argument("--text", required=True)
    p.add_argument("--cost", default=None,
                   help="what the workaround cost (time, wrong conclusion)")
    p.add_argument("--agent", default=None, help="who filed it")

    p = sub.add_parser("complaints", help="read the complaint channel")
    p.add_argument("--stats", action="store_true",
                   help="aggregate by category x suite (the tool agenda)")
    p.add_argument("--limit", type=int, default=20)

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
                    "markers": s.markers,
                    "patchable": list(s.patchable),
                    "retention": s.retention,
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
        if a.op == "datasets":
            if a.rm:
                res = datasets.rm(a.rm, delete_files=a.delete_files)
            elif a.create:
                res = datasets.create(
                    a.create, from_key=a.from_key, from_path=a.from_path,
                    synthetic=a.synthetic, size=a.size,
                    copy_cache=a.copy_cache)
            elif a.add:
                if not a.folder:
                    raise ParamError("--add needs at least one --folder")
                res = datasets.add(a.add, a.folder,
                                   tool_created=a.tool_created)
            else:
                res = {"registry": datasets.REGISTRY_PATH,
                       "datasets": datasets.list_entries(
                           show_paths=a.show_paths)}
        elif a.op == "config-schema":
            res = schema.query(section=a.section, key=a.key, cls=a.cls)
            if isinstance(res, dict) and "error" in res:
                print(json.dumps({**res, "type": "rejected"}, indent=1))
                return 2
        elif a.op == "submit":
            params = json.loads(a.params)
            if not isinstance(params, dict):
                raise ParamError("--params must be a JSON object")
            expect = None
            if a.expect is not None:
                expect = json.loads(a.expect)
            patch = None
            if a.config_patch is not None:
                patch = json.loads(a.config_patch)
                if not isinstance(patch, dict):
                    raise ParamError("--config-patch must be a JSON object")
            res = runner.submit(a.suite, a.variant, params, note=a.note,
                                expect=expect, config_patch=patch,
                                after=a.after)
        elif a.op == "wait":
            res = runner.wait(a.run_id, timeout_s=a.timeout_s)
        elif a.op == "status":
            res = runner.classify(a.run_id)
        elif a.op == "inspect":
            res = runner.classify(a.run_id)
            try:
                suite = get_suite(res["suite"])
            except ParamError:
                suite = None
            paths = RunPaths(run_id=res["run_id"], root=res["record"])
            res.update(inspect.build(
                paths, suite, res["alive"], tail_lines=a.tail,
                grep_pattern=a.grep, grep_context=a.context,
                show_config=a.show_config))
        elif a.op == "cancel":
            res = runner.cancel(a.run_id)
        elif a.op == "gc":
            from . import gc as gc_mod
            if a.apply:
                res = gc_mod.apply_removal(days=a.days,
                                           include_judged=a.include_judged)
            else:
                res = gc_mod.candidates(days=a.days)
        elif a.op == "complain":
            from . import complain as complain_mod
            res = complain_mod.file_complaint(
                a.text, a.category, run_id=a.run_id, cost=a.cost,
                agent=a.agent)
        elif a.op == "complaints":
            from . import complain as complain_mod
            res = complain_mod.complaints(stats=a.stats, limit=a.limit)
        elif a.op == "recompute":
            res = runner.recompute(a.run_id)
        elif a.op == "list":
            res = {"runs": runner.list_runs(a.limit)}
        elif a.op == "search":
            from . import search as search_mod
            if a.jobs:
                res = search_mod.jobs()
            else:
                res = search_mod.search(
                    suite=a.suite, variant=a.variant, status=a.status,
                    dataset=a.dataset, tag=a.tag, note=a.note,
                    metric=a.metric, marker=a.marker,
                    no_marker=a.no_marker, param=a.param, expect=a.expect,
                    since=a.since, limit=a.limit)
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
