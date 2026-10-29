"""h3_vae_fixtures suite: the VAE encode-quality measurement (owner regime,
SPEC--runtool-latent-quality.md).

Not a training run. mode=encode runs the REAL phase-1 cache pass
(build_datasets -> cache_latents_all_latents -> FileItemDTO/bucket
machinery) over a fixture dataset under one pipeline arm -- no TE, no
transformer (VAE-only door, verified real: model.load_vae + the trainer's
own caching entry). Every run starts cold LOUDLY (existing cache files are
deleted with a counted event: a warm start would silently grade the
previous arm's bytes -- the file-naming hash does NOT cover vae dtype or
precision, so two arms collide on one filename by design).

What lands in the record (the deliverable):
  artifacts/cache/ds<N>/<file>   the cache files this arm produced
  artifacts/cache_manifest.json  path/bytes/sha256/shape/dtype/num_frames/
                                 latent_space_version per cache file
  artifacts/concat/              ORIGINAL | ENCODE->DECODE side-by-side
                                 files for the owner's media-player watch
  artifacts/fixt.json            per-fixture rows (diffs, roundtrip dB)
mode=compare takes two encode records (run_a/run_b), pairs their cache
files and grades: Stage A byte-identity, Stage B declared-diff against
--diff_ceiling. It reads RECORDS ONLY (the recompute contract), and it
refuses the two runs when they share a dataset key -- same-folder arms
overwrite each other's cache files (identical hashes for same-semantics
encodes) and a compare of the directory would grade an overwrite.

Arms are the pipeline seam the vae_id registry opens, gated against the
model's MIGRATION STATE (declaring vae_id opts a model into registry
dispatch; H3 still overrides load_vae, so it is pre-migration):
old_path grades real today, and REFUSES once the model migrates (the
label could not name a path that no longer runs -- the pre-migration
baseline lives in the seeded pre-migration records compare reads);
registry_path FAILS LOUDLY until the model migrates (never a silent fall
back to old_path). stage=A grades cache identity (structure-only moves;
identity law = tensor bits + meta values, see SPEC); stage=B grades the
quality regime (diffs recorded, explained within ceiling, dB, the
owner's watch).
stranded_refused is declared as vocabulary for the fork-side meta
tripwire (bumped-version caches refused at open) -- the print is the
refactor session's half of the increment.
"""

import json
import os

from ..registry import Param, ParamError, RunPaths
from ..trainer_kit import TrainerKit, tag_metrics

_ARMS = ("old_path", "registry_path")  # arm truth is gated against the
# model's vae_id migration state at harness start (H3 migrated c578d205:
# registry_path real, old_path refuses)
_MODES = ("encode", "compare")

_FIXT_MARKERS = {
    # printed only when EVERY paired cache file is identical in the
    # identity law that is attainable: tensor bits equal + metadata
    # values equal. Raw file hashes are NOT identity -- the safetensors
    # __metadata__ key order is process-hashed (measured). Any mismatch
    # prints fixt_unexplained_diff rows instead
    "fixt_bytes_identical": r"fixt bytes identical: (?P<identical>\d+)/(?P<total>\d+)",
    # arithmetic from the manifest's own numbers (17n+5 -> 5n+2 latent T;
    # stills T=1; packed audio rows even and 32 wide). NOTE: warm-read
    # fidelity is deliberately NOT a suite marker -- inside the encoding
    # process it grades that process's own state; reader-vs-cache is
    # graded in real training under the vae_skip law, from a fresh process.
    "fixt_frame_parity": r"fixt frame parity ok: (?P<items>\d+)/(?P<total>\d+)",
    # stage B: every paired diff nonzero-but-below-ceiling
    "fixt_diff_declared": r"fixt diff declared: max (?P<max>\S+) <= ceiling (?P<ceiling>\S+)",
    # presence = FAIL, in any stage
    "fixt_unexplained_diff": r"fixt unexplained diff: (?P<detail>.+)",
    "concat_written": r"fixt concat written: (?P<count>\d+)",
    # the encode pass drew from the pinned stream; its bytes reproduce
    # iff the pipeline (not the RNG) is unchanged. Presence is the law.
    "fixt_rng_pinned": r"fixt rng pinned: seed=(?P<seed>\d+)",
    # forward vocabulary: fork-side tripwire (print lands with Stage-B)
    "stranded_refused": r"stranded cache refused",
    "latent_ver": r"latent_space_version=(?P<tag>\S+)",
}

_FIXT_METRICS = {
    "fixt_items": "fixture items graded by this record",
    "fixt_diff_max": "compare: max |a-b| over paired cache tensors",
    "fixt_diff_mean": "compare: mean |a-b| over paired cache tensors",
    "fixt_audio_row_diff": "compare: max |a-b| over paired packed audio rows",
    "roundtrip_db_video": "encode: min roundtrip dB (float, pre-codec) across video fixtures",
    "roundtrip_db_audio": "encode: min roundtrip SNR dB across audio-bearing fixtures",
    "rss_peak": "peak host RSS MB for the process group (kit mem watcher)",
}


def _rss_peak(paths: RunPaths):
    peak = None
    try:
        with open(paths.events) as f:
            for line in f:
                if '"mem_summary"' in line:
                    peak = json.loads(line).get("rss_peak", peak)
    except FileNotFoundError:
        pass
    return peak


@tag_metrics("artifacts/fixt.json | artifacts/compare.json + events mem_summary")
def _parse(paths: RunPaths, params):
    out = {k: None for k in _FIXT_METRICS}
    src = "compare.json" if params["mode"] == "compare" else "fixt.json"
    try:
        with open(os.path.join(paths.artifacts, src)) as f:
            data = json.load(f)
    except FileNotFoundError:
        out["rss_peak"] = _rss_peak(paths)
        return out
    for k in ("fixt_items", "fixt_diff_max", "fixt_diff_mean",
              "fixt_audio_row_diff", "roundtrip_db_video",
              "roundtrip_db_audio"):
        if data.get(k) is not None:
            out[k] = data[k]
    out["rss_peak"] = _rss_peak(paths)
    return out


def _precheck(params, ctx=None):
    if params["mode"] != "compare":
        return
    from ..runner import RUNS_ROOT
    problems = []
    for key in ("run_a", "run_b"):
        rid = params.get(key)
        if not rid or not os.path.exists(
                os.path.join(RUNS_ROOT, rid, "result.json")):
            problems.append(f"{key}={rid!r}: no finished h3_vae_fixtures "
                            f"record at that run id")
    if problems:
        raise ParamError("compare precheck refused:\n  - " + "\n  - ".join(problems))
    if params["stage"] == "B" and params.get("diff_ceiling") is None:
        raise ParamError(
            "stage B compare requires diff_ceiling: the declared-diff "
            "verdict needs the owner-set ceiling (tiny-but-nonzero is the "
            "bug dying; anything else is unexplained)")
    ds = {}
    for key in ("run_a", "run_b"):
        with open(os.path.join(RUNS_ROOT, params[key], "manifest.json")) as f:
            man = json.load(f)
        if man.get("suite") != "h3_vae_fixtures":
            problems.append(f"{key}: suite is {man.get('suite')!r}, not "
                            f"h3_vae_fixtures -- cannot compare across suites")
        if (man.get("params") or {}).get("mode") != "encode":
            problems.append(f"{key}: not a mode=encode record")
        ds[key] = (man.get("params") or {}).get("dataset")
    if problems:
        raise ParamError("compare precheck refused:\n  - " + "\n  - ".join(problems))
    if ds["run_a"] == ds["run_b"]:
        raise ParamError(
            f"compare refused: run_a and run_b both encode dataset key "
            f"{ds['run_a']!r}. Two arms on one dataset folder share the "
            f"same _latent_cache paths (the cache filename hash does not "
            f"cover vae precision), so the later run overwrote the "
            f"earlier one's bytes in-place; a compare here would grade an "
            f"overwrite. Use separate rt_ fixture copies (datasets "
            f"--create K_arm --from-key SRC) so each arm owns its folder.")


def _spawn(paths, params, manifest, cfg_path, job_dir):
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    argv = [sys.executable, os.path.join(here, "h3_vae_fixtures_runner.py"),
            "--config", cfg_path, "--mode", params["mode"],
            "--stage", params["stage"], "--artifacts", paths.artifacts]
    if params["mode"] == "encode":
        argv += ["--arm", params["arm"]]
    else:
        argv += ["--run-a", params["run_a"], "--run-b", params["run_b"]]
        if params.get("diff_ceiling") is not None:
            argv += ["--ceiling", str(params["diff_ceiling"])]
    return subprocess.call(argv, cwd=manifest["cwd"])


_kit = TrainerKit(
    namespace="H3VAE",
    module="tools.runtool.suites.h3_vae_fixtures",
    extra_params={
        "mode": Param("enum", default="encode", choices=_MODES,
                      desc="encode: run the real phase-1 cache pass under "
                           "one arm; compare: pair two encode records' "
                           "cache artifacts and grade (run_a/run_b)"),
        "arm": Param("enum", default="registry_path", choices=_ARMS,
                     desc="encode pipeline arm, gated against migration: "
                          "registry_path (real since the model declares "
                          "vae_id, c578d205) and old_path (refuses once "
                          "migrated -- the label could not name a path "
                          "that no longer runs; pre-migration baselines "
                          "are era-superseded per owner ruling 2026-10-29 "
                          "anyway). The runner refuses mismatches loudly; "
                          "nothing is ever substituted. Arm exported as "
                          "RUNTOOL_VAE_ARM for registry-side consumers."),
        "stage": Param("enum", default="A", choices=("A", "B"),
                       desc="A: structure-only moves, byte-identity "
                            "required. B: quality regime -- diffs are "
                            "EXPECTED NONZERO; the verdict is declared-"
                            "within-ceiling + roundtrip dB + the owner's "
                            "concat watch."),
        "run_a": Param("str", default=None, desc="compare: first encode record"),
        "run_b": Param("str", default=None, desc="compare: second encode record"),
        "diff_ceiling": Param(
            "float", default=None, min=0.0, max=10.0,
            desc="stage B compare: max abs per-element diff below which a "
                 "change is DECLARED (the bugs dying: removed casts, "
                 "precision collapse). Owner-set quality contract; "
                 "REQUIRED for stage B compare"),
    },
    extra_markers=_FIXT_MARKERS,
    build_env=lambda params, paths: {
        # inert until the vae_id registry reads it (forward seam; the arm
        # is authoritative in the manifest regardless)
        "RUNTOOL_VAE_ARM": params.get("arm") or "",
    },
    extra_flags=lambda params: [
        params["arm"] if params["mode"] == "encode" else "cmp",
        "s" + params["stage"],
    ],
    parse=_parse,
    spawn_job=_spawn,
)

_kit_precheck = _kit.precheck


def _precheck_hook(params, ctx=None):
    _kit_precheck(params, ctx)
    _precheck(params, ctx)


_kit.precheck = _precheck_hook

SUITE = _kit.suite(
    suite_id="h3_vae_fixtures",
    variants=("v1",),
    metrics=_FIXT_METRICS,
    timeout_s=30 * 60,
    retention="judged",
    desc="H3 VAE fixture encode/compare: real phase-1 cache pass per arm, "
         "cache-file pairs + original|decode concats into the record, "
         "record-only comparison verdicts (stage A byte-identity, stage B "
         "declared-diff + dB + owner watch)",
)


if __name__ == "__main__":
    import sys

    sys.exit(_kit.main())
