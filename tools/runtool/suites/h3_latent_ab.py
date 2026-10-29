"""h3_latent_ab suite: full same-seed trainings against variant cache sets
(SPEC--runtool-latent-quality.md Suite 2; owner verdict protocol).

Purpose: grade a latent-PIPELINE variant end to end -- real training runs
consuming the cache set that variant built, same seed, queued in REVERSE
(newest variant first via --after; the owner cancels the tail on a good
blind A/B listen/watch). Loss is recorded but is ONLY corroborating
evidence; the verdict is the owner's sample judgment (`judge` verb).

Builds are NOT this suite's job (one canonical mechanism per purpose):
a cache set is built by an h3_vae_fixtures mode=encode run on the
variant's own rt_ dataset copy -- cold reset, pinned draw stream, record
copies + cache_manifest.json. THIS suite takes `build_run` (that record)
and enforces the provenance laws before burning a single GPU minute:

  1. PROVENANCE: build_run must be a finished h3_vae_fixtures mode=encode
     record whose dataset key equals this run's dataset, whose arm matches
     the claimed latent_variant (fork_old<->old_path,
     fork_registry_stageA<->registry_path), and which actually produced
     cache artifacts (a refused-record has none).
  2. DRIFT: every cache file the record fingerprinted must still be there
     under the identity law (tensor bits + metadata values -- the raw file
     hash is process-stable at best, the safetensors __metadata__ key
     order is hashed per process). A drifted dataset trained bytes nobody
     built -- refused, counted, by name. Checked at SUBMIT (cheap refusal,
     no record at all) AND again at spawn (a --after chain can sit queued
     while someone re-encodes underneath it; that refusal IS recorded,
     marker latent_ab_refused).
  3. CACHE-READ law: variant trainings must READ caches, never re-encode
     (owner phased-loading law). Graded through the expect vocabulary:
     vae_skip present on a complete cache set is the machine's proof the
     VAE was never loaded; its absence is a named check failure.

Door status (declared now, forward arms fail LOUDLY, never silently
fall back -- incremental-support law):
  fork_old               real today (builds from old_path encode records)
  fork_registry_stageA   real once a registry_path build exists (the model
                         migration is the fork's; the build side already
                         refuses pre-migration)
  fork_stageB_fp32       refused: no fp32-regime build door in the tree
  paired_export          refused: consumes exported paired-repo cache
                         files; the exporter exists fork-side, the
                         consumption door is not built yet. Synthetic
                         verification built with the suite; the first
                         REAL exported file lands the same increment as
                         the real-export verification (no PASS claims on
                         synthetic bytes once real ones exist).

Samples: training with sample=true writes into the job dir (manifest
job_dir is in the record -- the blind A/B reads them from there); the
samples_written metric counts them from the record's own view.
"""

import hashlib
import json
import os

from ..registry import Param, ParamError, RunPaths
from ..trainer_kit import (METRICS, TrainerKit, load_map, parse_metrics)
from .h3_vae_fixtures_runner import _bits, _header_dict

_VARIANTS = ("fork_old", "fork_registry_stageA", "fork_stageB_fp32",
             "paired_export")
_BUILD_ARM = {"fork_old": "old_path", "fork_registry_stageA": "registry_path"}
_NO_DOOR = ("fork_stageB_fp32", "paired_export")

_AB_MARKERS = {
    # spawn-time provenance/drift refusal: the submit-time check passed and
    # the world changed underneath the queue. Presence = FAIL, named.
    "latent_ab_refused": r"latent_ab refused: (?P<detail>.+)",
}

_AB_METRICS = dict(METRICS)
_AB_METRICS["samples_written"] = (
    "sample files in the job dir at finish (blind A/B source; 0 with "
    "sample=true is a named finding for the expect, never a shrug)")


def _fp(path):
    """Canonical fingerprint under the identity law: sorted metadata
    values + every tensor's raw bits (key-sorted). Immune to the
    safetensors __metadata__ key-order shuffle, sensitive to anything a
    pipeline could mean (values, dtypes, shapes, tensor set)."""
    h = hashlib.sha256()
    head = _header_dict(path)
    h.update(json.dumps({k: v for k, v in sorted(head.items()) if
                         k != "__metadata__"}, sort_keys=True).encode())
    h.update(json.dumps(head.get("__metadata__", {}), sort_keys=True,
                        default=str).encode())
    from safetensors.torch import load_file
    state = load_file(path, device="cpu")
    for k in sorted(state):
        h.update(k.encode())
        h.update(_bits(state[k]).numpy().tobytes())
    return h.hexdigest()


def _check(params):
    """The provenance + drift laws as a list of named problems ([] = ok).
    Cheap: fingerprints ~40 small files, no GPU, no model."""
    from ..runner import RUNS_ROOT
    problems = []
    variant = params["latent_variant"]
    if variant in _NO_DOOR:
        return [f"variant {variant!r}: no build door exists in the tree "
                f"yet; declaring it early and refusing loudly is the "
                f"protocol (never a silent old_path fall-back)"]
    rid = params.get("build_run")
    if not rid or not os.path.exists(
            os.path.join(RUNS_ROOT, rid, "manifest.json")):
        return [f"build_run={rid!r}: no run record -- every ab run names "
                f"the exact encode record that built its cache set"]
    with open(os.path.join(RUNS_ROOT, rid, "manifest.json")) as f:
        bman = json.load(f)
    bp = bman.get("params") or {}
    if bman.get("suite") != "h3_vae_fixtures" or bp.get("mode") != "encode":
        problems.append(f"build_run {rid}: not an h3_vae_fixtures "
                        f"mode=encode record")
    if bp.get("dataset") != params["dataset"]:
        problems.append(f"build_run {rid} encoded dataset "
                        f"{bp.get('dataset')!r} but this run trains "
                        f"{params['dataset']!r}: the training would consume "
                        f"bytes the graded build never produced")
    want_arm = _BUILD_ARM.get(variant)
    if want_arm and bp.get("arm") != want_arm:
        problems.append(f"variant {variant!r} claims the {want_arm!r} "
                        f"pipeline but build_run {rid} ran arm "
                        f"{bp.get('arm')!r}")
    mpath = os.path.join(RUNS_ROOT, rid, "artifacts", "cache_manifest.json")
    if not os.path.exists(mpath):
        problems.append(f"build_run {rid} produced no cache_manifest.json "
                        f"(a refusal record? nothing was built)")
        return problems
    with open(mpath) as f:
        manifest = json.load(f)
    if not manifest:
        problems.append(f"build_run {rid}: empty manifest")
        return problems
    folders = load_map(params["dataset"])
    for key in sorted(manifest):
        i, name = key.split("/", 1)
        ds_i = int(i[2:])
        if ds_i >= len(folders):
            problems.append(f"{key}: dataset folder index {ds_i} not in "
                            f"{params['dataset']!r} anymore")
            continue
        live = os.path.join(folders[ds_i], "_latent_cache", name)
        rec = os.path.join(RUNS_ROOT, rid, "artifacts", "cache", key)
        if not os.path.exists(live):
            problems.append(f"{key}: missing from the live dataset cache "
                            f"(training would re-encode it -- the variant "
                            f"bytes would not be the built ones)")
            continue
        if not os.path.exists(rec):
            problems.append(f"{key}: record copy absent (broken build "
                            f"record)")
            continue
        if _fp(live) != _fp(rec):
            problems.append(f"{key}: DRIFTED since the build -- live cache "
                            f"no longer satisfies the identity law against "
                            f"the build record's copy")
    return problems


def _precheck_hook(params, ctx=None):
    # _check covers door status, build_run presence, provenance and drift
    # in one pass; a clean bill chains the kit's own guards (tool-dataset
    # law, template/param conflicts).
    problems = _check(params)
    if problems:
        raise ParamError("h3_latent_ab precheck refused:\n  - " +
                         "\n  - ".join(problems))
    _kit_base_precheck(params, ctx)


_kit = TrainerKit(
    namespace="H3AB",
    module="tools.runtool.suites.h3_latent_ab",
    extra_params={
        "latent_variant": Param(
            "enum", default="fork_old", choices=_VARIANTS,
            desc="which pipeline built the cache set this training "
                 "consumes; provenance is graded against build_run's "
                 "record, never against this label alone"),
        "build_run": Param(
            "str", default=None,
            desc="REQUIRED: the h3_vae_fixtures mode=encode record that "
                 "built this dataset's cache set (dataset key, arm and "
                 "file fingerprints are all checked against it)"),
    },
    extra_markers=_AB_MARKERS,
)


def _parse_ab(paths: RunPaths, params):
    out = parse_metrics(paths, params)
    n = None
    try:
        with open(paths.manifest) as f:
            job_dir = (json.load(f) or {}).get("job_dir")
        if job_dir and os.path.isdir(job_dir):
            n = sum(len([x for x in fl if not x.startswith(".")])
                    for _, _, fl in os.walk(os.path.join(job_dir, "samples")))
    except (OSError, ValueError):
        pass
    out["samples_written"] = n
    return out


def _spawn_ab(paths, params, manifest, cfg_path, job_dir):
    import subprocess
    import sys
    # the SECOND gate (the law may not rot underneath a queued chain):
    problems = _check(params)
    if problems:
        print("latent_ab refused: between submit and start, "
              f"{len(problems)} provenance/drift problem(s):", flush=True)
        for p in problems[:20]:
            print(f"  - {p}", flush=True)
        return 1
    print("latent_ab gate passed: build_run provenance + zero fingerprint "
          "drift", flush=True)
    return subprocess.call(
        [sys.executable, "run.py", cfg_path, "--log",
         os.path.join(paths.artifacts, "run.log")],
        cwd=manifest["cwd"],
    )


_kit_base_precheck = _kit.precheck
_kit.precheck = _precheck_hook
_kit.parse = _parse_ab
_kit.spawn_job = _spawn_ab

SUITE = _kit.suite(
    suite_id="h3_latent_ab",
    variants=("v1",),
    metrics=_AB_METRICS,
    timeout_s=4 * 3600,
    retention="judged",
    desc="same-seed trainings against variant-built cache sets (builds are "
         "h3_vae_fixtures encode runs; provenance + fingerprint-drift "
         "refused at submit AND at spawn); reverse-queued chains, owner "
         "verdict by blind sample A/B via judge",
)


if __name__ == "__main__":
    import sys

    sys.exit(_kit.main())
