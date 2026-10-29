# Phased Loading — Loader Conventions (4/4)

Part of the `phased-loading--` guide packet:
[component phases](./phased-loading--component-phases.md) ·
[embed caches](./phased-loading--embed-caches.md) ·
[staged sampling](./phased-loading--staged-sampling.md)

Anchors are as of commit `22f1c08` (2026-10-07).

Every model's `_load_vae` / `_load_text_encoder` / `_load_transformer`
(`toolkit/models/phased_load.py:1-31`) ends at the same place: a weight file
found through the shared resolver, read by a WeightSource, mapped onto a meta
shell. This document is the contract for that last mile. Read it before
writing a new loader or touching shared loading code — the rules here are the
ones `AGENTS.md` ("Loading rules", "Iron rules") enforces, with the code
anchors that show how. Vae loading for new/migrated models goes through the
`vae_id` registry — §5.

## 1. Resolution: never hardcode a path or a capacity

**Where `MODELS_PATH` itself comes from** (settled; do not re-investigate): the
UI settings page ("Models Folder Path", field `MODELS_PATH`,
`ui/src/app/settings/page.tsx:113`) saves it to the sqlite `Settings` table in
`aitk_db.db`; the cron worker injects it into the spawned `run.py` env
(`ui/cron/actions/startJob.ts:263-268`, an already-set env always wins), and
`run.py` `load_dotenv()` + `toolkit/paths.py:11-14` make it effective.
Precedence: `.env`/env > DB row > `<repo>/models` default. On this machine the
effective root is the value in repo `.env` (gitignored) — resolve it at runtime
with `from toolkit.paths import MODELS_PATH`, never from memory, and never from
`<repo>/models` unless that is what the env says. Weights land in its ComfyUI
subfolders (`diffusion_models/`, `text_encoders/`, `vae/`), created on demand
by the resolver. `COMFY_MODELS_PATH` (`paths.py:8`) is dead — no consumers.

Component files resolve **only** through the existing chain. Two entry points,
both in `toolkit/models/v2/resolver.py`:

- `resolve_comfy_candidates(candidates, repo_id, qtype=…, extra_roots=…)`
  (`:66`) — pick one among precision variants of a component. Local-first,
  ranked by `comfy_precision_rank` (`:19`) for the requested `qtype`, then
  list order; **download to `MODELS_PATH` only when nothing local exists**,
  into the comfy-layout subfolder. `extra_roots` lets a local checkpoint dir
  participate; downloads still land in `MODELS_PATH` regardless. The
  `split_files/` / `non_official/` repo prefixes are stripped for the local
  layout (`:59`).
- `resolve_named_file` / `find_file_recursive` (`:119`) — an explicit file
  reference (local path, `org/repo/filename`, or bare name searched under
  `MODELS_PATH` and `extra_roots`).

The `WeightSource.resolve` docstring (`toolkit/util/weight_source.py:29`)
states the toolkit-wide download policy in one line: "settings value if set,
system cache only when it is not" — weight fetches land in
`MODELS_PATH/<category>/`, never the ambient HF cache.

**Registration, not paths, is how a model declares its files.** Class
attributes on the v2 model (`toolkit/models/v2/_mixin.py`):
`aitk_subfolder` (`:109`), `aitk_comfy_repo` (`:116`),
`aitk_comfy_weight_names` (`:122`), default `aitk_qtype` (`:145`). A model
registers candidate files; the holder resolves them. Do not bake a repo id,
a folder layout, or a machine's RAM/VRAM into a loader. LTX-2.5 registers only
the shipped int8 ConvRot files
(`toolkit/models/v2/diffusion_models/ltx2.py`) so a `quantize: none` run cannot
silently rank the ~44GB bf16 twin first — the bf16 files stay reachable only
via an explicit `<component>_path` override.

Model-side entry (holder): `_resolve_comfy_file` (ltx2.py:1258) calls the
resolver with the `model_kwargs` override as `override_path` and the local
checkpoint dir as an `extra_root`; a miss raises with the fix spelled out
(`:1277`). `_resolve_dit_path` / `_resolve_te_path` (ltx2.py:1292) layer the
version-specific candidate lists on top.

## 2. Reading: WeightSource is the only reader

`toolkit/util/weight_source.py:25`. Contract points:

- `open(path)` (`:69`) is lazy — validates, reads nothing.
- `get(key, device)` (`:90`) faults **one** tensor onto `device`; nothing
  cpu-sized is allocated for it. This is what lets a 22GB file load onto a
  32GB card with no peak-RSS spike.
- `materialize(device)` for a GPU target is a **trap**: safetensors 0.9.x's
  device path stages every tensor through host RAM (docstring `:96`+), piling
  up model-sized peak rss. Use per-key `get()` for GPU placement;
  `materialize('cpu')` is the exact old `load_file()` behavior.

The whole-file `load_file()` batched state dict is now the **exception path**,
kept only where the requantizer genuinely needs the whole cpu dict —
low_vram parking or a user-requested requantization
(`_as_shipped_stream_target`, ltx2.py:1570: returns the stream device, or
`None` to select the batched cpu path).

## 3. Placement: meta shell + per-key plan (as-shipped path)

The streaming loader (reference: `_stream_load`, ltx2.py:1609):

1. Build the module under `init_empty_weights()` (a meta shell — correct
   shapes/dtypes, no storage).
2. Build a plan `{module_key: (WeightSource, file_key)}` by mapping each file
   key through the **same rename tables** the batched converter uses
   (`_remap_file_key`, ltx2.py:1593). The converters only rename keys — none
   reads a tensor value — so the tables drive a pure-key plan and no state
   dict is ever materialized. A key the converter's special-handler drops
   stays dropped (`_remap_file_key` returns `None`).
3. `get()` each tensor onto the target device and `load_state_dict(assign=True,
   strict=False)` it one at a time. Nothing is copied or recast in the process.
4. The final `meta`-parameter scan is the `strict=True` stand-in: any module
   key left on meta means the plan missed a tensor — fail loudly, don't ship a
   silently-half-loaded model.

**Quantize `none`/`false` means zero quantization code runs**: shipped dtypes
kept, tensors assigned as-is. There is no full-precision mode to fall into.

Pre-quantized comfy files (convrot/fp8) are the one part of the plan that is
not a plain assign. `_stream_load` groups each `*.comfy_quant` marker and hands
the group to `import_comfy_quantized_layers`
(`toolkit/util/comfy_quant_import.py:215`) module-at-a-time; the module's
unquantized siblings (bias, tables) are assigned alongside, and
`_mixed_file_post_load` (ltx2.py:1658) re-enables per-op input casting for the
mixed-precision result. The `_load_quantized_module` batched equivalent is
ltx2.py:1307.

## 4. New-loader rules (the acceptance bar)

These are `AGENTS.md` rules; the failure mode that earned each is in
parentheses — a loader bug is a device-placement or dtype bug, which read-all
in the file never catches:

1. **Equivalence-test against the old path on real files**: bitwise
   `state_dict` comparison, RSS delta during load, and a real forward pass.
   Read-through does not catch device-placement bugs.
2. **Smoke tests must exercise the real loaders.** A test that bypasses the
   loader to reach the feature is testing nothing.
3. **Shared loading code changes are opt-in**: class-level default off, every
   other model byte-for-byte unchanged, per-model rollout only after that
   model is verified live.
4. **Any new gated consumer declares itself in its gate and prints why it
   forced a load** — silent forced loads are regressions-in-waiting. See
   [component phases](./phased-loading--component-phases.md).
5. **A free goes through the memory machinery**, not `self.thing = None`:
   `MemoryManager.free` / device-state restore, or `del` + `flush()` at a
   phase boundary. `self.thing = None` frees nothing if a reference survives
   (e.g. a pipeline still holding it — see the `self.pipeline` rewiring in
   `_ensure_connectors`/`release_encode_phase_components`, ltx2.py:1775/:1795).

## 5. The vae_id registry (component-keyed vae resolution)

`toolkit/models/vae_registry.py` + the dispatch in
`PhasedLoadMixin.load_vae` (`toolkit/models/phased_load.py`). A model opts
in by declaring the class attribute `vae_id`; the registry maps that id to
the entry that owns the vae — construction, placement policy, and the
cache-tag pin — instead of the model's own `load_vae`/`_load_vae` body.
Models that declare no id are untouched (the only sense in which this is
opt-in: a model that declares one gets NO vae loading of its own left).

**Entry shape** (`VaeEntry`, frozen dataclass): `vae_id`; `components`
(role → component class — the class carrying `aitk_comfy_*` metadata, never
a path); `load(model)` (resolves candidates through `_resolve_comfy_file`,
streams to `model.vae_device_torch`, returns the module or bundle);
`dtype_policy` (`keep_shipped` = placement only moves, shipped precisions
survive verbatim; `as_configured` = placement also casts to
`model_config.vae_dtype`); `bundle` (multiple sub-vae members behind one
`self.vae` handle — forces `keep_shipped`, members may be None under
staged sampling); `latent_space_version` (the explicit cache tag — applied
at construction, not load time, because a `phases` run that never loads
the vae still hashes the tag into cache filenames); `encode_images` /
`decode_latents` (reserved for the encode migration — until an id fills
them, the model's own encode/decode methods are the call surface).

**Onboarding a model** (the H3 migration, commit `c578d205`, is the
reference):

1. Write the id module next to the model class (e.g.
   `minimax_h3_vae_id.py`): import the component classes + bundle, define
   `load`, `register_vae_entry(VaeEntry(...))`. Import it from the
   extension's `__init__.py`.
2. Declare `vae_id` on the model class; remove its `load_vae` override /
   `_load_vae` body, any vae entries in a model-level `COMFY_FILES`-style
   table, and any in-body `latent_space_version` assignment (the pin
   comes from the entry).
3. Duplicate ids, unknown ids, and cast policies on bundles all raise at
   registration/construction — do not soften these into warnings.

**Migration is replacement.** An agreed migration deletes the path it
supersedes in the same commit — no dual-path flags, no try-registry-except
-fallback, no feature toggles kept "for safety". The opt-in in rule 3 of §4
is about the *shared dispatch* staying inert for models that never declare
an id; it is never a reason to keep a migrated model's old loader alive. If
the old path must stay reachable, the migration is not agreed — ask.

**Byte identity is a guard, not the goal.** Diffing cache bytes old-path vs
new-path proves a *structure-only* move (the Stage A gate for the H3
migration: 40/40 files, 80/80 tensors, 40/40 meta dicts). It must never
veto a semantic change: any deliberate dtype/precision change is a
Stage-B change, and it breaks bytes ON PURPOSE with an explicit
`latent_space_version` bump. A bumped tag is stranded-by-construction —
the tag is hashed into cache filenames
(`dataloader_mixins.py` `get_latent_info_dict`), so old caches are never
opened, never "migrated", and simply sit unread. Quality is the goal;
identity only proves the move itself changed nothing.

## 6. Where the pieces live

| Concern | Anchor |
|---|---|
| Candidate ranking + download policy | `resolver.py:19` `comfy_precision_rank`, `:66` `resolve_comfy_candidates` |
| Model file registration | `_mixin.py:109/116/122/145` |
| vae_id registry + entry shape | `toolkit/models/vae_registry.py` |
| Registry dispatch + construction-time validation/pin | `phased_load.py` `PhasedLoadMixin.__init__` / `load_vae` |
| Reference vae_id migration (bundle, keep_shipped, tag pin, kill-list) | `minimax_h3/minimax_h3_vae_id.py`, commit `c578d205` |
| Lifecycle phase selection (`phases`) | `config_modules.py` `TrainConfig.phases`; guards/exits in `BaseSDTrainProcess.run()` |
| Lazy reader, per-key placement | `weight_source.py:25` `:90` |
| Stream-to-meta-shell reference | `ltx2.py:1609` `_stream_load`, `:1593` `_remap_file_key` |
| Batched path only when required | `ltx2.py:1570` `_as_shipped_stream_target` |
| Pre-quantized attach | `comfy_quant_import.py:215`, `ltx2.py:1658` |
| Variant filtering hook | `phased_load.py:36` `select_comfy_candidates` |
| Phased entry points (process-driven) | `phased_load.py:74/95/112`, `load_model` `:121` |
