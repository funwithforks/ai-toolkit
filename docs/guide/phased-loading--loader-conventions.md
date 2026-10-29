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
the entry that owns the vae — file resolution, load, and behavior — instead
of the model's own `load_vae`/`_load_vae` body. Models that declare no id
are untouched (the only sense in which this is
opt-in: a model that declares one gets NO vae loading of its own left).

**Entry shape** (`VaeEntry`, frozen dataclass): `vae_id`; `components`
(role → component class — the class carrying `aitk_comfy_*` metadata, never
a path); `load(model)` (resolves candidates through `_resolve_comfy_file`,
streams to `model.vae_device_torch`, returns the module or bundle);
`bundle` (multiple sub-vae members behind one `self.vae` handle — members
may be None under staged sampling); `encode_images` / `decode_latents` (the
vae's behavior: when filled, `PhasedLoadMixin.encode_images`/`decode_latents`
run them — their output is exactly what the latent cache stores; until
filled, the model's own methods are the call surface).

**Precision is not a policy.** A vae is loaded at the precision it ships in.
The entry's loader takes no `dtype` argument, and `load_vae`'s placement step
for registry models is `vae.to(device)` with no cast — so a downcast is not
something an id opts out of, it is not expressible. Upstream's blanket "cast
the vae to the job's dtype" (and the `aitk_cast_on_load` /
`convert_state_dict_on_load` machinery behind it) is bypassed rather than
disabled: each vae's module vendors its own loader — meta-parameter shell,
`WeightSource.open(path).materialize(device)` so tensors land on the gpu one
at a time (peak ram = largest tensor, never the model),
`load_state_dict(strict=True, assign=True)` — the pattern minimax_h3
established for its video/audio vaes. `qwen_image` follows it
(`toolkit/models/v2/vae/qwen_image.py`), which is why that module also vendors
the vae's architecture + latent normalization (`QWEN_IMAGE_VAE_CONFIG`): a
load needs the weights file and nothing else. Computed work may upcast where
the model is built for it; degrading stored weights may not.

**Cache identity is the id.** There is no latent-space-version concept on
the registry: the dataloader keys latent caches on `model.vae_id` directly
(`data_loader.py` — a model without an id fails loudly there), so every
model declaring `qwen_image` shares one cache namespace and one set of
files. Any deliberate semantic change to a vae's output is made by the vae
module's owner (the quality pass) and handled by deleting/stranding the id's
cache directory — caches are cheap.

**Onboarding a model** (the H3 migration, commit `c578d205`, is the
reference):

1. Write the id module next to the model class (e.g.
   `minimax_h3_vae_id.py`): import the component classes + bundle, define
   `load`, `register_vae_entry(VaeEntry(...))`. Import it from the
   extension's `__init__.py`.
2. Declare `vae_id` on the model class; remove its `load_vae` override /
   `_load_vae` body (and `_load_vae`-era constants), any vae entries in a
   model-level `COMFY_FILES`-style table, and any in-body
   `latent_space_version` assignment (identity is the id; there is no
   version to set).
3. Duplicate ids, unknown ids, and cast policies on bundles all raise at
   registration/construction — do not soften these into warnings.

**A vae-keyed cache holds the vae's output, nothing else.** What is
written under a `vae_id` is exactly what the entry's encode returns --
never a model's derived post-transform (models with deterministic
per-step layers on latents must recompute at batch time from the shared
latents, or keep their derived tensors in a separate model-keyed cache of
their own identity). A file carrying derived bytes under a vae tag breaks
the one promise the registry exists to keep: same id, compatible latents,
any model.

**Migration is replacement.** An agreed migration deletes the path it
supersedes in the same commit — no dual-path flags, no try-registry-except
-fallback, no feature toggles kept "for safety". The opt-in in rule 3 of §4
is about the *shared dispatch* staying inert for models that never declare
an id; it is never a reason to keep a migrated model's old loader alive. If
the old path must stay reachable, the migration is not agreed — ask.

**Byte identity is a guard, not the goal.** Diffing cache bytes old-path vs
new-path proves a *structure-only* move (the Stage A gate for the H3
migration: 40/40 files, 80/80 tensors, 40/40 meta dicts). It must never
veto a semantic change and it is not a gate on new work: caches are cheap,
and a deliberate quality/semantic change to a vae's output ships when the
vae module's owner says so — the id's cache directory is then cleared,
full stop. No version strings, no migration code. Quality is the goal;
identity only proves the move itself changed nothing.

## 6. Where the pieces live

| Concern | Anchor |
|---|---|
| Candidate ranking + download policy | `resolver.py:19` `comfy_precision_rank`, `:66` `resolve_comfy_candidates` |
| Model file registration | `_mixin.py:109/116/122/145` |
| vae_id registry + entry shape | `toolkit/models/vae_registry.py` |
| Registry dispatch + construction-time validation | `phased_load.py` `PhasedLoadMixin.__init__` / `load_vae` / `encode_images` / `decode_latents` |
| Cache identity = vae_id (loud refusal without one) | `data_loader.py` (latent cache keying) |
| Reference vae_id migration (bundle, vendored loaders, entry-owned encode/decode, kill-list) | `minimax_h3/minimax_h3_vae_id.py` |
| Reference single-vae migration (vendored loader + vendored config, entry-owned encode/decode) | `toolkit/models/v2/vae/qwen_image.py` (the `qwen_image` id; krea2 declares it) |
| Lifecycle phase selection (`phases`) | `config_modules.py` `TrainConfig.phases`; guards/exits in `BaseSDTrainProcess.run()` |
| Lazy reader, per-key placement | `weight_source.py:25` `:90` |
| Stream-to-meta-shell reference | `ltx2.py:1609` `_stream_load`, `:1593` `_remap_file_key` |
| Batched path only when required | `ltx2.py:1570` `_as_shipped_stream_target` |
| Pre-quantized attach | `comfy_quant_import.py:215`, `ltx2.py:1658` |
| Variant filtering hook | `phased_load.py:36` `select_comfy_candidates` |
| Phased entry points (process-driven) | `phased_load.py:74/95/112`, `load_model` `:121` |
