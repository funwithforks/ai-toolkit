# Phased Loading — Embed Caches (2/4)

Part of the `phased-loading--` guide packet:
[component phases](./phased-loading--component-phases.md) ·
[staged sampling](./phased-loading--staged-sampling.md) ·
[loader conventions](./phased-loading--loader-conventions.md)

Anchors are as of commit `22f1c08` (2026-10-07); sections added 2026-11-12
(control content keying, sample-embed caching, reference-latent sidecars)
anchor to their own commits.

This is the machinery that lets a warm run print
`Skipping text encoder load - every embedding is cached` and never pay for
the multi-GB encoder. It is three cooperating caches plus one on-disk format
system. If you are adding a model or changing what gets encoded, read this
before writing any new caching code — it already covers more than you think.

## 1. The on-disk format system (one file format, three classes)

Any embed on disk is a safetensors file written through:

- `PromptEmbeds.save` (`toolkit/prompt_utils.py:120`): keys
  `text_embed[_i]`, optional `pooled_embed`, `attention_mask[_i]`; **no
  metadata**.
- `AdvancedPromptEmbeds` (`toolkit/advanced_prompt_embeds.py`): an arbitrary
  named set of tensors (`connector_prompt_embeds=…` etc.). `save` (:138)
  writes metadata `class_name` + `frozen_dtype_keys`; requires exactly one
  tensor per key per save (`:144`); non-float tensors are auto-frozen against
  dtype casts on load (`:172`-`:175`). `concat_prompt_embeds` (:179) and
  `split_prompt_embeds` (:197) compose them across batch items.
- `AnimaPromptEmbeds` — model-local subclass, same mechanism.

**Loading is always `PromptEmbeds.load(path)`** — it dispatches on the
`class_name` metadata (`toolkit/prompt_utils.py:152`-`:157`) and returns the
right class. Never hand-open an embed file; the dispatcher is the contract.
`concat_prompt_embeds` (`toolkit/prompt_utils.py:260`) is likewise
class-dispatched, so mixing classes in one batch is safe.

`text_embedding_space_version` (`toolkit/models/base_model.py:296`) is the
model's knob for "my embed tensors changed shape/semantics" — bump it and
every path below changes, invalidating old caches automatically. (Latents
are keyed on `vae_id` — [loader conventions §5](./phased-loading--loader-conventions.md);
the old `latent_space_version` name survives only in non-phased legacy code.)

## 2. Caption caches — per dataset item

Owner: `TextEmbeddingCachingMixin` (`toolkit/dataloader_mixins.py`).

- **Path scheme**: `<media_dir>/_t_e_cache/<media_name>_<hash>.safetensors`
  (`_build_text_embedding_path`, `:2316`). The hash is base64-md5 of
  `get_text_embedding_info_dict` (`:2277`): caption text,
  `text_embedding_space_version`, `text_embedding_version`, plus control
  conditioning when present. Edit the caption file → different hash → the old
  entry is simply never looked at again.
- **Control conditioning is content-keyed** (commit `efad18a3`): for
  `encode_control_in_text_embeddings` items the hash dict carries the
  resolved control paths, their `size:mtime` content signatures, and (for
  archs that size references against the target) the item's crop dims — so
  two items sharing a caption never collide, and a control image edited in
  place lands on a fresh embed file instead of serving stale vision blocks.
  The per-dataset `cache_key_on_content: false` opt-out drops the
  signatures for stub-slimmed datasets (see §7).
- **One variant table** (`get_embed_variants`): the ONLY place an item's
  text-embed variants and their gates are stated — caption, DOP class twin
  (DOP only), shared dropout blank (the empty caption -- `''`, one
  content-addressed file per media dir per model; DOP adds one shared
  class-side entry the same way), and the fenced per-file D-OPSD teacher
  pair. The phase-2 planner (`_caption_embed_targets`), the train-time
  reader (`load_prompt_embedding`), and the completeness gate all consume
  this same table and dedupe by path (first wins — equal paths carry
  byte-equal hash dicts). Real-caption targets hash with a per-item
  basename prefix; the shared blank/class-side entries have none.
- **Writing**: `cache_text_embeddings()` (`:2549`) runs in phase 2 and
  encodes only targets failing `_embed_target_valid`.
- **Consumption**: at train time the DTO loads its entries by TABLE NAME
  from `get_embed_variants` with `PromptEmbeds.load` — the same table the
  planner wrote, so a path can never be read differently from how it was
  written; class-dispatch on load remains the file-format contract.
- **Completeness**: `text_embedding_complete()` (`:2537`) is what the TE gate
  asks per dataset; it walks the same `_caption_embed_targets` list.

## 3. Fixed-prompt caches — shared across jobs

Owner: `SDTrainer` (`extensions_built_in/sd_trainer/SDTrainer.py`).

Kinds (`_fixed_embed_targets`): `unconditional` (the negative prompt),
`blank`, `dop_class`, and one `sample_pos_N`/`sample_neg_N` pair
per sample prompt. Each target is `(kind, text, flags)`. There is
deliberately NO `trigger` kind: a trigger word never belongs in a
dropout slot (owner contract 2026-10-27; see
`docs/refactor_notes/DECISION--blank-embeds-and-dop-gating.md`).

- **Path scheme** (`_fixed_embed_cache_path`, `:248`):
  `<training_folder>/.prompt_embed_cache/<md5(model|space)[:12]>/<kind>_<md5(kind|model|space|text|flags)[:16]>.safetensors`
  — **shared by model identity, not by job**. What a prompt encodes to does
  not depend on which job asked; a new job must not re-load the TE just to
  re-encode prompts another job already cached. A legacy per-job location
  (`save_root/.prompt_embed_cache`) is still read if the shared file is
  absent.
- The sample-prompt pairs are reassembled into
  `sd.sample_prompts_cache` (`_load_fixed_embeds_from_cache`, `:340`) and
  consumed at generation by `BaseModel.generate_images`
  (`toolkit/models/base_model.py:660`-`:662`) — sampling never needs the
  encoder once the cache exists.
- **Control-encoding models cache their sample embeds too** (commit
  `efad18a3`, closing the writer/gate divergence): a sample with
  `ctrl_img*` set used to be encoded live every round with the TE resident.
  Now the ctrl identity — resolved paths, content signatures, target size —
  rides in the target flags via `_sample_ctrl_flags`, and both the gate
  (`_fixed_embed_targets`) and the writer (`cache_sample_prompts`) parse the
  sample identically (ctrl fields seeded in both). A changed control lands
  on a fresh cache file; a warm controlled run needs no TE.
- **Persistence guard**: entries are only written when a later run could
  reuse them (`unload_text_encoder` or dataset embed caching — the same guard
  in `_fixed_embed_targets` and `cache_sample_prompts` `:364`). A job that
  encodes live every epoch does not litter the shared cache.
- **Writes** go through `_save_fixed_embed` (`:273`): detach → cpu → save;
  failures print per-kind and degrade to re-encode, never crash a run.

## 4. The validity layer (`embed_file_valid`)

Existence of a file is not enough when a model's cache stores a *derived*
representation (e.g. post-projection embeds): a file from an older format
loads fine and poisons training silently.

- Contract: `BaseModel.embed_file_valid(path)`
  (`toolkit/models/base_model.py:422`) — default `True` = the long-standing
  exists-only semantics; **all models without an override behave exactly as
  before**.
- Wired into every completeness path:
  `_embed_target_valid` (dataloader_mixins.py:2528) and
  `_fixed_embed_valid` (SDTrainer.py:282). An invalid entry is simply
  re-encoded in phase 2 and **overwrites** the old file.
- Reference implementation: `LTX25Model.embed_file_valid` (ltx2.py:1879) —
  reads metadata `class_name == "AdvancedPromptEmbeds"` + required keys;
  treats only real file damage as invalid (a swallowed code bug there would
  silently re-encode everything, forever).

## 5. What makes a warm run skip the TE entirely

`needs_text_encoder_load` (`SDTrainer.needs_text_encoder_load`), in order: an
arch declaring no text encoder (`has_text_encoder=False` — an llm whose loss
consumes token ids from the cached artifact) → skip, prints; any dataset
incomplete → load; `train_text_encoder` / `embed_config` → load; validation
items → load (**frozen clause**); live-encoding mode → load; live-embed needs
(`_compute_live_text_encoder_needed`) → load; fixed targets missing/invalid →
load; otherwise skip and every embed is loaded from disk instead (the serve
branch in `cache_pre_train_text_embeddings`).

The encoderless refusal is keyed on the arch declaration (`has_text_encoder`),
never on the caching flags themselves: flags-set is the *normal* state of a
warm caching run, so a refusal keyed on the flags refuses every with-TE warm
resume (the Nov 2026 regression, `355c5fb1` — fixed by restoring the
arch-keyed form). An encoderless arch that sets the flags is refused loudly,
and the phase-2 dataset pass raises the same refusal before
`cache_text_embeddings` can crash inside `encode_prompt`.

## 6. New-model checklist

1. Implement `get_prompt_embeds`; return `PromptEmbeds` — or `AdvancedPromptEmbeds`
   if downstream consumes post-encoder transforms; encode via
   `encode_static_prompt` (SDTrainer.py:144) for fixed prompts so flags land
   in the cache key.
2. If the returned tensors are not what the encoder emits raw, implement
   `embed_file_valid` rejecting anything not in the current format.
3. If a module is loaded only to convert embeds (connectors, projectors),
   lazy-load it in `get_prompt_embeds` with a printed reason and free it in
   `release_encode_phase_components` — see
   [component phases §Model-owned phase components](./phased-loading--component-phases.md).
4. Do not add a fourth cache. Caption + fixed + sample-target caching
   generalizes; if something really doesn't fit, extend `_fixed_embed_targets`
   kinds rather than inventing a parallel store.
5. Verify the acceptance line on a real job, cold and warm: first run
   encodes and writes; second run prints the skip lines and loads nothing on
   the GPU except the transformer.
6. Read back what the caches stored and check the dtypes, not only the shapes:
   the tensors must be in the precision the producing component ships in, not
   the job's dtype. Existence and key checks (`embed_file_valid`) cannot see a
   precision change, so the check is done on a freshly written file. See the
   producer -> cache cast sweep in
   [loader conventions §5](./phased-loading--loader-conventions.md).

## 7. Control reference-latent sidecars — the vae-side analog

§6.4's "do not add a fourth cache" governs *embed* stores; control reference
latents are a phase-1 *latent* cache and live next to the target latents.
Commit `fd28556a`. Models that VAE-encode control/reference images at train
time (minimax_h3 ref2va family) instead encode them once in phase 1:

- **Opt-in**: the model declares `supports_control_latent_cache` and
  implements `get_control_reference_size` + `encode_control_reference`
  (`toolkit/models/base_model.py`). Everything else keeps per-step encoding.
- **Path scheme**: one sidecar per training item,
  `<media_dir>/_latent_cache/<media_name>.ctrl.safetensors`, written by
  `AiToolkitDataset.cache_control_reference_latents` after the target-latent
  pass (same vae residency window). Each stored reference is the *clean*
  resized latent; the model's per-step noise augmentation stays per-step.
- **Validity** (`_control_latent_sidecar_valid`): control file count +
  `size:mtime` content signatures + the item's bucket dims + the encoding
  `vae_id` (so two models sharing a dataset never reuse another vae's
  reference latents). Any mismatch → regenerate; the batch DTO carries
  `control_latent_list` and the model skips its per-step encode entirely.
- **Residency**: a model declaring `require_vae_during_training` keeps the
  vae only while sidecars are pending; the ladder re-evaluates after the
  phase-1 passes and downgrades to reload/unload with a printed reason (see
  [component phases](./phased-loading--component-phases.md)). Stochastic
  controls (same-folder pairing, aug replay) are not cacheable and keep the
  old path.
- **Acceptance line**: first run prints `Caching control reference latents`
  once and the vae loads; the second prints the skip lines and loads nothing
  on the GPU except the transformer. Verified on
  `20261112-143452_h3_train_v2` and `20261112-163024_h3_train_v2`.

## 8. Media-content keying and the stub-file opt-out

Every cache above that depends on source pixels keys on the media's
`size:mtime` signature (`get_quick_signature_string`): target latents
(`get_latent_info_dict`), control sidecars, and control-aware embed keys.
An image or video edited in place lands on a fresh cache file instead of
serving stale tensors. The old file is orphaned, never consulted.

`DatasetConfig.cache_key_on_content` (default on) drops the signatures for
datasets whose media is replaced by dimension-matched stub files *after*
caching: keying on a stub would re-encode it, and a stub-encoded latent or
embed is garbage. With the opt-out, validity falls back to path/geometry —
exactly the pre-signature behavior. The flag describes the media, not the
model: keep it consistent across the jobs that touch a dataset folder, or
each keying regime writes its own parallel cache set.
