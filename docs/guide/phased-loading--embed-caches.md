# Phased Loading — Embed Caches (2/4)

Part of the `phased-loading--` guide packet:
[component phases](./phased-loading--component-phases.md) ·
[staged sampling](./phased-loading--staged-sampling.md) ·
[loader conventions](./phased-loading--loader-conventions.md)

Anchors are as of commit `22f1c08` (2026-10-07).

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
have their own knob, `latent_space_version`, base_model.py:217.)

## 2. Caption caches — per dataset item

Owner: `TextEmbeddingCachingMixin` (`toolkit/dataloader_mixins.py`).

- **Path scheme**: `<media_dir>/_t_e_cache/<media_name>_<hash>.safetensors`
  (`_build_text_embedding_path`, `:2316`). The hash is base64-md5 of
  `get_text_embedding_info_dict` (`:2277`): caption text,
  `text_embedding_space_version`, `text_embedding_version`, plus control
  conditioning when present. Edit the caption file → different hash → the old
  entry is simply never looked at again.
- **One item can have up to five targets** (`_caption_embed_targets`,
  `:2484`): the caption, its DOP variant (trigger-stripped, only if the
  trigger actually appeared), the dropout (blank/trigger-only) variant under
  `caption_dropout_rate`, and the D-OPSD self-ref teacher variants. All share
  the same hash scheme with different caption overrides.
- **Writing**: `cache_text_embeddings()` (`:2549`) runs in phase 2 and
  encodes only targets failing `_embed_target_valid`.
- **Consumption**: at train time the DTO loads its files with
  `PromptEmbeds.load` (`:2456`, dop `:2469`, dopsd `:2475`) — dispatch again.
- **Completeness**: `text_embedding_complete()` (`:2537`) is what the TE gate
  asks per dataset; it walks the same `_caption_embed_targets` list.

## 3. Fixed-prompt caches — shared across jobs

Owner: `SDTrainer` (`extensions_built_in/sd_trainer/SDTrainer.py`).

Kinds (`_fixed_embed_targets`, `:202`): `unconditional` (the negative prompt),
`blank`, `trigger`, `dop_class`, and one `sample_pos_N`/`sample_neg_N` pair
per sample prompt. Each target is `(kind, text, flags)`.

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

`needs_text_encoder_load` (`SDTrainer.py:308`), in order: any dataset
incomplete → load; `train_text_encoder` / `embed_config` / `is_llm` → load;
validation items → load (**frozen clause**, `:321`-`:328`); live-encoding
mode → load; live-embed needs (`_compute_live_text_encoder_needed`, `:174`) →
load; fixed targets missing/invalid → load; otherwise skip and every embed is
loaded from disk instead.

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
