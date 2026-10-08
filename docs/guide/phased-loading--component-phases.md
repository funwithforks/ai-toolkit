# Phased Loading — Component Phases (1/4)

Part of the `phased-loading--` guide packet:
[caches](./phased-loading--embed-caches.md) ·
[staged sampling](./phased-loading--staged-sampling.md) ·
[loader conventions](./phased-loading--loader-conventions.md)

Anchors are as of commit `22f1c08` (2026-10-07). Verify each file:line before
relying on it; this document is the authoritative current-state map, the notes
in `docs/refactor_notes/` are point-in-time research and no longer authoritative.

## The contract in one paragraph

The training process (`jobs/process/BaseSDTrainProcess.py`) owns a strict
lifecycle: each phase brings up exactly the components that phase consumes,
runs all preparation that needs them, then frees them. A component is never
resident outside its phase unless a config knob (`vae_training_mode`) or a
model-declared feature (`staged_sampling`) moves a specific consumer into a
phase. The model side of the contract is the `PhasedLoadMixin` docstring
(`toolkit/models/phased_load.py:1-31`) — it lists the required
`_load_vae/_load_text_encoder/_load_transformer` hooks and the optional
pre/post hooks. This document covers the *process* side: ordering, gates, and
every release point.

## Lifecycle order (run())

| Anchor | Step | Resident after |
|---|---|---|
| `:1865` | **phase 1: vae + latent caching** begins | — |
| `:1912`–`:1934` | `vae_load_needed` gate; `cache_latents_all_latents()` only inside the gate | vae (if gate true) |
| `:1938`–`:1943` | **IRON RULE line**: phase 1→2 boundary; anything needing the vae ran in phase 1 | vae only if `vae_training_mode` keeps it |
| `:1955` | **phase 2: text encoder + embedding caching** begins | — |
| `:1956` | `needs_text_encoder_load()` gate | TE (if true) |
| `:1963` | `dataset.cache_text_embeddings()` (caption caches) | TE |
| `:1991` | `cache_pre_train_text_embeddings()` (fixed + sample prompt caches) | TE |
| `:2007` | `sd.release_encode_phase_components()` — model-owned phase-2 extras freed | — |
| TE unloaded by `:1956`-phase end | | — |
| `:2009`–`:2010` | **phase 3: transformer** (`sd.load_transformer()`) | transformer only |
| train loop | every step consumes: transformer + cached latents + cached embeds | transformer only |

Validation prep is split across the same boundaries on purpose: images→latents
runs while phase 1 holds the vae (`:1622`–`:1623`), prompts→embeds while phase
2 holds the TE (`:1694`–`:1695`). The validation clauses in both gates are
frozen — see the DO-NOT-TOUCH comment at `SDTrainer.needs_text_encoder_load`
(`extensions_built_in/sd_trainer/SDTrainer.py:321`–`:328`) and its paired
VAE-gate clause (`BaseSDTrainProcess.py:1925`).

## Sampling rounds (sample())

- Non-staged models: if `vae_training_mode == 'reload'` and the vae is not
  resident, the vae is loaded for the whole round (`BaseSDTrainProcess.py:382`
  –`:393`) and freed again after (`:396`–`:400`, "VAE unloaded until next
  sample round").
- `staged_sampling` models skip that preload (`:385` clause) — the denoise
  loop must run vae-free; the model loads and frees vae components itself in
  its decode phase. Details: [staged sampling](./phased-loading--staged-sampling.md).

## Gate inventory (every decision that forces or skips a load)

| Gate | Anchor | Forces load when |
|---|---|---|
| VAE phase-1 | `BaseSDTrainProcess.py:1916` | `vae_training_mode == 'keep'`, or any dataset needs live/cached latents, or validation items exist (frozen paired clause `:1925`) |
| TE phase-2 | `SDTrainer.py:308` | any dataset `text_embedding_complete()` false; `train_text_encoder`; embed_config; `is_llm`; validation items (frozen clause); live-encoding mode; `_compute_live_text_encoder_needed()` (`:174`); `fixed_embeds_cached()` false (`:290`) |
| Sampling vae preload | `BaseSDTrainProcess.py:382` | `vae_training_mode == 'reload'` and vae absent and not `staged_sampling` |

Rules learned the hard way in this repo (regressions, not theory):

1. **Every gate branch must print its reason** when it forces a load — the
   lines quoted in logs ("Loading vae for sampling", "Loading fixed prompt
   embeddings from cache", "Skipping text encoder load - every embedding is
   cached") are the gate's explanation surface. Silent forced loads are
   regressions-in-waiting.
2. **A new consumer of a gated resource declares itself in the gate** (count
   it in the condition) — e.g. staged sampling declared itself via the
   `:385` clause instead of quietly relying on the vae being there.
3. **`self.thing = None` is not a free.** Frees go through `MemoryManager` /
   device-state restore, or explicit `del` + `flush()` as the phase-boundary
   code does.

## Model-owned phase components (the connectors pattern)

Some models hold a small component that only exists to *convert* embeddings
during phase 2 (LTX-2.5's text-embedding connectors). The process cannot know
about each model's extras, so the boundary is a hook:

- `BaseModel.release_encode_phase_components()`
  (`toolkit/models/base_model.py:430`) — no-op default, called once at
  `BaseSDTrainProcess.py:2007` between phase 2 and phase 3.
- The model lazy-loads the component *during* phase 2 with a printed reason
  (`LTX2Model._ensure_connectors`, ltx2.py:1775) and frees it in the hook
  (ltx2.py:1795). Cache formats ensure warm runs never load it at all
  ([embed caches](./phased-loading--embed-caches.md)).

The same pattern applies to any future "loads only to produce cached state"
component: lazy-load with a printed reason, free at the hook, and make the
cache format check reject anything a stale run could have written.

## Inference/generation path

`load_model()` (phased_load.py:121) composes all components at once for
non-training runs; that is the *other* sanctioned residency regime and does
not obey the phase rules. Progress reporting loads the big component first.
