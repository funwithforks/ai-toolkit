# Phased Loading — Staged Sampling (3/4)

Part of the `phased-loading--` guide packet:
[component phases](./phased-loading--component-phases.md) ·
[embed caches](./phased-loading--embed-caches.md) ·
[loader conventions](./phased-loading--loader-conventions.md)

Anchors are as of commit `22f1c08` (2026-10-07).

Staged sampling splits a sample round into a **denoise pass** (transformer
resident, vae absent) and a **decode pass** (vae resident, transformer
stepped aside). It exists so a big transformer has the whole card to itself
during the expensive loop; the vae only returns at the end to turn latents
into pixels. It is opt-in per model and must stay completely invisible to
models that do not set it.

## The switch

`BaseModel.staged_sampling` (`toolkit/models/base_model.py:385`), default
`False`. A model opts in with a class attribute `staged_sampling = True`
(e.g. H3 `minimax_h3.py:779`, LTX-2.5 `ltx2.py:1408`).

The process honors it in exactly one place: the sampling vae preload in
`BaseSDTrainProcess.sample` is skipped for staged models
(`BaseSDTrainProcess.py:385`) — otherwise the vae would be sitting on the card
for the denoise loop, defeating the feature. When the round ends the same
`vae_training_mode` teardown frees the vae the model loaded for decoding
(`:396`-`:400`).

## What a staged model implements

Two hooks, one required and one optional:

1. `generate_single_image(...)` returns a **latents payload** (any
   model-defined dict/tuple), not a decoded image. The base loop detects
   `staged_sampling` and collects `(gen_config, idx, payload)` instead of
   saving (`base_model.py:831`-`:836`).
2. Decode, one of:
   - `decode_staged_samples(staged_samples)`
     (`base_model.py:851`) — **model owns the whole decode phase**. Used when
     more than one vae must be brought up and freed independently (H3 video vs
     audio vae; LTX-2.5 video/audio/vocoder). If present, the base calls it and
     returns; the base never loads a vae.
   - `decode_sample_payload(payload) -> image`
     (`base_model.py:387`) — the simpler fallback: base loads one vae
     (`"Loading VAE to decode staged samples"`, `:861`) and calls this per
     payload. H3 keeps it defined but unused while `decode_staged_samples`
     exists (`minimax_h3.py:1325`).

Dispatch order (`base_model.py:851`): if the model defines
`decode_staged_samples` it wins; otherwise the base loads the vae and loops
`decode_sample_payload`. Both paths then do the identical tail the non-staged
path does — `save_image_atomic`, `log_image`, `_after_sample_image`, `flush`.

## Invariants a staged model must keep

- **Never load the vae during the denoise pass.** Denoise consumes cached
  embeds + the transformer only. If the loop reaches for a vae, the feature is
  not actually working — that is a silent-cost bug, not a free one (see the
  repo rule that a failure which looks costless is still broken).
- **Latent spaces are the contract** between the two passes. Whatever
  `generate_single_image` stashes, `decode_*` must invert. Keep payloads
  exactly as the transformer emitted them (normalized, packed/unpacked as the
  model's own decode expects) — do the inverse transforms in the decode phase,
  where the vae's stats are available.
- **Pixel/audio tensors go to CPU** at decode time; only one vae is resident
  at a time when there are several.
- **The non-staged path must not change.** Every line of the base that a
  `staged_sampling=False` model touches is unchanged; the feature lives behind
  one `if` per site.
- **Control / latent-conditioned samples**: staged models must refuse them
  loudly (raise) rather than silently fall back to a resident-vae path — LTX-2.5
  `NotImplementedError` in its `generate_single_image`.

## Reference: LTX-2.5 (three components, three sub-loads)

`staged_sampling = True` (`ltx2.py:1408`).

- Denoise: `generate_single_image` (`ltx2.py:1410`) feeds cached
  connector-space embeds to `denoise_staged(...)`
  (`staged_sample.py`, a transcription of the pinned `LTX2Pipeline.__call__`
  loop) and returns `{latents, audio_latents, meta, is_video, fps}`. No vae,
  no connectors touched.
- Decode: `decode_staged_samples` (`ltx2.py:1494`) — the model-owned branch.
  It brings up the video vae (`_stream_video_vae`, `:1974`) alone, decodes all
  payloads, frees it; then audio vae + vocoder (`:1993`, `:2018`), decodes
  waveforms, frees them; then muxes mp4/writes. Frames and waveforms are moved
  to CPU before the save loop. Reuses the same per-component `_stream_*`
  builders `_load_vae` composes (`:2044`), so there is one loader per component,
  not two.

## Reference: H3 (two vaes, per-payload API but model-owned decode)

`staged_sampling = True` (`minimax_h3.py:779`). `generate_single_image`
(`:1270`) passes `decode=not self.staged_sampling` into its sampler so the
loop skips the in-loop decode (`:1315`); returns latents when staged
(`:1317`). `decode_staged_samples` (`:1342`) loads/frees the video then audio
vae one at a time, mirroring LTX-2.5. `decode_sample_payload` (`:1325`) is the
kept-but-unused single-payload fallback.

## Vendoring rule for transcribed sampler loops

`staged_sample.py` copies the diffusers pipeline loop only because the loop
and the decode are being pulled apart — the math is not ours. Every helper it
calls (`retrieve_timesteps`, `calculate_shift`, `rescale_noise_cfg`,
`LTX2Pipeline._*_latents`, `randn_tensor`) is the real pinned diffusers code,
and latent-space pack/unpack uses `transformer.config` patches + the vae
config's own compression ratios, never re-derived. If you touch a transcribed
loop, diff it against the pinned pipeline call; do not "simplify" it.
