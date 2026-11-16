# krea2 latent dataset generation

This feature **exists, is committed, and is tested end-to-end**. A future
session arriving here because it thinks latents need to be saved-to-disk or
that a generation script needs to be written: **it does not need to be
built.** Read section 1, then section 6 (the annotated job config), and you
can generate a dataset without opening any other file.

## 1. The one-paragraph answer

A latent dataset is produced by a **normal `sd_trainer` job with
`train.steps: 0` and `model_kwargs.latent_output: true`** on the krea2 arch.
The job loads the text encoder, caches prompt embeddings, loads the
transformer, runs the normal sample rounds, and saves each result as a
**fp32 latent safetensors** whose metadata carries the prompt, resolution,
seed, steps, guidance, schedule, and model identity. The VAE is never
loaded, nothing is decoded, no images are produced. The save path is
`Krea2Model.decode_staged_samples` (krea2.py), driven by the existing
staged-sampling contract — there is no standalone script, no new job type,
and no new config schema. A verified template config ships at
`docs/guide/krea2_latent_dataset.yaml`.

## 2. Common misreadings, pre-answered

These are conclusions other agents have reached from reading the code. Each
is wrong for the current tree.

- **"No code path saves latents to disk; every round ends in decode."**
  Wrong. `Krea2Model.decode_staged_samples` (krea2.py, under
  `generate_single_image`) saves each staged payload as a safetensors
  dataset file with full metadata and an atomic tmp+rename. It is selected
  by the existing `hasattr(self, "decode_staged_samples")` dispatch in
  `BaseModel.generate_images`.
- **"Which model — staged or non-staged?"** krea2. Setting
  `model_kwargs.latent_output: true` makes `Krea2Model.staged_sampling`
  true, which rides the existing staged-sampling contract
  (docs/guide/phased-loading--staged-sampling.md) with zero base changes:
  the sampling-round VAE preload is skipped
  (BaseSDTrainProcess.py, the `not getattr(self.sd, 'staged_sampling', ...)`
  gate), `generate_single_image` returns raw fp32 latents, and the model
  owns the "decode" phase.
- **"Option A (sample rounds save latents) or option B (standalone
  script)?"** It is A-shaped, and it already exists: sample rounds save
  latents. There is no standalone script and none is needed — the job IS
  the generation entry point (`run.py <config>.yaml`).
- **"Feed the latents back in via GenerateImageConfig.latents?"** No.
  `GenerateImageConfig.latents` is an *input* latent (img2img-style init)
  and unrelated. The output of this feature is a **VAE training dataset**:
  fp32 latents consumed by a VAE training job's dataloader, not by
  krea2 sampling.
- **"Use steps: 0 + drop_transformer_for_decode."** No.
  `drop_transformer_for_decode` is a one-shot VAE-decode *testing* switch
  that frees the transformer; the latent job keeps the transformer
  resident and never decodes. Just set `steps: 0` and do not set that flag.
- **"The VAE preload must also be skipped."** Already handled: the
  staged-sampling gate skips it; nothing extra to set.

## 3. Requirements

- MODELS_PATH (or the hub via `HF_ENDPOINT`) containing:
  - `diffusion_models/krea2_turbo_int8_convrot.safetensors`
  - `text_encoders/qwen3vl_4b_int8_convrot.safetensors`
  Both resolve through the comfy-candidates door / hub reference in the
  template config; local files under MODELS_PATH win.
- A CUDA GPU (developed and measured on RTX 5090 and RTX 6000 Pro).
- First run builds one CUDA extension
  (`toolkit/kernels/rot_fht/fused.cu`, ~1 minute, cached afterwards).
  If the build fails the job still runs — it prints a notice and falls
  back to the gemm rotate + act-quant chain (slower, same outputs up to
  fp32 sum order).

## 4. The job config, every key explained

Start from `docs/guide/krea2_latent_dataset.yaml` — it is
verified end-to-end and carries generic example prompts. The keys that
matter:

### job / process shape

- `job: extension`, process `type: sd_trainer` — a normal trainer process;
  there is no latent-generation job type.
- `training_folder: output` — relative to the repo root is fine; outputs
  land under `<training_folder>/<config.name>/samples/`.

### model block

- `arch: krea2`, `name_or_path: krea/Krea-2-Turbo` — the turbo checkpoint
  is the distilled 8-step model; the raw model needs different sampling
  settings and is not what this template targets.
- `dtype: bf16` — **required**. Omitting train dtype resolves the model to
  fp32 and every step runs ~2.5x slower (measured); this is the single
  easiest way to lose an hour.
- `quantize: none` — keep the shipped int8 quantization exactly as-is.
- `model_kwargs`:
  - `checkpoint_filename: diffusion_models/krea2_turbo_int8_convrot.safetensors`
    — resolved against MODELS_PATH; the turbo int8 convrot checkpoint.
  - `text_encoder_path: Comfy-Org/Krea-2/text_encoders/qwen3vl_4b_int8_convrot.safetensors`
    — hub-form reference: a local copy under MODELS_PATH wins, otherwise
    it downloads once. (The int8 TE is intentional; fp8 is not needed at
    any resolution on a 550 W 5090.)
  - `int8_bwd: false` — the int8 STE *backward* is a training feature;
    inference never uses it. Set false to skip a misleading log line.
  - `schedule_mu: 1.15` — the turbo distillation's shift for the mu
    schedule. Unused on the simple schedule (below) but harmless to set.
  - `sample_schedule: simple` — ComfyUI-compatible schedule; measured
    preferred over mu-shift on output quality. Default (`mu_shift`)
    reproduces the repo's historical sampling.
  - `fast_attention_paths: true` — skips the 256-token sequence pad and
    the L x L mask build when the sequence is all-valid. **Inference-only
    opt-in**: without it, training numerics would shift at the ulp level.
    ~45 ms/step at 4MP. Do not set it in training configs.
  - `int8_fused_rotate: true` — fused rotate+hadamard+quantize CUDA kernel
    (~1.6x over the gemm rotate + act-quant chain; wide rows fall back to
    the chain per launch). Training keeps the gemm path; this flag only
    stamps inference modules. Do not set it in training configs.
  - `int8_cutlass_gemm: true` — fused CUTLASS int8 GEMM+dequant instead
    of `_int_mm` + triton epilogue. Composes with the above.
  - `latent_output: true` — **the feature switch.** Makes the model a
    staged-sampling model: sample rounds return fp32 latents and
    `decode_staged_samples` writes them to disk. The VAE is never loaded.
    Refuses loudly on edit/control models.

### train block

- `dtype: bf16` — see above; this is the one that matters.
- `steps: 0` — a sampling-only run. The train loop is empty; no optimizer
  or lr scheduler is built; a config with `steps > 0` and no trainable
  parameters fails loudly instead of silently training nothing.
- `skip_first_sample: true` — without it a steps-0 run samples up to
  three times (pre-loop, first-sample block, post-loop).
- `unload_text_encoder: true` — required for the prompt embeddings to be
  cached to disk in the text-encoder phase and the encoder to be freed
  before sampling. Without it the encoder stays resident for nothing.

### save block

- `save.disable_saving: true` — no model checkpoints or optimizer state;
  a latent job needs neither (dataset resumption is metadata-driven, see
  the stop conditions below — not checkpoint-based).

### sample block

- `neg: ''` — **must be a string** (or omitted-but-set like this). A bool
  `false` reaches the prompt encoder and crashes the concat. Real krea2
  configs use `neg: ''`.
- `guidance_scale: 1.0` — the turbo distillation runs without CFG
  (guidance 0 after the model's 1.0 normalization). The SampleConfig
  default of 7 would double the step cost and wreck a distilled model.
- `sample_steps: 8` — the turbo step count. The default 20 is wrong for
  the distilled checkpoint.
- `seed: 42` — one seed per sample entry unless `walk_seed: true`.
  Recorded in the output metadata either way.
- `prompts` — a list of strings, OR a path to a `.txt` (one prompt per
  line) or `.json` (list of strings) file. Every prompt is generated at
  every entry of `resolutions`.
- `resolutions: [[w, h], ...]` — the resolution cross-product. Widths and
  heights are snapped down to multiples of 16 (VAE f8 x patch 2).
- `num_random_prompts: N` + `random_prompt_seed: S` — optional second
  stop condition: draw N prompts seeded-random from the list instead of
  all of them. Omit to generate all prompts.
- `schedule_mu` is NOT needed here (the simple schedule ignores it) but
  is set in the template so mu-shift runs of the same config stay correct.

## 5. Outputs

- Latents: `<training_folder>/<name>/samples/[time]_..._[count].safetensors`
  — one file per generated latent, fp32, shape `(16, H/8, W/8)`, written
  via a hidden `.tmp` dir + atomic rename (watchers never see a partial
  file). Metadata keys: `prompt`, `negative_prompt`, `width`, `height`,
  `seed`, `steps`, `guidance_scale`, `arch`, `model`, `vae_id`,
  `schedule_mu`, `latent_channels`, `software`, `format: pt`.
- Embeddings: `<training_folder>/.prompt_embed_cache/<model-hash>/` — one
  copy per unique prompt text regardless of how many resolutions reuse it.
  **Delete this directory after the latents are verified**; it is a cache,
  not an output, and it is large (~1-2 MB per prompt).

## 6. Two stop conditions

1. **All prompts x all resolutions**: the default — the sample list is the
   cross product.
2. **N random prompts x all resolutions**: add
   `num_random_prompts: N` (+ `random_prompt_seed` for reproducibility).

**Resume is automatic and metadata-driven**: on start the job scans the
output directory's safetensors headers and skips any sample whose
(prompt, resolution, seed) already exists on disk. A killed run becomes
"run it again"; an extended prompt list generates only the new prompts;
nothing is deleted. Filename-based skipping is impossible (names carry a
generation timestamp), which is why the skip reads metadata headers.
There is no third stop condition.

## 7. Performance (measured, RTX 5090 @ 550 W, RTX 6000 Pro @ 550 W)

Per image, 8 steps, int8 convrot, simple schedule, all fast paths on:

| resolution | this repo (5090) | ComfyUI int8 (RTX 6000 Pro) |
|---|---|---|
| 1MP | 2.0-2.3 s | 2.25 s |
| 1536² | 6.0-6.3 s | 5.77-5.81 s |
| 4MP | 12.9-13.1 s | 10.53-10.65 s |

Notes:

- The GPU power cap matters: at the old lower cap the 4MP gap vs Comfy was
  ~1 s; at 550 W it flipped to a win. Bench comparisons must be same-cap.
- **Batching is a measured loss** (512x512 B=8: 0.98x; 768x512: 0.94x) —
  GPU work scales linearly with batch members and CPU dispatch overlaps
  asynchronously. Do not add batch machinery.
- Per-sample prep (noise draw on-device, embed transfer) is 0.1-0.2% of a
  sample; do not hoist it.
- The rotate+quant fast kernel runs on rows up to a 48 KB shared-memory
  budget (K <= ~12k); the mlp down projections (K=16384) fall back to the
  gemm chain per launch — that is measured optimal, not a gap.

## 8. Training safety

Every fast path above is opt-in and defaults off: `fast_attention_paths`,
`int8_fused_rotate`, `int8_cutlass_gemm` are inference-only stamps,
`latent_output` only exists in the generation config, and the steps-0
optimizer skip only fires when nothing trains. A normal krea2 training
config (no new keys) runs byte-identical to the pre-feature code. Do not
set the inference flags in training configs: the ulp-level numerics shift
they cause are fine for dataset generation and unvalidated for training.

## 9. Known failure modes

- Missing `train.dtype: bf16` → fp32 model, ~2.5x slower steps. The run
  completes; only the timing is wrong.
- `neg: false` (bool) → crash in the prompt encoder (`can only concatenate
  str`). Use `neg: ''`.
- `drop_transformer_for_decode: true` in a latent job → the transformer is
  freed before sampling completes; do not combine it with this feature.
- Extension build failure → loud notice at load, falls back to the gemm
  chain (same outputs to fp32 sum order, ~1.6x slower rotate+quant).
- Setting `int8_fused_rotate`/`int8_cutlass_gemm`/`fast_attention_paths`
  in a training config → ulp-level training numerics shift. Don't.
