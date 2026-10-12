# Comfy Remote Sampling

How to render training samples on a separate ComfyUI box instead of locally,
and how to add workflows for new model archs.

Implementation: `toolkit/comfy_sampler.py`. Hooks into training are three
thin points: `SampleConfig` fields (`toolkit/config_modules.py`), the
backend branch in `BaseSDTrainProcess.sample()` and the exit drain in
`BaseSDTrainProcess.done_hook()` (`jobs/process/BaseSDTrainProcess.py`).

## Requirements

- ComfyUI ≥ 0.3.9 reachable over HTTP from the training machine.
- The Comfy API v2 proxy in front of it, installed and running **on the
  ComfyUI machine** (the proxy must see ComfyUI's model tree, so it runs
  co-located):

  ```bash
  # in the ComfyUI venv, on the ComfyUI box
  pip install comfy-api-proxy

  comfy-api-proxy \
    --host 0.0.0.0 \
    --allow-insecure-bind \
    --comfyui http://127.0.0.1:8188 \
    --port 8189 \
    --comfyui-base-dir /path/to/your/ComfyUI \
    --max-upload-mb 2000
  ```

  `--comfyui-base-dir` is not optional for this feature: without it the
  proxy refuses uploads with `file_path=models/...` and LoRAs can never be
  placed. Verify end to end:

  ```bash
  curl http://HOST:8189/api/v2/jobs            # {"jobs": [], ...}
  curl -X POST http://HOST:8189/api/v2/assets \
       -F "file=@a_real_lora.safetensors" \
       -F "content_type=application/octet-stream" \
       -F "file_path=models/loras/aitk_test/probe.safetensors"
  # then confirm it lists on the ComfyUI side:
  curl -s http://HOST:8188/object_info/LoraLoaderModelOnly \
       | grep aitk_test
  ```

## Enable per job

The comfy keys live in the ordinary `sample:` block next to everything
that already works:

```yaml
sample:
        sampler: "flowmatch"        # ignored with comfy: true (workflow owns it)
        sample_every: 100
        sample_start_step: 0
        width: 1024                 # ignored with comfy: true
        height: 1024                # ignored with comfy: true
        samples:
          - prompt: "a subject standing in a doorway, full shot"
          - prompt: "a subject taking a selfie, smiling"
            # network_multiplier: 0.8   # optional per-item lora strength
        neg: ""                     # ignored with comfy: true
        seed: 42                    # ignored with comfy: true (static seed in workflow)
        walk_seed: true             # ignored with comfy: true
        guidance_scale: 3           # ignored with comfy: true
        sample_steps: 30            # ignored with comfy: true
        comfy: true                 # route sample rendering to ComfyUI
        comfy_url: "http://comfy-box:8189"   # the v2 proxy, not the :8188 UI
```

The remote box has no trained LoRA unless the sampler ships one, so
`comfy: true` without a reachable proxy simply leaves rounds queued in the
journal — training is unaffected either way. `train.disable_sampling:
true` still outranks everything: no rounds happen, ComfyUI is never
contacted.

That is the whole surface. The sampler picks the workflow 1:1 by model
arch: `arch: minimax_h3` → `docs/Comfy/workflows/minimax_h3.json`.
`comfy_workflow_dir` moves the folder; `comfy_workflow_path` overrides the
lookup for one job. An unknown arch aborts the job listing what exists.

With `comfy: true` the workflow owns sampler, steps, cfg, resolution and
seed (`walk_seed` is not applied; the seed is static so step-over-step
samples stay comparable). The job config owns only prompts and per-item
`network_multiplier` (delivered as `strength_model`). Width/height/
`sampler`/`sample_steps`/`guidance_scale` keep their meaning only for
local sampling.

## What one round does

1. Snapshots the network (EMA applied if enabled, multiplier forced 1.0 —
   strength arrives via the workflow) to `output/{job}/comfy_snapshots/`.
2. Uploads it to `models/loras/aitk/{job_name}/` through the proxy.
3. Submits **one job per sample item** (same LoRA, different
   `{{AITK_PROMPT}}`/strength) to `POST /api/v2/jobs`. Training continues
   immediately.
4. Polls `GET /api/v2/jobs/{id}`, downloads outputs into
   `output/{job}/samples/[time]_{step}_[count].<ext>` (same naming as local
   sampling; extension follows the remote output, e.g. `.mp4`).
5. Deletes the asset record and the local snapshot.

The first (pre-training baseline) round ships **no lora at all**: the
sampler bypasses the tagged lora node mechanically — removes it and
rewires its consumers onto the model that fed it — which is the same
graph as exporting the workflow with that node bypassed in the UI. No
snapshot file is written, so a baseline sample never costs an upload.

## Failure semantics

- ComfyUI down or slow: rounds queue in order in
  `output/{job}/comfy_samples.json` and drain when the box is back.
  Training never waits.
- Job restart: the journal is reloaded and unfinished rounds resume in
  order (uploads, resubmits or just collects, depending on how far they got).
- A job that runs and errors remotely marks that one item failed and moves
  on; the reason is kept in the journal.
- Process exit: `done_hook` gives the worker 15 minutes to deliver
  (`wait_for_pending`), then leaves the rest in the journal.

## Adding a workflow for a new arch

1. Build the sampling workflow in ComfyUI for the base model, with the
   LoRA applied through a `LoraLoader` / `LoraLoaderModelOnly` node.
2. Export with **Save (API Format)** — the UI save format is rejected.
3. In the exported JSON, replace exactly three values with the tags:
   - the LoRA node's `lora_name` → `{{AITK_LORA}}` (the node must have a
     `model` input link — a `LoraLoader`/`LoraLoaderModelOnly` in a normal
     chain always does; the first-sample bypass needs it)
   - its strength field (`strength_model`, or `strength_model`+
     `strength_clip` where the node has both) → `{{AITK_LORA_STRENGTH}}`
     as a string
   - the prompt node's prompt text → `{{AITK_PROMPT}}`
4. Keep the workflow's own prompt **generic** ("a subject ..."). Trigger
   text never goes in tracked files; the real prompt is injected from the
   training config at runtime.
5. Save it as `docs/Comfy/workflows/<arch>.json` where `<arch>` is the
   exact `arch:` string the training job uses. Multiple archs sharing one
   sampling setup each get their own copy/named file — the map is 1:1.

The sampler only replaces values equal to a tag (or strings containing
one), so other nodes — fixed seed, schedulers, extra fixed LoRAs — pass
through untouched. `filename_prefix` on save nodes is overridden per round
automatically.

## Workspace on the Comfy box

Everything the trainer places lands under `models/loras/aitk/` and
outputs under `aitk/{job_name}/` in the output folder. Proxy-side asset
records are deleted automatically; **placed lora files are not** — sweep
`models/loras/aitk/` when it grows. Remote output artifacts expire with the
job record (~24 h).

## Smoke check without training

```bash
# proxy alive + job round-trip shape
curl http://HOST:8189/api/v2/jobs
curl -X POST http://HOST:8189/api/v2/jobs -H 'Content-Type: application/json' \
     -d '{"workflow": {}, "metadata": "probe"}'   # expect 422 invalid_workflow
```

A real single-item round runs through the same code path used by training;
run it by pointing a scratch job's `sample:` block at the workflow.
