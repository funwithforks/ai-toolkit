"""LTX-2.3 component file registration and resolution.

Two source layouts exist for 2.3 and both must work:

- the upstream mono checkpoint (Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors,
  ~46 GB bf16): one file carrying every component under comfy prefixes
  (``model.diffusion_model.``, ``vae.``, ``audio_vae.``, ``vocoder.``,
  ``text_embedding_projection.``).
- ComfyUI-style split files (Kijai/LTX2.3_comfy): a transformer-only file
  (several precision variants), a separate text-projection file, and bare-key
  vae files. Component classes below register those as candidates so the
  shared resolver door (local-first, download-when-missing, precision-ranked)
  works unchanged.

Candidate tables live on 2.3-specific subclasses -- never on the classes the
2.5 rail registers -- so a 2.5 job can never resolve a 2.3 file or vice
versa. The text encoder has no comfy file in the 2.3 split set; it resolves
from hub repos by a quality-ordered candidate list (higher-precision sources
first, gated repos last) unless an explicit ``te_name_or_path`` is set.
"""

import os
from typing import Optional, Tuple

from toolkit.models.v2.diffusion_models.ltx2 import (
    LTX2TextConnectors,
    LTX2VideoTransformer3DModel,
)
from toolkit.models.v2.resolver import (
    find_file_recursive,
    resolve_comfy_file,
    resolve_named_file,
)
from toolkit.models.v2.vae.ltx2 import LTX2AudioVAE, LTX2VideoVAE
from toolkit.paths import MODELS_PATH

# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

# the upstream mono checkpoint (every component under comfy prefixes)
LTX23_REPO = "Lightricks/LTX-2.3"
LTX23_MONO_FILE = "ltx-2.3-22b-dev.safetensors"

# comfy split files (transformer-only + separate projection + bare-key vaes).
# Only the shipped int8 convrot transformer is registered as a candidate: the
# bf16 split variant stays reachable exclusively via the explicit dit_path
# model_kwargs override -- registering it as a candidate would make a
# quantize-off run rank the clean file first and silently download a ~44GB
# file the box already has a trainable int8 twin of (the mono checkpoint is
# the bf16 source for quantize-on runs).
LTX23_COMFY_REPO = "Kijai/LTX2.3_comfy"
LTX23_DIT_FILE_INT8 = "diffusion_models/ltx-2.3-22b-dev_transformer_only_int8_convrot.safetensors"
LTX23_TEXT_PROJECTION_FILE = "text_encoders/ltx-2.3_text_projection_bf16.safetensors"
LTX23_VIDEO_VAE_FILE = "vae/LTX23_video_vae_bf16.safetensors"
LTX23_AUDIO_VAE_FILE = "vae/LTX23_audio_vae_bf16.safetensors"

# comfy prefixes inside the mono checkpoint (and inside the Kijai dit /
# audio-vae files, which reuse the same layout)
dit_prefix = "model.diffusion_model."
vae_prefix = "vae."
audio_vae_prefix = "audio_vae."
vocoder_prefix = "vocoder."
text_projection_prefix = "text_embedding_projection."

# the vae_id both the registry entry and the model declare
LTX2_VAE_ID = "ltx2"

# the text encoder rides in its own repos (no 2.3 comfy TE file exists in the
# split set). Quality order for the default resolution: full-precision
# instruct weights first -- a q4-QAT source feeds the connectors low-
# precision hidden states for no reason -- and gated repos last, so a missing
# token never blocks an open source. Local files always win regardless of
# order: any te file that exists on disk is used as-is. Each entry is
# (kind, source): "repo" loads through Gemma3TextEncoder.load_model (handles
# shards), "file" is a single comfy gemma3 safetensors for the streamed route.
LTX23_TE_CANDIDATES = (
    ("repo", "unsloth/gemma-3-12b-it"),  # ungated bf16 mirror
    ("repo", "google/gemma-3-12b-it"),  # gated bf16 original
    ("repo", "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"),  # upstream default (q4 QAT)
)
# upstream's 2.3 default, the explicit fallback when nothing better resolves
LTX23_TE_UPSTREAM_DEFAULT = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"

# tokenizer source for the comfy-gemma3 route: the tokenizer is identical
# across all of these (same vocab, same tokenizer.json); this repo is ungated
LTX23_TOKENIZER_SOURCE = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"


def find_local_te_file(filename: str) -> Optional[str]:
    """Any te file already on disk wins, whatever its precision: recursive
    search under the text_encoders category, then the models root."""
    for cat in ("text_encoders", ""):
        root = os.path.join(MODELS_PATH, cat) if cat else MODELS_PATH
        found = find_file_recursive(root, filename)
        if found:
            return found
    return None


class LTX23VideoTransformer(LTX2VideoTransformer3DModel):
    """2.3 candidate registration for the transformer-only comfy files."""

    aitk_subfolder = "diffusion_models"
    aitk_comfy_repo = LTX23_COMFY_REPO
    aitk_comfy_weight_names = {LTX23_COMFY_REPO: [LTX23_DIT_FILE_INT8]}


class LTX23VideoVAE(LTX2VideoVAE):
    aitk_subfolder = "vae"
    aitk_comfy_repo = LTX23_COMFY_REPO
    aitk_comfy_weight_names = {LTX23_COMFY_REPO: [LTX23_VIDEO_VAE_FILE]}


class LTX23AudioVAE(LTX2AudioVAE):
    aitk_subfolder = "vae"
    aitk_comfy_repo = LTX23_COMFY_REPO
    # the file bundles the BWE vocoder under "vocoder." keys
    aitk_comfy_weight_names = {LTX23_COMFY_REPO: [LTX23_AUDIO_VAE_FILE]}


def _resolve_split_file(
    model, component_cls, component: str, override_key: str
) -> Optional[str]:
    """Resolve one component through the shared comfy door (the class's
    candidate table, the per-component model_kwargs override, local roots,
    MODELS_PATH layout, hub)."""
    return component_cls.resolve_comfy_weights(
        model.model_config.name_or_path,
        subfolder="",
        hf_token=os.getenv("HF_TOKEN"),
        status_fn=model.print_and_status_update,
        component=component,
        model_kwargs=model.model_config.model_kwargs,
        override_path=model.model_config.model_kwargs.get(override_key, None),
        extra_roots=(
            [model.model_config.name_or_path]
            if model.model_config.name_or_path
            and os.path.isdir(model.model_config.name_or_path)
            else None
        ),
    )


def is_mono_checkpoint(model) -> bool:
    """True when name_or_path points at the upstream mono checkpoint (a
    .safetensors file), which carries every component under comfy prefixes."""
    name_or_path = model.model_config.name_or_path
    return bool(name_or_path) and name_or_path.endswith(".safetensors")


def resolve_component_source(model, component: str) -> Tuple[str, str]:
    """Resolve a 2.3 component to (path, key_prefix).

    Mono checkpoint: every component streams from the one file under its
    comfy prefix. Split files: the dit and the audio vae carry prefixes, the
    video vae and the text projection are bare-key. The mono route also
    honors an explicit ``<component>_path`` override (an explicit file always
    wins, mono or split -- its layout is detected from its own keys)."""
    overrides = model.model_config.model_kwargs or {}
    override = overrides.get(f"{component}_path", None)
    if override is not None:
        path = resolve_named_file(
            override, component=f"ltx2.3 {component}", hf_token=os.getenv("HF_TOKEN")
        )
        return path, _detect_prefix(path, component)

    if is_mono_checkpoint(model):
        path = resolve_named_file(
            model.model_config.name_or_path,
            component=f"ltx2.3 {component}",
            hf_token=os.getenv("HF_TOKEN"),
        )
        return path, {
            "dit": dit_prefix,
            "video_vae": vae_prefix,
            "audio_vae": audio_vae_prefix,
            "text_projection": text_projection_prefix,
        }[component]

    component_cls, override_key = {
        "dit": (LTX23VideoTransformer, "dit_path"),
        "video_vae": (LTX23VideoVAE, "video_vae_path"),
        "audio_vae": (LTX23AudioVAE, "audio_vae_path"),
    }.get(component, (None, None))
    if component_cls is not None:
        path = _resolve_split_file(
            model, component_cls, component, override_key
        )
    elif component == "text_projection":
        # no component class: the known split file resolves through the
        # comfy door directly (override: text_projection_path)
        path = resolve_comfy_file(
            LTX23_TEXT_PROJECTION_FILE,
            LTX23_COMFY_REPO,
            override_path=(model.model_config.model_kwargs or {}).get(
                "text_projection_path", None
            ),
            extra_roots=(
                [model.model_config.name_or_path]
                if model.model_config.name_or_path
                and os.path.isdir(model.model_config.name_or_path)
                else None
            ),
            hf_token=os.getenv("HF_TOKEN"),
            status_fn=model.print_and_status_update,
        )
    else:
        path = None
    if path is None:
        raise FileNotFoundError(
            f"LTX-2.3: no file resolved for component '{component}' under "
            f"name_or_path '{model.model_config.name_or_path}'. Point "
            f"'{component}_path' at an existing file, set name_or_path to the "
            f"mono checkpoint, or set it to a repo id with 2.3 comfy files."
        )
    return path, _detect_prefix(path, component)


def _detect_prefix(path: str, component: str) -> str:
    """Which comfy prefix (if any) the resolved file uses for this component.
    Header-only: WeightSource.keys() reads nothing but the header."""
    from toolkit.util.weight_source import WeightSource

    keys = WeightSource.open(path).keys()
    for prefix in (
        dit_prefix,
        vae_prefix,
        audio_vae_prefix,
        vocoder_prefix,
        text_projection_prefix,
    ):
        if any(k.startswith(prefix) for k in keys):
            return prefix
    return ""
