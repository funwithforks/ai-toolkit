"""ComfyUI Wan-layout VAE key mapping.

ComfyUI packs the wan-family video VAEs (Qwen-Image, Qwen-Image-2.1) in the
wan-native module layout (`encoder.downsamples.*`, root `conv1`/`conv2`,
`middle.*` sequentials) with a size-1 temporal axis on every conv, while the
diffusers models expect `down_blocks`/`up_blocks`, `quant_conv`, and 4-dim
conv weights. These are pure name/shape mappers — no model code — so any
OstrisModelMixin VAE whose checkpoint ships in that layout can reuse them via
convert_state_dict_on_load.
"""

# Comfy `<block>.residual.<n>` / `shortcut` -> the diffusers resnet submodule.
_COMFY_RESNET_PARTS = {
    "residual.0": "norm1",
    "residual.2": "conv1",
    "residual.3": "norm2",
    "residual.6": "conv2",
    "shortcut": "conv_shortcut",
}
# Comfy's `middle` Sequential is resnet, attention, resnet.
_COMFY_MID_PARTS = {"0": "resnets.0", "1": "attentions.0", "2": "resnets.1"}


def _comfy_resnet_suffix(inner: list) -> list:
    """`residual.<n>.*` / `shortcut.*` -> the diffusers resnet submodule path."""
    consumed = 2 if inner[0] == "residual" else 1
    return [_COMFY_RESNET_PARTS[".".join(inner[:consumed])]] + inner[consumed:]


def comfy_wan_vae_key(key: str) -> str:
    """One ComfyUI VAE parameter name -> its diffusers name."""
    parts = key.split(".")
    # the quant convs sit at the checkpoint root
    if parts[0] == "conv1":
        return ".".join(["quant_conv"] + parts[1:])
    if parts[0] == "conv2":
        return ".".join(["post_quant_conv"] + parts[1:])

    side, rest = parts[0], parts[1:]
    if rest[0] == "conv1":
        return ".".join([side, "conv_in"] + rest[1:])
    if rest[0] == "head":
        # head.0 is the output norm and head.2 the output conv (head.1 is SiLU)
        tail = "norm_out" if rest[1] == "0" else "conv_out"
        return ".".join([side, tail] + rest[2:])
    if rest[0] == "middle":
        part, inner = _COMFY_MID_PARTS[rest[1]], rest[2:]
        if part.startswith("resnets"):
            inner = _comfy_resnet_suffix(inner)
        return ".".join([side, "mid_block", part] + inner)

    # Two layouts exist across the wan-family repacks:
    #   nested (Qwen-Image-2.1): encoder.downsamples.<i>.downsamples.<j>.*
    #   flat (Qwen-Image):       encoder.downsamples.<i>.residual.<n>.*  (and
    #                            the resample/time_conv stage tails)
    # Distinguish by whether a second group token follows the stage index.
    block = "down_blocks" if side == "encoder" else "up_blocks"
    sampler = "downsampler" if side == "encoder" else "upsampler"
    if rest[0] == "downsamples" or rest[0] == "upsamples":
        if len(rest) > 2 and rest[2] in ("downsamples", "upsamples"):
            # nested: stage rest[1], inner stage rest[3], payload rest[4:]
            stage, inner_index, inner = rest[1], rest[3], rest[4:]
        else:
            # flat: stage rest[1], payload rest[2:]
            stage, inner_index, inner = rest[1], None, rest[2:]
        if inner_index is None:
            # a flat stage is a single resnet (or a resample/time_conv tail)
            if inner[0] in ("resample", "time_conv"):
                return ".".join([side, block, stage, sampler] + inner)
            return ".".join(
                [side, block, stage, "resnets", "0"] + _comfy_resnet_suffix(inner)
            )
        if inner[0] in ("resample", "time_conv"):
            # the stage's last entry is the resampler, not a resnet
            return ".".join([side, block, stage, sampler] + inner)
        return ".".join(
            [side, block, stage, "resnets", inner_index] + _comfy_resnet_suffix(inner)
        )
    # legacy bare form (no stage grouping)
    stage, inner_index, inner = rest[0], rest[2], rest[3:]
    if inner[0] in ("resample", "time_conv"):
        return ".".join([side, block, stage, sampler] + inner)
    return ".".join(
        [side, block, stage, "resnets", inner_index] + _comfy_resnet_suffix(inner)
    )


def convert_comfy_wan_vae_state_dict(state_dict: dict) -> dict:
    """Comfy Wan-layout VAE state dict -> diffusers layout. Already-diffusers
    state dicts pass through untouched (no root `conv1` marker)."""
    if "conv1.weight" not in state_dict:
        return state_dict
    return {
        comfy_wan_vae_key(key): (value.squeeze(2) if value.ndim == 5 else value)
        for key, value in state_dict.items()
    }
