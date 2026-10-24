"""MOSS-Image text conditioning: Qwen3.5-9B final-normed last hidden states.

The chat-templated prompt (thinking on, generation prompt appended) goes
through the language tower; the conditioning is the final-normed last hidden
state, stored per item as (L, 4096) in an ``AdvancedPromptEmbeds``. That is
exactly what the encoder emits, so the caption cache stores the encoder output
and the DiT's 2-block text refiner runs per step (it is LoRA-targetable -- the
fibo2 precedent forbids caching a trainable module's output).

Weights: the Comfy-Org/MOSS-Image int8 ConvRot repack of the stock
Qwen3.5-9B (the vendor text_encoder/ is that model with A_log / gated norm in
bf16). Config and tokenizer come from the public Qwen/Qwen3.5-9B repo, except
the chat template: the tokenizer is loaded from the MOSS vendor repo's
tokenizer/ folder so the thinking template is the vendor's.

EDIT TODO (for the moss_image edit pass): upstream's Qwen35VLEncoder (vision
tower kept, lm_head/mtp dropped, patch_vision_patch_embed GEMM swap) + an
AutoImageProcessor from the Qwen repo; see the holder's EDIT TODO header.
"""

from typing import List

import torch

from toolkit.models.v2.text_encoders.qwen35 import Qwen35TextOnlyEncoder

from .transformer import COMFY_REPO, VENDOR_BASE_REPO

QWEN35_9B_REPO = "Qwen/Qwen3.5-9B"
MAX_TEXT_TOKENS = 4096

# the shipped int8 ConvRot repack of the 9B encoder inside the MOSS package
# (the DiT registration convention: int8 only; the bf16 / w4a8 twins stay
# reachable only through a text_encoder_path override)
COMFY_TE_FILE = "text_encoders/qwen3.5_9b_int8_convrot.safetensors"


class MossQwen35Encoder(Qwen35TextOnlyEncoder):
    """Qwen3.5-9B language tower from the MOSS package repo's int8 ConvRot
    repack; lm_head, mtp head and vision tower are dropped on load (the
    text-only conditioning never runs them)."""

    # the comfy file is weights-only; the public Qwen repo supplies the config
    aitk_config_repo = QWEN35_9B_REPO
    aitk_comfy_repo = COMFY_REPO  # same package repo as the DiT candidates
    aitk_comfy_weight_names = {
        VENDOR_BASE_REPO: [COMFY_TE_FILE],
        COMFY_REPO: [COMFY_TE_FILE],
    }


@torch.no_grad()
def encode_text_batch(text_encoder, tokenizer, texts, max_length, device) -> List[torch.Tensor]:
    """Chat-templated prompts -> per-item (L_i, D) final states from ONE forward.
    Right-padded, no attention mask: for the causal model the pads sit after
    each prompt's real tokens, so neither causal attention nor the gated-delta
    recurrence can reach them -- each item's first L_i states equal its
    unpadded forward (the HF mask path only zeroes padded states, which is
    left-padding hygiene, and would also drop attention off its fast path).
    Default arange positions are right for right padding."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        truncation=True,
        max_length=max_length,
    )
    lengths = enc["attention_mask"].sum(1).tolist()
    states = text_encoder(
        input_ids=enc["input_ids"].to(device), use_cache=False
    ).last_hidden_state
    return [states[i, :n] for i, n in enumerate(lengths)]
