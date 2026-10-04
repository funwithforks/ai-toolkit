"""MiniMax-H3 diffusion transformer (33B), weight-compatible with the original
``MiniMaxAI/MiniMax-H3`` checkpoint keys (``video_patch_proj``, ``blocks.N.*``,
``final_layer.*``, ...).

MiniMax-H3 runs one stack of blocks over a single packed 1-D sequence holding
``[text | keyframe-condition video rows | audio rows | target video rows]``.
Full self-attention, no cross-attention, no per-modality weights: modality-
specific behavior comes only from the two input patch projections, the per-row
AdaLN modality tag (0 = video, 1 = text, 2 = audio) and the two output heads.

Deviations from the reference implementations (both are pure generalizations —
with batch size 1 and no padding the math is identical):

  - every structural input (``position_ids``, ``token_tags``, ``row_timesteps``)
    carries a batch axis, so items with different prompt lengths can share a
    training batch (pad rows are tagged ``-1`` and masked out of attention)
  - timesteps are passed per row and deduplicated internally instead of the
    caller supplying ``timestep`` + ``timestep_indices``

Mixed precision follows the shipped checkpoint: ``video_patch_proj``,
``audio_patch_proj``, ``time_embedder``, ``final_layer.video_out/audio_out``
and ``rope.inv_freq`` are float32 islands, everything else runs bf16. Inputs
are aligned to each projection's own parameter dtype at the call site.

Timesteps are consumed unscaled in [0, 1] with t = 1 - sigma (t=1 means clean),
and both heads predict the data-ward velocity ``clean - noise``.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from toolkit.models.v2._mixin import OstrisModelMixin

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .kernels import fused_norm as _fused_norm, swiglu_mul as _fused_swiglu_mul
from .adaln_fusion import gated_residual, modulated_norm

MODALITY_NUM = 3  # 0 = video, 1 = text, 2 = audio; -1 marks padding rows


@dataclass
class MiniMaxH3TransformerParams:
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128  # heads * head_dim = 7168 > hidden_size
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    # VSA-trained checkpoints (FastVideo FastH3) carry a per-token gate for
    # the coarse compression branch: blocks.N.attn.to_gate_compress
    gate_compress: bool = False
    # "pruned" checkpoints (e.g. Comfy-Org *_pruned_*) replace the timestep
    # MLP with a small lookup table: ``adaln_t_table`` of shape
    # (adaln_t_table_size, time_embed_dim) sampled by linear interpolation at
    # t * (size - 1), consumed by the AdaLN projections WITHOUT the SiLU.
    # These checkpoints also shrink time_embed_dim (8 in the released files)
    # and add a bias to the block AdaLN linears, which the original weights
    # lack (the final layer's AdaLN carries a bias in both variants).
    adaln_t_table_size: Optional[int] = None

    @property
    def adaln_apply_silu(self) -> bool:
        return self.adaln_t_table_size is None

    @property
    def adaln_bias(self) -> bool:
        return self.adaln_t_table_size is not None


class MiniMaxH3Rope(nn.Module):
    """3-axis rotary embedding over the packed (t, h, w) coordinates.

    One shared ``inv_freq`` of 16 frequencies per axis; the three 16-angle
    blocks concatenate to 48 and duplicate to 96, so the first 96 of the 128
    head channels rotate (rotate-half convention) and the last 32 pass through.
    """

    def __init__(self, inv_freq_len: int = 16, theta: float = 10000.0):
        super().__init__()
        dim = 2 * inv_freq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # present in the checkpoint, so persistent
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, position_ids: torch.Tensor):
        """position_ids (B, S, 3) -> cos, sin each (B, S, 96), float32."""
        # compute on the input's device: the frequency buffer may be left
        # CPU-resident by layer offloading / low_vram loads
        position_ids = position_ids.to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(position_ids.device)
        freqs = position_ids.unsqueeze(-1) * inv_freq.view(1, 1, 1, -1)
        # (B, S, 3, 16) -> (B, S, 48) in (t, h, w) axis order -> duplicate to 96
        freqs = freqs.flatten(2, 3)
        freqs = torch.cat([freqs, freqs], dim=-1)
        return freqs.cos(), freqs.sin()


_FROST_SDPA = False


def set_frost_sdpa(flag: bool) -> None:
    global _FROST_SDPA
    _FROST_SDPA = bool(flag)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x (B, S, H, D); cos/sin (B, S, rot) rotate the leading ``rot`` channels."""
    # fused Triton path when available; returns None (and the env kill switch
    # lives in rope_fusion) to fall through to the torch ops below
    from .rope_fusion import fused_rope

    out = fused_rope(x, cos, sin)
    if out is not None:
        return out
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    cos = cos.to(x.dtype).unsqueeze(2)
    sin = sin.to(x.dtype).unsqueeze(2)
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return torch.cat([x_rot * cos + rotated * sin, x_pass], dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    """Sinusoidal embedding (cos before sin, unscaled t in [0, 1]) -> MLP.

    A float32 island in the checkpoint; the output stays float32 so every
    block's AdaLN applies its SiLU at full precision before casting down.
    """

    def __init__(self, freq_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True)
        self.proj_out = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        emb = emb.to(self.proj_in.weight.dtype)
        return self.proj_out(F.silu(self.proj_in(emb)))


class MiniMaxH3Attention(nn.Module):
    """Fused-QKV self-attention with per-head RMSNorm on q/k and partial RoPE."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        head_dim: int,
        qk_norm_eps: float,
        gate_compress: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)
        self.to_gate_compress = (
            nn.Linear(hidden, inner, bias=False) if gate_compress else None
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, S, hidden)
        rotary_emb=None,  # (cos, sin) each (B, S, rot) or None
        attn_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, S) bool, True = attend
        varlen_ctx=None,  # (idx, cu, max_len) packed live rows, or None
        vsa=None,  # H3VSAContext or None (dense)
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(b, s, self.heads, self.head_dim)
        k = k.view(b, s, self.heads, self.head_dim)
        v = v.view(b, s, self.heads, self.head_dim)

        q = _fused_norm(q, self.q_norm)
        k = _fused_norm(k, self.k_norm)
        if rotary_emb is not None:
            q = apply_rotary_emb(q, *rotary_emb)
            k = apply_rotary_emb(k, *rotary_emb)

        if vsa is not None and self.to_gate_compress is not None:
            from .vsa import vsa_attention

            gate = self.to_gate_compress(x).view(b, s, self.heads, self.head_dim)
            out = vsa_attention(q, k, v, gate, vsa)
            return self.out_proj(out.reshape(b, s, -1))

        if varlen_ctx is not None:
            # packed live rows -> flash varlen; exact key-mask semantics on
            # any pad layout (the (B,1,1,S) mask form forces torch onto the
            # sm80 mem-efficient kernels at ~3x cost on these shapes). Pad
            # outputs are left zero; the final heads select the live rows,
            # so pad values are discarded everywhere downstream.
            idx, cu, max_len = varlen_ctx
            from .flash_varlen import flash_varlen_sdpa

            qf = q.reshape(b * s, self.heads, self.head_dim)[idx]
            kf = k.reshape(b * s, self.heads, self.head_dim)[idx]
            vf = v.reshape(b * s, self.heads, self.head_dim)[idx]
            of = flash_varlen_sdpa(qf, kf, vf, cu, max_len, self.head_dim**-0.5)
            out = q.new_zeros(b * s, self.heads * self.head_dim)
            out = out.index_copy(0, idx, of.reshape(-1, self.heads * self.head_dim))
            return self.out_proj(out.view(b, s, -1))

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # torch SDPA is the default (see 74dbca1): on datasets with
        # mixed resolutions cudnn plan compiles fire on every new sequence
        # length.
        # model_kwargs {frost_sdpa: true} opts the dense path into the
        # sm120 kernels -- intended for jobs with uniform dataset shapes,
        # where the plan set is finite.
        out = None
        if attn_mask is None and _FROST_SDPA:
            from toolkit.attention.frost import fused_sdpa

            out = fused_sdpa(q, k, v, self.head_dim**-0.5)
        if out is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.out_proj(out)


class MiniMaxH3Mlp(nn.Module):
    """SwiGLU: fc1 packs [gate | up]; silu(gate) * up -> fc2. No biases."""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(_fused_swiglu_mul(gate, up))


class MiniMaxH3AdalnProj(nn.Module):
    """Timestep embedding -> per-(timestep, modality) modulation parameters.

    (M, time_embed_dim) -> ``expand`` tensors of (M * modalities, hidden); row
    layout ``[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]``, addressed by
    ``timestep_index * MODALITY_NUM + tag``. The whole projection computes in
    float32 (pruned checkpoints store these linears fp16, whose 65504 ceiling
    overflows here); the weights stay at their stored dtype and are only
    upcast for the matmul, so whole-model casts can't undo this. Outputs are
    float32; the use sites cast them to the block dtype.
    """

    def __init__(
        self,
        t_dim: int,
        hidden: int,
        expand: int,
        modalities: int,
        apply_silu: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=bias)

    def forward(self, temb: torch.Tensor):
        if self.apply_silu:
            temb = F.silu(temb)
        # follow temb's device: the raw params may be CPU-resident under layer
        # offloading / low_vram, and F.linear bypasses the paging hooks
        weight = self.linear.weight.to(device=temb.device, dtype=torch.float32)
        bias = self.linear.bias
        if bias is not None:
            bias = bias.to(device=temb.device, dtype=torch.float32)
        x = F.linear(temb.float(), weight, bias)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


class MiniMaxH3RefinerBlock(nn.Module):
    """Plain pre-norm block over the projected text stream. No AdaLN, no RoPE."""

    def __init__(self, p: MiniMaxH3TransformerParams):
        super().__init__()
        self.norm1 = nn.RMSNorm(p.hidden_size, eps=p.norm_eps)
        self.norm2 = nn.RMSNorm(p.hidden_size, eps=p.norm_eps)
        self.attn = MiniMaxH3Attention(
            p.hidden_size, p.num_attention_heads, p.attention_head_dim, p.qk_norm_eps
        )
        self.mlp = MiniMaxH3Mlp(p.hidden_size, p.ffn_hidden_size)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.attn(_fused_norm(x, self.norm1), attn_mask=attn_mask)
        x = x + self.mlp(_fused_norm(x, self.norm2))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, p: MiniMaxH3TransformerParams):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3RefinerBlock(p) for _ in range(p.token_refiner_num_layers)]
        )
        self.final_norm = nn.RMSNorm(p.hidden_size, eps=p.final_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask)
        return _fused_norm(x, self.final_norm)


class MiniMaxH3Block(nn.Module):
    """Pre-norm attention + SwiGLU MLP, each modulated by AdaLN parameters
    gathered per row from the (timestep, modality) table."""

    def __init__(self, p: MiniMaxH3TransformerParams):
        super().__init__()
        self.norm1 = nn.RMSNorm(p.hidden_size, eps=p.norm_eps)
        self.norm2 = nn.RMSNorm(p.hidden_size, eps=p.norm_eps)
        self.attn = MiniMaxH3Attention(
            p.hidden_size,
            p.num_attention_heads,
            p.attention_head_dim,
            p.qk_norm_eps,
            gate_compress=p.gate_compress,
        )
        self.mlp = MiniMaxH3Mlp(p.hidden_size, p.ffn_hidden_size)
        self.adaln_proj = MiniMaxH3AdalnProj(
            p.time_embed_dim,
            p.hidden_size,
            expand=6,
            modalities=MODALITY_NUM,
            apply_silu=p.adaln_apply_silu,
            bias=p.adaln_bias,
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, S, hidden)
        temb: torch.Tensor,  # (M, time_embed_dim) float32
        adaln_indices: torch.Tensor,  # (B, S) long into the (M * 3) table
        rotary_emb,  # (cos, sin)
        attn_mask: Optional[torch.Tensor] = None,
        varlen_ctx=None,  # (idx, cu, max_len) packed live rows, or None
        vsa=None,  # H3VSAContext or None (dense)
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_proj(temb)
        )

        # one fused kernel per modulation point: the eager form gathered six
        # (B, S, hidden) float32 table rows and cast them per block, which
        # dominated the bs4 elementwise traffic (see adaln_fusion.py)
        h = modulated_norm(
            _fused_norm(x, self.norm1), scale_msa, shift_msa, adaln_indices
        )
        x = gated_residual(
            x,
            gate_msa,
            self.attn(h, rotary_emb, attn_mask, varlen_ctx, vsa),
            adaln_indices,
        )

        h = modulated_norm(
            _fused_norm(x, self.norm2), scale_mlp, shift_mlp, adaln_indices
        )
        x = gated_residual(x, gate_mlp, self.mlp(h), adaln_indices)
        return x


class MiniMaxH3FinalLayer(nn.Module):
    """Shared shift/scale-modulated RMSNorm + the two per-modality heads.

    Both heads run over every row (matching the reference); the caller selects
    each modality's rows from the results. Heads are float32 islands.
    """

    def __init__(self, p: MiniMaxH3TransformerParams):
        super().__init__()
        video_patch_dim = (
            p.latents_dim * p.patch_size[0] * p.patch_size[1] * p.patch_size[2]
        )
        self.norm = nn.RMSNorm(p.hidden_size, eps=p.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            p.time_embed_dim,
            p.hidden_size,
            expand=2,
            modalities=1,
            apply_silu=p.adaln_apply_silu,
            bias=True,
        )
        self.video_out = nn.Linear(p.hidden_size, video_patch_dim, bias=True)
        self.audio_out = nn.Linear(p.hidden_size, p.audio_latents_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,  # (B, S, hidden)
        temb: torch.Tensor,  # (M, time_embed_dim)
        timestep_indices: torch.Tensor,  # (B, S) long into the M rows
    ):
        shift, scale = self.adaln_proj(temb)
        dt = x.dtype
        h = self.norm(x) * (1.0 + scale[timestep_indices].to(dt)) + shift[
            timestep_indices
        ].to(dt)
        h = h.to(self.video_out.weight.dtype)
        return self.video_out(h), self.audio_out(h)


class MiniMaxH3Transformer(nn.Module, OstrisModelMixin):
    # comfy checkpoints carry a deliberate bf16/fp16/fp32 mix
    aitk_cast_on_load = False

    # every H3 DiT variant ships in the Comfy-Org repack; which one a job
    # uses is chosen by the model's partition filter (see
    # MinimaxH3Model.select_comfy_candidates), not a separate lookup
    aitk_comfy_repo = "Comfy-Org/MiniMax-H3"
    aitk_comfy_weight_names = {
        "Comfy-Org/MiniMax-H3": [
            "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
        ],
    }

    @classmethod
    def aitk_config_from_state_dict(cls, state_dict):
        params = MiniMaxH3TransformerParams()
        table = state_dict.get("adaln_t_table", None)
        if table is not None:
            # pruned checkpoint: factored timestep table instead of the MLP
            params.adaln_t_table_size = table.shape[0]
            params.time_embed_dim = table.shape[1]
        params.gate_compress = "blocks.0.attn.to_gate_compress.weight" in state_dict
        return params

    @classmethod
    def aitk_from_config(cls, config):
        from accelerate import init_empty_weights

        with init_empty_weights(include_buffers=False):
            return cls(config)

    @classmethod
    def get_transformer_block_names(cls):
        return ["blocks"]

    def __init__(self, params: Optional[MiniMaxH3TransformerParams] = None):
        super().__init__()
        if params is None:
            params = MiniMaxH3TransformerParams()
        self.params = params
        p = params

        video_patch_dim = (
            p.latents_dim * p.patch_size[0] * p.patch_size[1] * p.patch_size[2]
        )
        self.video_patch_proj = nn.Linear(video_patch_dim, p.hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(p.audio_latents_dim, p.hidden_size, bias=True)
        self.condition_proj = nn.Linear(p.text_dim, p.hidden_size, bias=True)

        if p.adaln_t_table_size is not None:
            # pruned checkpoints: the timestep MLP is replaced by a lookup
            # table sampled with linear interpolation at t * (size - 1)
            self.time_embedder = None
            self.register_buffer(
                "adaln_t_table",
                torch.zeros(p.adaln_t_table_size, p.time_embed_dim),
                persistent=True,
            )
        else:
            self.time_embedder = MiniMaxH3TimeEmbedder(
                p.timestep_input_dim, p.time_embed_hidden_size, p.time_embed_dim
            )
        self.rope = MiniMaxH3Rope(p.rope_inv_freq_len, p.rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(p)
        self.blocks = nn.ModuleList([MiniMaxH3Block(p) for _ in range(p.num_layers)])
        self.final_layer = MiniMaxH3FinalLayer(p)

        self.gradient_checkpointing = False
        # None = dense attention. Set by the FastH3 model wrapper; only takes
        # effect on gate_compress checkpoints when the caller passes the grid.
        self.vsa_sparsity: Optional[float] = None

    # float32 islands of the shipped checkpoint; used by the loader to keep
    # these keys at full precision when the rest is cast to bf16
    FP32_KEY_PREFIXES = (
        "video_patch_proj",
        "audio_patch_proj",
        "time_embedder",
        "final_layer.video_out",
        "final_layer.audio_out",
        "rope",
    )

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable
        self.token_refiner.gradient_checkpointing = enable

    def disable_gradient_checkpointing(self):
        self.enable_gradient_checkpointing(False)

    @property
    def device(self):
        return self.condition_proj.weight.device

    @property
    def dtype(self):
        # working dtype of the block stack (norm weights are never quantized)
        return self.token_refiner.final_norm.weight.dtype

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """(M,) float32 in [0, 1] -> (M, time_embed_dim) float32."""
        if self.time_embedder is not None:
            return self.time_embedder(t)
        table = self.adaln_t_table.float()
        pos = t.clamp(0.0, 1.0) * (table.shape[0] - 1)
        # the table may stay CPU-resident under offloading while timesteps
        # arrive on cuda: interpolate on the table's device, return on t's
        pos = pos.to(table.device)
        lo = pos.floor().long()
        hi = (lo + 1).clamp(max=table.shape[0] - 1)
        frac = (pos - lo.float()).unsqueeze(1)
        return (table[lo] * (1.0 - frac) + table[hi] * frac).to(t.device)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, Nv, 96) patchified video rows (cond + target)
        audio_hidden_states: torch.Tensor,  # (B, Na, 32) audio rows
        encoder_hidden_states: torch.Tensor,  # (B, L, 5120) text conditioning
        row_timesteps: torch.Tensor,  # (B, S) float in [0, 1], t = 1 - sigma
        token_tags: torch.Tensor,  # (B, S) long: 0 video, 1 text, 2 audio, -1 pad
        position_ids: torch.Tensor,  # (B, S, 3) float (t, h, w) rotary coords
        video_indices: torch.Tensor,  # (Nv,) long positions of video rows in the pack
        audio_indices: torch.Tensor,  # (Na,) long
        text_indices: torch.Tensor,  # (L,) long
        vsa_video_grid: Optional[
            Tuple[int, int, int]
        ] = None,  # target-video token grid (t, h, w)
    ):
        """Returns (video_out (B, Nv, 96), audio_out (B, Na, 32)) — the
        data-ward velocity ``clean - noise`` for every row, in input order.
        Conditioning rows come back unmasked; discarding them is the caller's
        job."""
        batch_size, seq_len = token_tags.shape

        rotary_emb = self.rope(position_ids)

        video_embeds = self.video_patch_proj(
            hidden_states.to(self.video_patch_proj.weight.dtype)
        )
        audio_embeds = self.audio_patch_proj(
            audio_hidden_states.to(self.audio_patch_proj.weight.dtype)
        )
        text_embeds = self.condition_proj(encoder_hidden_states.to(self.dtype))

        # pad rows never act as attention keys; as queries they see everything
        # (their outputs are discarded), which keeps SDPA rows finite
        attn_mask = None
        text_attn_mask = None
        varlen_ctx = None
        is_pad = token_tags < 0
        if bool(is_pad.any()):
            live = ~is_pad
            text_attn_mask = live[:, text_indices][:, None, None, :]
            from .flash_varlen import varlen_ok

            if varlen_ok():
                # pack live rows (b-major) into a varlen stream; works for
                # interleaved pads, which a length-based ragged expression
                # cannot express. Sync is free here: is_pad.any() above
                # already drained the pipeline once per model forward.
                idx = live.reshape(-1).nonzero(as_tuple=True)[0]
                n_live = live.sum(dim=1)
                cu = torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=live.device
                )
                cu[1:] = n_live.cumsum(0).to(torch.int32)
                varlen_ctx = (idx, cu, int(n_live.max()))
            else:
                attn_mask = live[:, None, None, :]

        text_embeds = self.token_refiner(text_embeds, text_attn_mask)

        x = text_embeds.new_zeros((batch_size, seq_len, text_embeds.shape[-1]))
        x = x.index_copy(1, text_indices, text_embeds)
        x = x.index_copy(1, video_indices, video_embeds.to(x.dtype))
        x = x.index_copy(1, audio_indices, audio_embeds.to(x.dtype))

        # one timestep embedding per distinct noise level across the batch
        unique_t, inverse = torch.unique(
            row_timesteps.to(torch.float32), sorted=True, return_inverse=True
        )
        temb = self._time_embedding(unique_t)
        adaln_indices = inverse * MODALITY_NUM + token_tags.clamp(min=0)

        vsa_ctx = None
        if (
            self.vsa_sparsity is not None
            and self.params.gate_compress
            and vsa_video_grid is not None
        ):
            from .vsa import build_vsa_context, vsa_is_available

            # no triton -> warn once and run every block dense instead
            vsa_ctx = (
                None
                if not vsa_is_available()
                else build_vsa_context(
                    seq_len=seq_len,
                    num_text_rows=int(text_indices.shape[0]),
                    video_grid=tuple(int(g) for g in vsa_video_grid),
                    token_tags=token_tags,
                    sparsity=self.vsa_sparsity,
                    device=x.device,
                )
            )

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = checkpoint(
                    block,
                    x,
                    temb,
                    adaln_indices,
                    rotary_emb,
                    attn_mask,
                    varlen_ctx,
                    vsa_ctx,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    temb,
                    adaln_indices,
                    rotary_emb,
                    attn_mask,
                    varlen_ctx,
                    vsa_ctx,
                )

        video_all, audio_all = self.final_layer(x, temb, inverse)
        video_out = video_all.index_select(1, video_indices)
        audio_out = audio_all.index_select(1, audio_indices)
        return video_out, audio_out
