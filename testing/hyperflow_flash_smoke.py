"""HyperFlow H3 Flash load + train-step smoke test.

Exercises the real single-file load path (meta build, comfy_quant attach,
strict state-dict load), the int8 convrot container math against a manual
reconstruction from the file's codes/scales/rotation, a two-time forward +
backward on a synthetic packed batch, the _row_endpoints step math, and the
single-time opt-in guard.

Run: venv/bin/python testing/hyperflow_flash_smoke.py
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch

from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3Transformer,
    MiniMaxH3TransformerParams,
)

CHECKPOINT = (
    "/run/media/computer/240gb pny/huggingface/diffusion_models/"
    "minimax_h3_fl2va_hyperflow_flash_int8_convrot.safetensors"
)
DEVICE = torch.device("cuda")


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(f"smoke test failed at: {name}")


def regular_hadamard_rot(size: int) -> torch.Tensor:
    """Block-diagonal Kronecker power of the 4x4 regular Hadamard /2, the
    rotation the flash converter used (flash int8.py:100-109 semantics)."""
    h4 = torch.tensor(
        [[1.0, 1.0, 1.0, -1.0],
         [1.0, 1.0, -1.0, 1.0],
         [1.0, -1.0, 1.0, 1.0],
         [-1.0, 1.0, 1.0, 1.0]],
    ) / 2.0
    rot = h4
    while rot.shape[0] < size:
        rot = torch.kron(rot, h4)
    assert rot.shape[0] == size
    return rot


def main():
    # ------------------------------------------------------------------
    # 1. real load path (strict): validates every key maps (47 blocks,
    #    lowrank down/up, split q/k/v, endpoint embedder, gate/sigmas buffers)
    # ------------------------------------------------------------------
    print(f"loading {Path(CHECKPOINT).name} ...")
    model = MiniMaxH3Transformer.load_model(
        CHECKPOINT, dtype=torch.bfloat16, aitk_device=DEVICE
    )
    p = model.params
    check("num_layers sniffed to 47", p.num_layers == 47, str(p.num_layers))
    check("hyperflow_gate sniffed", p.hyperflow_gate == 0.25, str(p.hyperflow_gate))
    check("sigma grid sniffed", p.hyperflow_sigmas is not None and len(p.hyperflow_sigmas) == 7,
          str(p.hyperflow_sigmas))
    check("102 low-rank linears sniffed", p.lowrank_ranks is not None and len(p.lowrank_ranks) == 102,
          str(len(p.lowrank_ranks or {})))
    check("time_embed_dim sniffed to 16", p.time_embed_dim == 16, str(p.time_embed_dim))
    check("endpoint embedder built", hasattr(model, "endpoint_time_embedder"))
    n_split = sum(
        1 for name, m in model.named_modules()
        if name.startswith("blocks.") and name.endswith(".attn")
        and getattr(m, "qkv_proj", None) is None
    )
    check("17 split-qkv attention blocks", n_split == 17, str(n_split))

    # ------------------------------------------------------------------
    # 2. int8 convrot weight math: toolkit's effective weight vs manual
    #    reconstruction (qweight * wscale) @ R^T from the file
    # ------------------------------------------------------------------
    from toolkit.util.ostris_quant import OstrisLinear

    qkv = model.blocks[0].attn.qkv_proj
    check("blocks.0 qkv is OstrisLinear (int8 convrot)", isinstance(qkv, OstrisLinear))
    check("convrot group size 256", getattr(qkv, "cr8_rot_size", 0) == 256)
    weight = qkv.dequantize_weight().float()
    qdata = qkv.cr8_qdata.float().cpu()
    scales = qkv.cr8_scales.view(torch.float32).reshape(-1).cpu()
    rot = regular_hadamard_rot(256)
    in_features = qdata.shape[1]
    rot_full = torch.zeros(in_features, in_features)
    for i in range(0, in_features, 256):
        rot_full[i:i + 256, i:i + 256] = rot
    w_manual = (qdata * scales[:, None]) @ rot_full.T
    row_absmax = w_manual.abs().amax(dim=1).clamp(min=1e-6)
    rel_err = ((weight.cpu() - w_manual).abs().amax(dim=1) / row_absmax).max()
    check("int8 weight matches manual reconstruction", rel_err < 0.02, f"max row rel err {rel_err:.4f}")

    # ------------------------------------------------------------------
    # 3. two-time forward + backward on a synthetic packed batch
    # ------------------------------------------------------------------
    from extensions_built_in.diffusion_models.minimax_h3.src.packing import (
        AUDIO_SIGMA_SHIFT,
        VIDEO_SIGMA_SHIFT,
        remap_sigma,
    )

    grid = torch.tensor(p.hyperflow_sigmas, dtype=torch.float32)
    sigma_i, sigma_next = grid[2], grid[3]  # a middle deployment step
    sigma_a, sigma_a_next = (
        remap_sigma(sigma_i, VIDEO_SIGMA_SHIFT, AUDIO_SIGMA_SHIFT),
        remap_sigma(sigma_next, VIDEO_SIGMA_SHIFT, AUDIO_SIGMA_SHIFT),
    )
    t_v, r_v = 1.0 - sigma_i, 1.0 - sigma_next
    t_a, r_a = 1.0 - sigma_a, 1.0 - sigma_a_next

    n_text, n_video, n_audio = 8, 4, 2
    seq = n_text + n_video + n_audio
    token_tags = torch.full((1, seq), -1, dtype=torch.long)
    token_tags[0, :n_text] = 1
    token_tags[0, n_text:n_text + n_video] = 0
    token_tags[0, n_text + n_video:] = 2
    video_indices = torch.arange(n_text, n_text + n_video)
    audio_indices = torch.arange(n_text + n_video, seq)
    text_indices = torch.arange(n_text)
    row_t = torch.full((1, seq), float(t_v))
    row_t[0, audio_indices] = float(t_a)
    row_r = torch.full((1, seq), float(r_v))
    row_r[0, audio_indices] = float(r_a)
    position_ids = torch.rand(1, seq, 3) * 4

    kwargs = dict(
        hidden_states=torch.randn(1, n_video, 96, device=DEVICE),
        audio_hidden_states=torch.randn(1, n_audio, 32, device=DEVICE),
        encoder_hidden_states=torch.randn(1, n_text, 5120, device=DEVICE, dtype=torch.bfloat16),
        row_timesteps=row_t.to(DEVICE),
        row_endpoints=row_r.to(DEVICE),
        token_tags=token_tags.to(DEVICE),
        position_ids=position_ids.to(DEVICE),
        video_indices=video_indices.to(DEVICE),
        audio_indices=audio_indices.to(DEVICE),
        text_indices=text_indices.to(DEVICE),
    )
    video_out, audio_out = model(**kwargs)
    check("video head shape", tuple(video_out.shape) == (1, n_video, 96), str(tuple(video_out.shape)))
    check("audio head shape", tuple(audio_out.shape) == (1, n_audio, 32), str(tuple(audio_out.shape)))
    check("outputs finite", bool(torch.isfinite(video_out).all() and torch.isfinite(audio_out).all()))

    loss = video_out.float().mean() + audio_out.float().mean()
    loss.backward()
    check("lowrank factor gets grad",
          model.blocks[1].attn.q_proj.down.weight.grad is not None
          and bool(torch.isfinite(model.blocks[1].attn.q_proj.down.weight.grad).all()))
    check("endpoint embedder gets grad",
          model.endpoint_time_embedder.proj_out.weight.grad is not None)
    check("int8 block survived backward", model.blocks[0].attn.qkv_proj.cr8_qdata is not None)
    model.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # 4. _row_endpoints step math via a stubbed wrapper
    # ------------------------------------------------------------------
    from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3FlashModel

    wrapper = MinimaxH3FlashModel.__new__(MinimaxH3FlashModel)
    wrapper.model = model
    sigma_v = torch.tensor([sigma_i, grid[0]])
    row_t_b = torch.stack([
        torch.full((seq,), float(t_v)),
        torch.full((seq,), 0.0),
    ])
    row_r_b = wrapper._row_endpoints(
        row_t_b, sigma_v, None,
        video_indices, audio_indices, num_cond=0, num_cond_audio=0,
    )
    check("endpoint column matches grid step",
          torch.allclose(row_r_b[0, video_indices], torch.full((n_video,), float(r_v)), atol=1e-6)
          and torch.allclose(row_r_b[1, video_indices], torch.full((n_video,), 1.0 - grid[1]), atol=1e-6),
          f"row0 r={row_r_b[0, video_indices[0]]:.4f} expected {float(r_v):.4f}")

    # ------------------------------------------------------------------
    # 5. single-time opt-in guard: base params reject row_endpoints
    # ------------------------------------------------------------------
    small = MiniMaxH3TransformerParams(
        hidden_size=64, num_layers=1, num_attention_heads=2, attention_head_dim=32,
        ffn_hidden_size=64, text_dim=32, time_embed_dim=2688, rope_inv_freq_len=4,
    )
    base = MiniMaxH3Transformer(small).to(DEVICE, torch.bfloat16)
    base_kwargs = dict(
        hidden_states=torch.randn(1, 2, 96, device=DEVICE, dtype=torch.bfloat16),
        audio_hidden_states=torch.randn(1, 1, 32, device=DEVICE, dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(1, 3, 32, device=DEVICE, dtype=torch.bfloat16),
        row_timesteps=torch.full((1, 6), 0.5, device=DEVICE),
        token_tags=torch.tensor([[1, 1, 1, 0, 0, 2]], device=DEVICE),
        position_ids=torch.rand(1, 6, 3, device=DEVICE),
        video_indices=torch.tensor([3, 4], device=DEVICE),
        audio_indices=torch.tensor([5], device=DEVICE),
        text_indices=torch.tensor([0, 1, 2], device=DEVICE),
    )
    v_out, a_out = base(**base_kwargs)
    check("single-time forward unchanged", tuple(v_out.shape) == (1, 2, 96))
    try:
        base(**base_kwargs, row_endpoints=torch.full((1, 6), 0.25, device=DEVICE))
        check("single-time rejects row_endpoints", False)
    except ValueError:
        check("single-time rejects row_endpoints", True)

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
