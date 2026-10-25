"""Acceptance test: Automagic3FFT is BITWISE identical to Automagic3 (v3)
run pinned (min_lr == max_lr == lr) from a fresh start.

Contract: the v3 polarity controller's only output is the per-step lr nudge,
clamped to [min_lr, max_lr]; pinned, that clamp is a constant, so every number
an FFT run sees must match whether the ring exists or not. SR draws come from
the global RNG and the vote consumes none, so seeded fresh runs match
bit-for-bit at every step (bf16 stochastic-rounding branch and fp32 branch,
2D factored and 1D full second-moment branches).

Resume-continuation bitwise identity is deliberately NOT asserted: the v3
resume path stores loaded state tensors BY REFERENCE when no dtype cast
applies, so two optimizers loaded from one in-memory dict share moments and
the first to step mutates the checkpoint the second reads (root-caused
2026-10-25, docs/fullfinetune/moss/findings.md 5f). Fresh-start v3 is
deterministic.

CPU-only. Run:
    PYTHONPATH=. venv/bin/python docs/fullfinetune/moss/test_vote_identity.py
"""

import torch
from torch import nn

from toolkit.optimizers.automagic3 import Automagic3
from toolkit.optimizers.automagic3fft import Automagic3FFT

LR = 1e-5
STEPS = 12


class Tiny(nn.Module):
    """l0 bf16 (stochastic-rounding writeback), l1 fp32 (plain writeback);
    biases cover the 1D/full-second-moment branch."""

    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(48, 32, dtype=torch.bfloat16)
        self.l1 = nn.Linear(32, 1)

    def forward(self, x):
        return self.l1(self.l0(x).float())


def make_model(seed=0):
    torch.manual_seed(seed)
    return Tiny()


def data():
    g = torch.Generator().manual_seed(1234)
    return [torch.randn(8, 48, generator=g) for _ in range(STEPS)]


def run(cls):
    torch.manual_seed(99)  # identical init for both arms
    model = make_model(99)
    params = list(model.parameters())
    if cls is Automagic3:
        opt = cls(params, lr=LR, min_lr=LR, max_lr=LR)
    else:
        opt = cls(params, lr=LR)
    xs = data()
    for s in range(STEPS):
        torch.manual_seed(s)  # identical RNG entry per step on both arms
        model(xs[s].to(torch.bfloat16)).square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return [p.detach().clone() for p in params]


def main():
    w3 = run(Automagic3)
    wf = run(Automagic3FFT)
    for i, (a, b) in enumerate(zip(w3, wf)):
        assert torch.equal(a, b), (
            f"param {i} diverges: max abs diff "
            f"{(a.float() - b.float()).abs().max().item()}"
        )
    print(f"fresh start: bitwise equal ({len(w3)} params x {STEPS} steps)")

    # second run length / different seed data must also match (guards against
    # a coincidental single-run match): rerun with different data ordering.
    torch.manual_seed(7)
    m3, mf = make_model(7), make_model(7)
    p3, pf = list(m3.parameters()), list(mf.parameters())
    o3 = Automagic3(p3, lr=LR, min_lr=LR, max_lr=LR)
    of = Automagic3FFT(pf, lr=LR)
    g = torch.Generator().manual_seed(555)
    for s in range(STEPS):
        x = torch.randn(6, 48, generator=g)
        torch.manual_seed(1000 + s)
        m3(x.to(torch.bfloat16)).square().mean().backward()
        o3.step(); o3.zero_grad(set_to_none=True)
        torch.manual_seed(1000 + s)
        mf(x.to(torch.bfloat16)).square().mean().backward()
        of.step(); of.zero_grad(set_to_none=True)
    assert all(torch.equal(a, b) for a, b in zip(p3, pf)), "second trial diverges"
    print("second trial (different init/data): bitwise equal")

    # FFT state stays tiny: no ring, no per-param lr tensors
    keys = {k for st in of.state.values() for k in st}
    assert "sign_history" not in keys and "lr" not in keys, keys
    print("FFT optimizer state keys:", sorted(keys))
    print("\nALL IDENTITY CHECKS PASSED")


if __name__ == "__main__":
    main()
