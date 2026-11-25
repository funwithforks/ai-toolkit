"""Validation probe grids derived from an inference scheduler choice.

The validation probe evaluates single denoising predictions at a fixed set
of sigmas. A schedule config names the scheduler the trained checkpoint
will be sampled with; the grid is that scheduler's sigma selection applied
to the model's own discrete sigma ladder. The sampler (euler) does not
change the grid -- only the scheduler does.

The math matches the reference sampler's scheduler implementations:
- normal: linspace in timestep space over the ladder's span, mapped back
  through the schedule (linear for flow-matching ladders)
- simple: even-index samples of the ladder, starting at sigma_max
- beta: beta-CDF quantile indices into the ladder (alpha = beta = 0.6)

All functions expect the ladder as a descending 1-D tensor of sigmas
(index 0 = sigma_max) and return a descending list of floats. The terminal
sigma 0 is never emitted: a probe at sigma 0 carries no information.
"""
from typing import List

import numpy as np
import torch
from scipy import stats

# probe cost is linear in forward passes; the grid is capped so a long
# inference schedule does not turn a validation round into a benchmark
MAX_PROBE_POINTS = 8

SCHEDULE_NAMES = ("normal", "simple", "beta")


def _as_descending_numpy(ladder: torch.Tensor) -> np.ndarray:
    sigmas = ladder.detach().float().cpu().numpy().astype(np.float64)
    if sigmas[0] < sigmas[-1]:
        sigmas = sigmas[::-1]
    return sigmas


def _subsample(grid: np.ndarray, max_points: int = MAX_PROBE_POINTS) -> List[float]:
    if len(grid) <= max_points:
        return [float(s) for s in grid]
    idx = np.linspace(0, len(grid) - 1, max_points)
    idx = np.round(idx).astype(int)
    return [float(grid[i]) for i in idx]


def normal_grid(ladder: torch.Tensor, steps: int) -> np.ndarray:
    sigmas = _as_descending_numpy(ladder)
    t = sigmas * 1000.0
    # ascending side for interpolation
    t_asc = t[::-1]
    s_asc = sigmas[::-1]
    ts = np.linspace(t[0], t[-1], steps)
    return np.interp(ts, t_asc, s_asc)


def simple_grid(ladder: torch.Tensor, steps: int) -> np.ndarray:
    sigmas = _as_descending_numpy(ladder)
    stride = len(sigmas) / steps
    return np.array([sigmas[int(x * stride)] for x in range(steps)])


def beta_grid(ladder: torch.Tensor, steps: int, alpha: float = 0.6, beta: float = 0.6) -> np.ndarray:
    sigmas = _as_descending_numpy(ladder)
    # quantile indices are defined against an ascending ladder (comfy's
    # orientation); flip, index, and the result comes back descending
    ascending = sigmas[::-1]
    total = len(ascending) - 1
    quantiles = 1.0 - np.linspace(0.0, 1.0, steps, endpoint=False)
    idx = np.rint(stats.beta.ppf(quantiles, alpha, beta) * total).astype(int)
    out = []
    last = -1
    for i in idx:
        if i != last:
            out.append(ascending[i])
        last = i
    return np.array(out)


_GRID_FUNCS = {
    "normal": normal_grid,
    "simple": simple_grid,
    "beta": beta_grid,
}


def sigmas_from_schedule(schedule: dict, ladder: torch.Tensor) -> List[float]:
    """Sigma probe grid for a {'scheduler': name, 'steps': n} config."""
    name = schedule.get("scheduler")
    steps = schedule.get("steps")
    if name not in _GRID_FUNCS:
        raise ValueError(
            f"validation_schedule.scheduler must be one of {list(_GRID_FUNCS)}, got {name!r}"
        )
    if not isinstance(steps, int) or steps < 1:
        raise ValueError(f"validation_schedule.steps must be a positive int, got {steps!r}")
    grid = _GRID_FUNCS[name](ladder, steps)
    grid = grid[grid > 0.0]
    if len(grid) == 0:
        raise ValueError("validation_schedule produced an empty sigma grid")
    return _subsample(grid)
