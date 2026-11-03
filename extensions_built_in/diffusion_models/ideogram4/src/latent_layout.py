"""ideogram4 latent layout: the DiT's packing vs the shared flux2 cache.

The ``flux2`` vae_id cache holds the vae's own encode output: the packed
mean in ``(c, pi, pj)`` channel order, BatchNorm-normalized by the vae's
own running stats. The ideogram4 DiT was trained on the same underlying
packed mean in ``(pi, pj, c)`` channel order, normalized by the fixed
``LATENT_SHIFT``/``LATENT_SCALE`` tables. The two are related by an exact
per-channel affine through the vae's own stats -- measured at ~5e-7 max
error against the native math on the real checkpoint:

    to_train(z):  z[idx] * sg[idx] + mu[idx], then (x - shift) / scale
    to_cache(z):  z[inv] * scale[inv] + shift[inv], then (x - mu) / sg

The conversion happens at the noising/target boundary (``add_noise`` /
``get_loss_target``) and at decode -- never on the cache path. The cache
stays the vae's canonical output, shared with every other model on the
id. Noising must happen AFTER the conversion: the DiT was trained on
``x_t = (1-sigma) * latent + sigma * noise`` with N(0, I) noise in its
own layout, and an affine-mapped noise component is not N(0, I).

The constants come from the vae's running stats -- the resident module's
buffers when it is loaded, otherwise a header read of the resolved file
(warm runs never load the vae, and the buffers are 1 KB either way).
"""

import torch
from safetensors import safe_open

from toolkit.models.v2.vae.flux2_kl import AutoEncoder
from .latent_norm import LATENT_SHIFT, LATENT_SCALE

_BN_EPS = 1e-4  # AutoEncoder.bn_eps


def _pack_index_maps():
    """Canonical ``(c, pi, pj)`` channel index <-> train ``(pi, pj, c)``."""
    idx = torch.empty(128, dtype=torch.long)  # train channel -> canonical channel
    inv = torch.empty(128, dtype=torch.long)  # canonical channel -> train channel
    for c in range(32):
        for pi in range(2):
            for pj in range(2):
                j = c * 4 + pi * 2 + pj
                i = pi * 64 + pj * 32 + c
                idx[i] = j
                inv[j] = i
    return idx, inv


class LatentLayout:
    def __init__(self, sg: torch.Tensor, mu: torch.Tensor, device) -> None:
        self.idx, self.inv = _pack_index_maps()
        self.sg = sg.float().reshape(-1).to(device).view(1, -1, 1, 1)
        self.mu = mu.float().reshape(-1).to(device).view(1, -1, 1, 1)
        self.shift = torch.tensor(LATENT_SHIFT, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.scale = torch.tensor(LATENT_SCALE, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    def to_train(self, z: torch.Tensor) -> torch.Tensor:
        """Canonical cache form -> the DiT's layout: de-normalize with the
        vae's stats, permute the pack, renormalize with the tables."""
        zc = (z * self.sg + self.mu)[:, self.idx]
        return (zc - self.shift) / self.scale

    def to_cache(self, z: torch.Tensor) -> torch.Tensor:
        """The DiT's layout -> canonical cache form: de-normalize with the
        tables, permute back, renormalize with the vae's stats."""
        zt = (z * self.scale + self.shift)[:, self.inv]
        return (zt - self.mu) / self.sg


def layout_from_vae(vae, device) -> LatentLayout:
    """From a resident (or parked) flux2 AutoEncoder's own buffers."""
    return LatentLayout(
        (vae.bn.running_var + vae.bn_eps).sqrt(), vae.bn.running_mean, device
    )


def layout_from_file(path: str, device) -> LatentLayout:
    """From the resolved checkpoint's header alone -- no weight read, no
    residency. This is the warm-run path: a run whose latents are all
    cached never loads the vae, and the conversion still needs its stats."""
    with safe_open(path, framework="pt") as f:
        mu = f.get_tensor("bn.running_mean")
        var = f.get_tensor("bn.running_var")
    return LatentLayout((var + _BN_EPS).sqrt(), mu, device)


def resolve_vae_file(model) -> str:
    """The shared comfy door, honoring the job's ``vae_path`` override."""
    return model._resolve_comfy_file("vae", AutoEncoder)
