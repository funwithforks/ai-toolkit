# registers the "iris3b" vae_id entry (pixel-space identity FakeVAE)
from . import iris3b_vae_id  # noqa: F401
from .iris3b import Iris3BModel

__all__ = ["Iris3BModel"]
