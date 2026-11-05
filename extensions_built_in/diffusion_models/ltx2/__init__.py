from .ltx2 import LTX2Model, LTX23Model, LTX25Model

# registers the "ltx2" vae_id entry (video+audio bundle; vocoder streams on
# demand at decode) -- import-order: after the model classes
from . import ltx23_vae_id  # noqa: F401
