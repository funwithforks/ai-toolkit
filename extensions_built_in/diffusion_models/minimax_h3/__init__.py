from .minimax_h3 import (
    MinimaxH3Model,
    MinimaxH3Ref2VAModel,
    MinimaxH3FastModel,
    MinimaxH3FlashModel,
)
from .minimax_h3_image import MinimaxH3ImageModel
# registers the "minimax_h3" vae_id entry (video+audio bundle) and the
# "minimax_h3_image" entry (image vae alone)
from . import minimax_h3_vae_id  # noqa: F401
