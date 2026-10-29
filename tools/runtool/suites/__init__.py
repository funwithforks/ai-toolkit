"""Suite modules are collected here; registry.py imports this once."""

from .h3_train import SUITE as H3_TRAIN
from .attn_micro import SUITE as ATTN_MICRO
from .h3_probe import SUITE as H3_PROBE
from .h3_vae_fixtures import SUITE as H3_VAE_FIXTURES
from .ltx2_train import SUITE as LTX2_TRAIN
from .flux3_train import SUITE as FLUX3_TRAIN
from .moss_train import SUITE as MOSS_TRAIN

ALL = (H3_TRAIN, ATTN_MICRO, H3_PROBE, H3_VAE_FIXTURES, LTX2_TRAIN,
       FLUX3_TRAIN, MOSS_TRAIN)
