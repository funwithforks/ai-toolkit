"""Suite modules are collected here; registry.py imports this once."""

from .h3_train import SUITE as H3_TRAIN
from .attn_micro import SUITE as ATTN_MICRO
from .h3_probe import SUITE as H3_PROBE

ALL = (H3_TRAIN, ATTN_MICRO, H3_PROBE)
