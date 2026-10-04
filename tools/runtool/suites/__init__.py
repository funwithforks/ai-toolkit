"""Suite modules are collected here; registry.py imports this once."""

from .h3_train import SUITE as H3_TRAIN
from .attn_micro import SUITE as ATTN_MICRO

ALL = (H3_TRAIN, ATTN_MICRO)
