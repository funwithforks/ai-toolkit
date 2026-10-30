from .mageflow import MageFlowModel, MageFlowEditModel

# registers the "mage" vae_id (the vae class and its load live in src/vae.py)
from . import mageflow_vae_id  # noqa: F401
