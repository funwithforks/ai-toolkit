from . import qwen38_vae_id  # noqa: F401  (registers the "qwen38" vae_id entry)
from .qwen25_omni import Qwen25OmniLLM
from .qwen38 import Qwen38LLM

AI_TOOLKIT_MODELS = [
    Qwen25OmniLLM,
    Qwen38LLM,
]
