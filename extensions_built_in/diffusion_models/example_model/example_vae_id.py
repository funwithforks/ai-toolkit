"""example vae_id: registering the template's own vae.

This is the half of the contract a new model writes for its autoencoder: say
which class implements the vae, and how to load it. Everything the vae *does*
(its encode and decode) lives on the class, in ``src/vae.py``. The model side
is one line -- ``vae_id = "example"`` -- and it keeps no vae code of its own.

Cache identity is the ``vae_id`` itself: latent cache files hash the id, so
every model declaring the same id shares the same latents, and there is no
version knob to keep in step.
"""

from toolkit.models.vae_registry import VaeEntry, register_vae_entry

from .src.vae import ExampleVAE


def _load_example_vae(model):
    """Resolve the named comfy file through the generic door and stream it
    onto the vae device at the precision it ships in.

    The door owns the search: a ``vae_path`` model_kwarg short-circuits it,
    ``MODELS_PATH/vae`` is searched, a local checkpoint dir counts as a search
    root, and a miss downloads from the repo the vae class declares -- always
    into the comfy layout under MODELS_PATH, never the HF cache."""
    path = model._resolve_comfy_file("vae", ExampleVAE)
    model.print_and_status_update(f"Loading example VAE from {path}")
    return ExampleVAE.load_to_device(path, model.vae_device_torch)


register_vae_entry(
    VaeEntry(
        vae_id="example",
        components={"vae": ExampleVAE},
        load=_load_example_vae,
    )
)
