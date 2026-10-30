"""mage vae_id: the Mage-Flow autoencoder behind one id.

Registration side only: say which class implements the vae and how to load it.
Everything the vae *does* -- ``encode_images`` / ``decode_latents`` -- lives on
``MageVAE`` (src/vae.py), and both mageflow archs (``mageflow`` text-to-image,
``mageflow_edit`` instruction edit) join it with the one line
``vae_id = "mage"``. They are the same physical encoder, so one cached latent is
valid for either of them and a dataset encoded by one trains the other.

Cache identity is the ``vae_id`` itself: latent cache files hash the id, so
there is no version knob to keep in step.

The load is the model's former ``_load_vae`` body, shortened to the door plus
the class's own loader: the file is the named comfy single file (resolved
through the comfy door, ``vae_path`` in ``model_kwargs`` still short-circuits
it), and it is streamed to the vae device at the precision it ships in. The old
body's ``vae_sample_posterior`` kwarg is gone with it -- sampled versus mean
latents are two different latent distributions, and one id has one of them. The
class still takes the argument; a job that ever wants mean latents needs its own
id, not a silent switch under this one.
"""

from toolkit.models.vae_registry import VaeEntry, register_vae_entry

from .src.vae import MageVAE


def _load_mage_vae(model):
    """-> MageVAE resident on model.vae_device_torch at its shipped dtype."""
    path = model._resolve_comfy_file("vae", MageVAE)
    model.print_and_status_update(f"Loading MageVAE from {path}")
    return MageVAE.load_to_device(path, model.vae_device_torch)


register_vae_entry(
    VaeEntry(
        vae_id="mage",
        components={"vae": MageVAE},
        load=_load_mage_vae,
    )
)
