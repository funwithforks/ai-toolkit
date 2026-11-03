"""iris3b vae_id: the pixel-space identity vae behind one id.

Iris-3B denoises raw RGB -- there is no real vae and no weights to
resolve, so the entry's load constructs ``FakeVAE`` (identity) instead of
going through the comfy door. The toolkit's "latents" are the image in
[-1, 1]; the latent caches store pixels through the vae phase unchanged,
at the vae's own fp32 (the legacy path rounded them to the job dtype on
the way in).

Cache identity is the ``vae_id`` itself: files hash the id, so any model
declaring ``iris3b`` shares the same cache namespace. The export clamp the
pipeline applies after decode stays in the pipeline -- it is range
post-processing for image output, not vae math.
"""

from toolkit.models.FakeVAE import FakeVAE
from toolkit.models.vae_registry import VaeEntry, register_vae_entry


def _load_iris3b_vae(model):
    """-> FakeVAE(scaling_factor=1.0): identity, nothing resident to page."""
    model.print_and_status_update("Preparing pixel-space VAE (identity)")
    vae = FakeVAE(scaling_factor=1.0)
    vae.eval()
    vae.requires_grad_(False)
    return vae


register_vae_entry(
    VaeEntry(
        vae_id="iris3b",
        components={"vae": FakeVAE},
        load=_load_iris3b_vae,
    )
)
