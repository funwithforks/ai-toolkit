"""vae_id registry: component-keyed VAE resolution for phased loading.

A model opts in by declaring the class attribute ``vae_id``; the phased
loader (toolkit.models.phased_load.PhasedLoadMixin.load_vae) then resolves
construction and placement from the registered entry instead of the model's
own ``_load_vae`` body. Models that declare no id are untouched -- the
default-off rule for shared loading code.

Each id owns one vae object: its module carries the file resolution and the
load (the entry below), and the encode/decode live as methods on the vae
class itself -- the object answers for its own math and its own precision,
so there is no per-model encode path to keep in step and nothing
encode/decode-shaped on the entry. Component candidate metadata stays where
it lives today -- on the component classes as ``aitk_comfy_repo`` /
``aitk_comfy_weight_names`` -- so the registry registers candidates through
class references and NEVER paths; file resolution stays the comfy door
(toolkit.models.v2.resolver: explicit <component>_path override > local
roots > MODELS_PATH layout > hub).

A bundle (multiple sub-VAEs behind the single ``self.vae`` handle, e.g.
video + audio with members that may be None under staged sampling) is
owned by the id: ``bundle=True`` only says so, which is what lets
``self.vae`` members be probed for residency.

Precision is not a policy: a vae is loaded at the precision it ships in, by
its own vendored loader, and placement only moves it. There is no cast
option to choose -- upstream's blanket "cast the vae to the job's dtype"
loses precision for no benefit and is not offered.

``te_id`` is the future sibling: it will use this entry shape through its
own table; nothing here needs a reshape for it.

Registration happens at extension import time (the id module imports
itself next to the model class). An unknown id fails at model construction,
not at first encode -- see PhasedLoadMixin.__init__.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Type

@dataclass(frozen=True)
class VaeEntry:
    vae_id: str
    # role -> component class; a plain vae is one role, a bundle has several
    components: Dict[str, Type]
    # (model) -> vae module or bundle: the id's own loader resolves candidates
    # through the comfy door and streams the checkpoint onto the target device
    # at shipped precision. Placement upstream only moves it -- no cast exists
    load: Callable[[object], object]
    bundle: bool = False


_REGISTRY: Dict[str, VaeEntry] = {}


def register_vae_entry(entry: VaeEntry) -> None:
    if entry.vae_id in _REGISTRY:
        raise ValueError(f"duplicate vae_id registration: {entry.vae_id!r}")
    _REGISTRY[entry.vae_id] = entry


def get_vae_entry(vae_id: str) -> VaeEntry:
    try:
        return _REGISTRY[vae_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise ValueError(
            f"unknown vae_id {vae_id!r}; registered ids: {known}. An id is "
            "registered by importing its module -- if the owning extension "
            "is installed, its model module imports the id module."
        ) from None
