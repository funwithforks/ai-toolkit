"""Model-specific overrides with a private instance of each shared module."""

from .. import _shared

__path__ = [*__path__, *_shared.__path__]
