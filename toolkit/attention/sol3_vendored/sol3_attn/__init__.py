"""Sol³ forward inference."""

from .interface import release_workspace, sol3_attn, sol3_attn_with_route

__all__ = ["sol3_attn", "sol3_attn_with_route", "release_workspace"]
