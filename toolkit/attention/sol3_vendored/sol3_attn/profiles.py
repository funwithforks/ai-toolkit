"""Per-model launch policies for the SM80 and SM120 backends."""

from dataclasses import dataclass
from types import MappingProxyType


LONG_SEQUENCE = 16_385


@dataclass(frozen=True)
class Profile:
    capability: tuple[int, int]
    representative_stages: int | None = None
    centered_tiles: tuple[int, int] | None = None
    ordered_pages: bool = False
    repair_plan: bool = False
    long_heads: int | None = None
    joint_launch: tuple[int, int] | None = None


PROFILES = MappingProxyType({
    "wan": Profile((8, 0), representative_stages=3, centered_tiles=(64, 64), long_heads=16),
    "hy": Profile((8, 0), representative_stages=2),
    "sana": Profile((8, 0), ordered_pages=True),
    "ltx": Profile((8, 0), ordered_pages=True, long_heads=16),
    "h3": Profile((8, 0), representative_stages=2, ordered_pages=True, repair_plan=True, long_heads=16),
    "fast": Profile((12, 0), long_heads=14, joint_launch=(64, 4)),
})
