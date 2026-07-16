"""Canonical multiplet-label schema + adapters unifying the two datasets.

The synthetic and experimental datasets store labels under different field names and units:

- **synthetic** (``data/custom_spin_systems/*.npz`` key ``labels``):
  ``proton_number``, ``center_position_in_points``, ``line_width_in_points``,
  ``bounding_box_range_in_points``, ``coupling_constants_in_points`` (couplings in **points**).
- **experimental** (``structured_output/roi_S*.npz`` key ``ground_truth``; and ``roi_S*.json``):
  ``proton_count``, ``chemical_shift_in_points``, ``coupling_constants`` (**Hz**), ``chemical_shift_ppm``.

``from_synthetic`` / ``from_experimental`` normalize either into a common :class:`Multiplet` so
downstream code (metrics, plots, GUI) treats both uniformly. See ``docs/DATA_SCHEMA.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

POINTS_PER_HZ = 5.12


def _as_list(x) -> list:
    if x is None:
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _opt_float(v):
    return None if v is None else float(v)


@dataclass
class Multiplet:
    """Canonical multiplet label. Positions in points; couplings always in Hz."""

    proton_count: int
    center_in_points: float
    coupling_constants_hz: list = field(default_factory=list)
    line_width_in_points: float | None = None
    bounding_box_range_in_points: float | None = None
    chemical_shift_ppm: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def from_synthetic(label: dict, points_per_hz: float = POINTS_PER_HZ) -> Multiplet:
    """Adapt a synthetic ``labels`` entry (couplings in points) to the canonical schema."""
    couplings = _as_list(label.get("coupling_constants_in_points"))
    return Multiplet(
        proton_count=int(label["proton_number"]),
        center_in_points=float(label["center_position_in_points"]),
        coupling_constants_hz=[float(c) / points_per_hz for c in couplings],
        line_width_in_points=_opt_float(label.get("line_width_in_points")),
        bounding_box_range_in_points=_opt_float(label.get("bounding_box_range_in_points")),
    )


def from_experimental(label: dict) -> Multiplet:
    """Adapt an experimental ``ground_truth`` / json label (couplings in Hz) to canonical."""
    return Multiplet(
        proton_count=int(label["proton_count"]),
        center_in_points=float(label["chemical_shift_in_points"]),
        coupling_constants_hz=[float(c) for c in _as_list(label.get("coupling_constants"))],
        chemical_shift_ppm=_opt_float(label.get("chemical_shift_ppm")),
    )
