"""An explicit outer-arc convention for open-cable knot experiments.

The bow is fixed in world space; it must be chosen for the experiment. This is
not a universal, topology-neutral closure of arbitrary open curves.
"""

from dataclasses import dataclass

import numpy as np

from mjsim.utils.knots import (
    KnotClassificationError,
    _prepare,
    _segment_distances,
    classify_closed_knot,
)


class ClosureError(KnotClassificationError):
    """The chosen virtual return cannot safely classify this cable snapshot."""


def outer_return_arc(points, *, bow=(0.0, -0.30, 0.0), segments=24):
    """Return an end-to-start arc, including both physical endpoints.

    Uses the cosine interpolation and sinusoidal bow of the closed overhand
    demo. The arc adapts continuously to moving endpoints without silently
    switching closure directions to obtain a desired knot label.
    """
    p = np.asarray(points, dtype=float)
    bow = np.asarray(bow, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ClosureError("Expected at least three finite ordered 3D points")
    if bow.shape != (3,) or not np.isfinite(bow).all() or np.linalg.norm(bow) == 0:
        raise ValueError("bow must be a finite nonzero 3D vector")
    if not isinstance(segments, int) or segments < 4:
        raise ValueError("segments must be an integer >= 4")
    if np.linalg.norm(p[-1] - p[0]) < 1e-8:
        raise ClosureError("Endpoints are too close for this open-cable closure")
    theta = np.linspace(0, np.pi, segments + 1)
    u = (1 - np.cos(theta)) / 2
    arc = (1 - u[:, None]) * p[-1] + u[:, None] * p[0]
    arc += np.sin(theta)[:, None] * bow
    arc[0], arc[-1] = p[-1], p[0]
    return arc


def close_with_outer_arc(points, *, clearance=0.005, **kwargs):
    """Return a closed polygon and arc, rejecting near/intersecting returns.

    Clearance is in the same units as the points. The two segment pairs sharing
    the physical endpoints are exempt from the distance check only; full
    embedded-polygon validation still checks overlapping adjacent segments.
    """
    if not np.isfinite(clearance) or clearance <= 0:
        raise ValueError("clearance must be finite and positive")
    arc = outer_return_arc(points, **kwargs)
    p = np.asarray(points, dtype=float)
    polygon = np.vstack((p, arc[1:-1]))
    try:
        _prepare(polygon, 1e-8)
    except KnotClassificationError as error:
        raise ClosureError(str(error)) from error
    i, j = np.indices((len(p) - 1, len(arc) - 1))
    i, j = i.ravel(), j.ravel()
    shared = ((i == len(p) - 2) & (j == 0)) | ((i == 0) & (j == len(arc) - 2))
    i, j = i[~shared], j[~shared]
    gap = float(_segment_distances(p[i], p[i + 1], arc[j], arc[j + 1]).min())
    if gap < clearance:
        raise ClosureError(
            f"Virtual return clearance {gap * 1000:.2f} mm is below {clearance * 1000:.2f} mm"
        )
    return polygon, arc


@dataclass
class KnotProgress:
    """Consecutive valid samples, with current success and a first-hit timestamp."""

    required_matches: int = 5
    streak: int = 0
    first_success_time: float | None = None
    label: str | None = None
    error: str | None = None
    fingerprint: tuple[float, ...] | None = None
    closure_bow: tuple[float, float, float] = (0.0, -0.30, 0.0)

    def __post_init__(self):
        if not isinstance(self.required_matches, int) or self.required_matches < 1:
            raise ValueError("required_matches must be a positive integer")

    @property
    def successful(self):
        return self.streak >= self.required_matches

    def sample(self, points, timestamp, *, tracking_error=0.0):
        """Consume one scheduled sample; invalid samples reset the streak."""
        self.error = None
        self.label = None
        self.fingerprint = None
        try:
            if not np.isfinite(tracking_error) or tracking_error > 0.002:
                raise ClosureError("Endpoint gripper tracking error exceeds 2 mm")
            polygon, _ = close_with_outer_arc(points, bow=self.closure_bow)
            result = classify_closed_knot(polygon)
            self.label = result.knot
            self.fingerprint = result.alexander_roots
            self.streak = self.streak + 1 if result.knot == "trefoil" else 0
            if self.successful and self.first_success_time is None:
                self.first_success_time = float(timestamp)
        except KnotClassificationError as error:
            self.error = str(error)
            self.streak = 0
