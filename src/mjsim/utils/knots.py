"""Numerical Alexander fingerprints for a single closed polygonal centerline.

This is practical recognition within a small catalogue, not a knot-equivalence
proof. Other knots can share these fingerprints; mirrors are indistinguishable.
No virtual closure, smoothing, or geometric simplification is performed.
"""

from dataclasses import dataclass

import numpy as np

FINGERPRINTS = {
    "unknot": (1.0, 1.0, 1.0),
    "trefoil": (3.0, 2.0, 1.0),
    "figure-eight": (5.0, 4.0, 3.0),
}


class KnotClassificationError(ValueError):
    """Invalid centerline or an inconclusive numerical calculation."""


@dataclass(frozen=True)
class KnotResult:
    """A fingerprint match, with None meaning outside the supported catalogue."""

    knot: str | None
    alexander_roots: tuple[float, float, float]
    crossings: tuple[int, ...]


def _segment_distances(a, b, c, d):
    """Distances for arrays of segment pairs, including parallel segments."""
    u, v, w = b - a, d - c, a - c
    uu = np.einsum("ij,ij->i", u, u)
    vv = np.einsum("ij,ij->i", v, v)
    uv = np.einsum("ij,ij->i", u, v)
    uw = np.einsum("ij,ij->i", u, w)
    vw = np.einsum("ij,ij->i", v, w)
    denom = uu * vv - uv * uv
    s = np.divide(
        uv * vw - vv * uw,
        denom,
        out=np.zeros_like(denom),
        where=denom > 1e-14 * uu * vv,
    )
    t = (uv * s + vw) / vv
    interior = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    distance = np.where(
        interior, np.linalg.norm(w + s[:, None] * u - t[:, None] * v, axis=1), np.inf
    )
    for point, origin, direction, length in (
        (a, c, v, vv),
        (b, c, v, vv),
        (c, a, u, uu),
        (d, a, u, uu),
    ):
        fraction = np.clip(
            np.einsum("ij,ij->i", point - origin, direction) / length, 0, 1
        )
        distance = np.minimum(
            distance,
            np.linalg.norm(point - origin - fraction[:, None] * direction, axis=1),
        )
    return distance


def _prepare(points, tolerance):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise KnotClassificationError(
            "Expected at least three finite ordered 3D points"
        )
    scale = np.max(np.ptp(p, axis=0))
    if not np.isfinite(scale) or scale <= 0:
        raise KnotClassificationError("Centerline has no spatial extent")
    p = (p - p[0]) / scale
    if np.linalg.norm(p[-1] - p[0]) <= tolerance:
        p = p[:-1]
    if len(p) < 3:
        raise KnotClassificationError("A closed polygon needs at least three vertices")
    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= tolerance):
        raise KnotClassificationError("Centerline contains a zero-length edge")
    # Consecutive edges may meet, but must not double back over each other.
    unit = edges / lengths[:, None]
    if np.any(np.linalg.norm(unit + np.roll(unit, 1, axis=0), axis=1) <= tolerance):
        raise KnotClassificationError("Adjacent edges overlap")
    i, j = np.triu_indices(len(p), 2)
    keep = ~((i == 0) & (j == len(p) - 1))
    i, j = i[keep], j[keep]
    if np.any(
        _segment_distances(p[i], p[i] + edges[i], p[j], p[j] + edges[j]) <= tolerance
    ):
        raise KnotClassificationError(
            "Nonadjacent centerline segments intersect or are too close"
        )
    return p, edges, i, j


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _diagram(p, edges, i, j, direction, tol):
    direction = direction / np.linalg.norm(direction)
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    u = np.cross(direction, axis)
    u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(direction, u)))
    q, e = p @ basis, edges @ basis
    if np.any(np.linalg.norm(e, axis=1) <= tol):
        return None
    denominator = _cross2(e[i], e[j])
    parallel = np.abs(denominator) <= tol * np.linalg.norm(
        e[i], axis=1
    ) * np.linalg.norm(e[j], axis=1)
    offset = q[j] - q[i]
    # A straight cable contains many disjoint collinear segments. Only an
    # overlap of their finite projected intervals makes the diagram singular.
    lower = np.maximum(np.minimum(q[i], q[i] + e[i]), np.minimum(q[j], q[j] + e[j]))
    upper = np.minimum(np.maximum(q[i], q[i] + e[i]), np.maximum(q[j], q[j] + e[j]))
    overlap = np.all(lower <= upper + tol, axis=1)
    if np.any(parallel & overlap & (np.abs(_cross2(offset, e[i])) <= tol)):
        return None
    safe = np.where(parallel, 1.0, denominator)
    s, t = _cross2(offset, e[j]) / safe, _cross2(offset, e[i]) / safe
    possible = ~parallel & (s >= -tol) & (s <= 1 + tol) & (t >= -tol) & (t <= 1 + tol)
    if np.any(possible & ((s <= tol) | (s >= 1 - tol) | (t <= tol) | (t >= 1 - tol))):
        return None
    indices = np.flatnonzero(possible)
    if not len(indices):
        return []
    locations = q[i[indices]] + s[indices, None] * e[i[indices]]
    if len(locations) > 1:
        a, b = np.triu_indices(len(locations), 1)
        if np.any(np.linalg.norm(locations[a] - locations[b], axis=1) <= tol):
            return None
    crossings = []
    for k in indices:
        depth = np.dot(
            p[i[k]] + s[k] * edges[i[k]] - p[j[k]] - t[k] * edges[j[k]], direction
        )
        if abs(depth) <= tol:
            return None
        first, second = i[k] + s[k], j[k] + t[k]
        over, under = (first, second) if depth > 0 else (second, first)
        sign = np.sign(denominator[k] * depth)
        crossings.append((over, under, sign))
    return crossings


def _fingerprint(crossings):
    n = len(crossings)
    if n == 0:
        return np.ones(3)
    under = np.sort([c[1] for c in crossings])
    result = []
    for root in (2, 3, 4):
        t = np.exp(2j * np.pi / root)
        matrix = np.zeros((n, n), dtype=complex)
        for row, (over, below, sign) in enumerate(crossings):
            incoming = int(np.searchsorted(under, below))
            outgoing = (incoming + 1) % n
            over_arc = int(np.searchsorted(under, over)) % n
            # Wirtinger relation, with inverse t for the opposite crossing.
            value = t if sign > 0 else 1 / t
            matrix[row, over_arc] += 1 - value
            matrix[row, incoming] += value
            matrix[row, outgoing] -= 1
        result.append(abs(np.linalg.det(matrix[:-1, :-1])))
    return np.asarray(result)


def classify_closed_knot(
    body_positions, *, tolerance: float = 1e-8, max_crossings: int = 100
) -> KnotResult:
    """Match a closed polygon to unknot/trefoil/figure-eight fingerprints.

    Points must follow the cable. The last-to-first edge is implicit; an exactly
    repeated endpoint is also accepted. This function cannot infer whether a
    physical cable is actually closed. ``tolerance`` is relative to its extent.
    Three generic projections must agree. Degenerate projections are retried;
    invalid geometry, resource limits and disagreement raise
    :class:`KnotClassificationError`, never a negative match.

    Values are |Alexander(exp(2*pi*i/r))| for r=2,3,4. Matching values are
    evidence within the demo catalogue, not proof for arbitrary knots.
    """
    if not np.isfinite(tolerance) or not 0 < tolerance < 1e-2:
        raise ValueError("tolerance must be finite and between 0 and 0.01")
    if not isinstance(max_crossings, int) or max_crossings < 1:
        raise ValueError("max_crossings must be a positive integer")
    p, edges, i, j = _prepare(body_positions, tolerance)
    rng = np.random.default_rng(1729)
    fingerprints, counts = [], []
    for direction in rng.normal(size=(24, 3)):
        diagram = _diagram(p, edges, i, j, direction, tolerance)
        if diagram is None or len(diagram) > max_crossings:
            continue
        fp = _fingerprint(diagram)
        if not np.isfinite(fp).all():
            raise KnotClassificationError("Non-finite Alexander calculation")
        if fingerprints and not np.allclose(fp, fingerprints[0], atol=1e-6, rtol=1e-6):
            raise KnotClassificationError("Projection fingerprints disagree")
        fingerprints.append(fp)
        counts.append(len(diagram))
        if len(counts) == 3:
            match = next(
                (
                    name
                    for name, expected in FINGERPRINTS.items()
                    if np.allclose(fp, expected, atol=1e-6, rtol=1e-6)
                ),
                None,
            )
            return KnotResult(match, tuple(float(x) for x in fp), tuple(counts))
    raise KnotClassificationError(
        "Could not find three regular projections within the crossing limit"
    )


def is_target_knot(body_positions, target: str = "trefoil", **kwargs) -> bool:
    """Return a Python bool for a supported fingerprint match; errors propagate.

    ``overhand`` aliases ``trefoil`` for an already closed curve. It does not
    enable classification of open ropes or choose a virtual closure.
    """
    if target == "overhand":
        target = "trefoil"
    if target not in FINGERPRINTS:
        raise ValueError(
            f"Unknown target {target!r}; choose from {tuple(FINGERPRINTS)}"
        )
    return classify_closed_knot(body_positions, **kwargs).knot == target
