"""Simple Cartesian Dynamic Movement Primitive."""

from __future__ import annotations

import numpy as np

from mjsim.ctrl.dmp_cartesian.canonical_system import CanonicalSystem


class DMPPosition:
    """Lightweight Dynamic Movement Primitive for Euclidean positions.

    Args:
        n_bfs: Number of basis functions.
        alpha: Attractor gain.
        beta: Secondary attractor gain; defaults to ``alpha / 4``.
        cs: Optional shared canonical system.
        roto_dilatation: Unused backward-compatibility flag.
    """

    def __init__(
        self,
        n_bfs: int = 50,
        alpha: float = 25.0,
        beta: float | None = None,
        cs: CanonicalSystem | None = None,
        roto_dilatation: bool | None = None,  # kept for backward compatibility
    ) -> None:
        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta if beta is not None else alpha / 4.0
        self.cs = cs or CanonicalSystem(alpha=self.alpha)

        # Basis function parameters.
        self.centers = np.exp(-np.linspace(0, 1, self.n_bfs) * self.cs.alpha)
        self.widths = (self.n_bfs**1.5) / (self.centers * self.cs.alpha)

        self.y0: np.ndarray | None = None
        self.goal: np.ndarray | None = None
        self.weights = np.zeros((self.n_bfs, 3))
        self.dim = 3

        self.y = np.zeros(self.dim)
        self.dy = np.zeros(self.dim)
        self.trained = False

    def reset(self) -> None:
        """Reset state to the last trained initial condition."""
        if self.y0 is None:
            self.y = np.zeros(self.dim)
        else:
            self.y = self.y0.copy()
        self.dy = np.zeros_like(self.y)

    def _basis(self, x: float) -> np.ndarray:
        return np.exp(-self.widths * (x - self.centers) ** 2)

    def train(self, positions: np.ndarray, ts: np.ndarray, tau: float) -> None:
        """Fit basis function weights to a demonstration.

        Args:
            positions: Demonstration positions ``(T, 3)``.
            ts: Time stamps matching ``positions``.
            tau: Duration scaling.
        """
        positions = np.asarray(positions, dtype=float)
        ts = np.asarray(ts, dtype=float)
        self.dim = positions.shape[1] if positions.ndim > 1 else 1

        assert positions.shape[0] == ts.shape[0], "positions and ts must align in length"

        # Derivatives.
        dt = np.gradient(ts)
        dy = np.gradient(positions, axis=0) / dt[:, None]
        ddy = np.gradient(dy, axis=0) / dt[:, None]

        self.y0 = positions[0].copy()
        self.goal = positions[-1].copy()
        self.reset()

        x_track = self.cs.rollout(ts, tau)

        psi = self._basis(x_track[:, None])
        psi_sum = np.sum(psi, axis=1, keepdims=True) + 1e-8
        psi_norm = psi / psi_sum
        B = psi_norm * x_track[:, None]

        f_target = (tau**2 * ddy) - self.alpha * (
            self.beta * (self.goal - positions) - tau * dy
        )

        weights = []
        for dim in range(self.dim):
            w, _, _, _ = np.linalg.lstsq(B, f_target[:, dim], rcond=1e-6)
            weights.append(w)
        self.weights = np.stack(weights, axis=1)
        self.trained = True

    def step(
        self,
        x: float,
        dt: float,
        tau: float,
        force_disturbance: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance the DMP by a single time step.

        Args:
            x: Phase variable.
            dt: Time step.
            tau: Temporal scaling.
            force_disturbance: Optional disturbance added to acceleration.

        Returns:
            Tuple of ``(position, velocity, acceleration)``.
        """
        if self.goal is None or self.y0 is None:
            raise RuntimeError("Call train() before using step().")

        disturbance = (
            np.asarray(force_disturbance, dtype=float)
            if force_disturbance is not None
            else 0.0
        )

        psi = self._basis(x)
        psi_sum = np.sum(psi) + 1e-8
        forcing = (psi @ self.weights) * x / psi_sum

        ddy = (self.alpha * (self.beta * (self.goal - self.y) - tau * self.dy)) / (
            tau**2
        )
        ddy += forcing / (tau**2)
        ddy += disturbance

        self.dy += ddy * dt
        self.y += self.dy * dt
        return self.y.copy(), self.dy.copy(), ddy
