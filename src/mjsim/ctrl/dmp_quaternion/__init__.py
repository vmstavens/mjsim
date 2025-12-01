"""Dynamic Movement Primitive for unit quaternions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from mjsim.ctrl.dmp_cartesian.canonical_system import CanonicalSystem
from mjsim.ctrl.dmp_position import DMPPosition


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion must not be zero.")
    return q / norm


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


class DMPQuaternion:
    """Orientation DMP implemented through rotation-vector coordinates.

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
        roto_dilatation: bool | None = None,  # compatibility flag
    ) -> None:
        self.cs = cs or CanonicalSystem(alpha=alpha)
        self._dmp = DMPPosition(
            n_bfs=n_bfs, alpha=alpha, beta=beta, cs=self.cs, roto_dilatation=False
        )

    def reset(self) -> None:
        """Reset internal DMP state."""
        self._dmp.reset()

    def train(self, quaternions: np.ndarray, ts: np.ndarray, tau: float) -> None:
        """Fit the DMP on a quaternion demonstration.

        Args:
            quaternions: Sequence of quaternions in wxyz ordering.
            ts: Time stamps matching ``quaternions``.
            tau: Duration scaling.
        """
        quaternions = np.asarray(quaternions, dtype=float)
        if quaternions.ndim != 2 or quaternions.shape[1] != 4:
            raise ValueError("quaternions must be shaped (N, 4) in wxyz ordering.")

        # Normalize and convert to rotation vectors for Euclidean DMP fitting.
        rotvecs = []
        for q in quaternions:
            q_norm = _normalize_quat(q)
            r = Rotation.from_quat(_wxyz_to_xyzw(q_norm))
            rotvecs.append(r.as_rotvec())
        rotvecs = np.asarray(rotvecs)

        self._dmp.train(rotvecs, ts, tau)

    def step(
        self,
        x: float,
        dt: float,
        tau: float,
        torque_disturbance: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance the quaternion DMP by a single step.

        Args:
            x: Phase variable.
            dt: Time step.
            tau: Temporal scaling.
            torque_disturbance: Optional disturbance added to angular acceleration.

        Returns:
            Tuple of ``(quaternion_wxyz, angular_velocity, angular_acceleration)``.
        """
        rotvec, drotvec, ddrotvec = self._dmp.step(
            x, dt, tau, force_disturbance=torque_disturbance
        )
        quat = _xyzw_to_wxyz(Rotation.from_rotvec(rotvec).as_quat())
        omega = drotvec
        d_omega = ddrotvec
        return quat, omega, d_omega
