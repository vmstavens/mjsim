"""
Train and roll out a Cartesian/quaternion Dynamic Movement Primitive.
This example does not require a running MuJoCo simulation; it just shows
how to fit and sample a trajectory.
"""

import numpy as np
import spatialmath as sm

from mjsim.ctrl import DMPCartesian


def main() -> None:
    # Build a simple straight-line demo in task space.
    ts = np.linspace(0.0, 2.0, 50)
    poses = [
        sm.SE3(0.0 + 0.4 * t, 0.1, 0.2 + 0.2 * t) * sm.SE3.Rx(0.1 * t)
        for t in np.linspace(0, 1, len(ts))
    ]

    dmp = DMPCartesian(n_bfs=40)
    dmp.load(poses, dt=ts[1] - ts[0])

    # Integrate the high-gain attractor at 2 ms rather than the coarse 41 ms
    # demonstration sampling interval, which makes this explicit rollout unstable.
    rollout_ts = np.linspace(ts[0], ts[-1], 1001)
    p, _dp, _ddp, q, _omega, _d_omega = dmp.rollout(rollout_ts, tau=ts[-1])
    if not np.isfinite(p).all() or np.max(np.linalg.norm(p, axis=1)) > 1.0:
        raise RuntimeError("DMP rollout diverged from the demonstration workspace")
    print(f"Rollout generated {len(p)} waypoints.")
    print("First pose:", p[0], "Last pose:", p[-1])
    print("First orientation (quaternion wxyz):", np.asarray(q[0]))


if __name__ == "__main__":
    main()
