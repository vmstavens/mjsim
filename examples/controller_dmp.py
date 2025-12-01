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

    p, dp, ddp, q, omega, d_omega = dmp.rollout(ts, tau=ts[-1])
    print(f"Rollout generated {len(p)} waypoints.")
    print("First pose:", p[0], "Last pose:", p[-1])
    print("First orientation (quaternion wxyz):", np.asarray(q[0]))


if __name__ == "__main__":
    main()
