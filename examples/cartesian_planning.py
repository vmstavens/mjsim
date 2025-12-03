"""
Cartesian/task-space planning example.

Uses OMPL's RRT-Connect wrapper (`mjsim.utils.ompl.xplan`) to plan a straight
path between two poses while avoiding a spherical obstacle at the origin.

Requires the optional `ompl` dependency; exits gracefully if missing.
"""

import numpy as np

try:
    from mjsim.utils.ompl import xplan
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    raise SystemExit("OMPL is not installed. Install `ompl` to run this example.")


class _StubRobot:
    """Minimal robot stub exposing the fields used by xplan."""

    class Info:
        def __init__(self) -> None:
            self.n_joints = 6
            self.joint_limits = np.array([[-np.pi, np.pi]] * self.n_joints)

    def __init__(self) -> None:
        self.info = _StubRobot.Info()


def main() -> None:
    def validity_check(state) -> bool:
        # Reject states inside a ball of radius 0.15 around the origin.
        return (state.getX() ** 2 + state.getY() ** 2 + state.getZ() ** 2) > 0.15

    ok, path = xplan(
        robot=_StubRobot(),
        start_pose=np.array([0.3, 0.0, 0.0, 0, 0, 0, 1]),
        goal_pose=np.array([-0.3, 0.0, 0.0, 0, 0, 0, 1]),
        validity_check_fn=validity_check,
        timeout=2.0,
        simplify=True,
    )

    print("Cartesian plan:", "succeeded" if ok else "failed")
    if ok:
        print(f"  waypoints: {len(path)}")
        print("  first:", np.round(path[0], 3))
        print("  last :", np.round(path[-1], 3))


if __name__ == "__main__":
    main()
