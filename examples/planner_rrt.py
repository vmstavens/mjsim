"""
Minimal OMPL RRT-Connect examples for joint- and task-space planning.
"""

from __future__ import annotations

import numpy as np

try:
    from mjsim.utils.ompl import qplan, xplan
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    raise SystemExit(
        "OMPL is not installed. Install `ompl` to run the planner examples."
    )


class MockRobot:
    """Tiny robot stub exposing the fields used by qplan/xplan."""

    class Info:
        def __init__(self) -> None:
            self.n_joints = 2
            self.joint_limits = np.array([[-1.0, 1.0], [-1.0, 1.0]])

    def __init__(self) -> None:
        self.info = MockRobot.Info()


def joint_space_demo() -> None:
    robot = MockRobot()

    def validity_check(state) -> bool:
        # Keep both joints inside limits.
        for i, (lo, hi) in enumerate(robot.info.joint_limits):
            if state[i] < lo or state[i] > hi:
                return False
        return True

    ok, path = qplan(
        robot=robot,
        start=np.array([-0.5, -0.3]),
        goal=np.array([0.5, 0.4]),
        validity_check_fn=validity_check,
        timeout=1.0,
        simplify=True,
    )
    print("Joint-space plan:", "succeeded" if ok else "failed")
    if ok:
        print(f"  path has {len(path)} waypoints")


def task_space_demo() -> None:
    # A trivial validity check that rejects a ball around the origin.
    def validity_check(state) -> bool:
        return (state.getX() ** 2 + state.getY() ** 2 + state.getZ() ** 2) > 0.1

    ok, path = xplan(
        robot=MockRobot(),
        start_pose=np.array([0.5, 0.0, 0.0, 0, 0, 0, 1]),
        goal_pose=np.array([-0.5, 0.0, 0.0, 0, 0, 0, 1]),
        validity_check_fn=validity_check,
        timeout=1.0,
        simplify=True,
    )
    print("Task-space plan:", "succeeded" if ok else "failed")
    if ok:
        print(f"  path has {len(path)} waypoints")


if __name__ == "__main__":
    joint_space_demo()
    task_space_demo()
