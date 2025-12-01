import os

import numpy as np
import pytest

if os.environ.get("MJSIM_LIGHT_IMPORT") == "1":
    pytest.skip("Skipping OMPL-dependent tests in light mode", allow_module_level=True)

ompl = pytest.importorskip("ompl")

from mjsim.utils.ompl import qplan


class MockRobot:
    class Info:
        def __init__(self) -> None:
            self.n_joints = 2
            self.joint_limits = np.array([[-1.0, 1.0], [-1.0, 1.0]])

    def __init__(self) -> None:
        self.info = MockRobot.Info()


def test_qplan_returns_path():
    robot = MockRobot()

    def validity(state) -> bool:
        return True  # allow everything within joint limits

    solved, path = qplan(
        robot=robot,
        start=np.array([0.0, 0.0]),
        goal=np.array([0.5, -0.5]),
        validity_check_fn=validity,
        timeout=0.5,
    )
    assert solved
    assert path is not None
    assert path.shape[1] == robot.info.n_joints
