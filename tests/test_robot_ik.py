import os
from typing import Any, List

import numpy as np
import pytest
import spatialmath as sm

if os.environ.get("MJSIM_LIGHT_IMPORT") == "1":
    pytest.skip("Skipping IK tests that depend on mink/mujoco in light mode", allow_module_level=True)

import mjsim.base.robot as robot_module


class _StubConfig:
    """Minimal mink.Configuration stand-in."""

    def __init__(self, q: np.ndarray):
        self.q = q

    def update(self, q: np.ndarray) -> None:
        self.q = q

    def integrate(self, vel: np.ndarray, dt: float) -> np.ndarray:
        return self.q + vel * dt


class _StubFrameTask:
    """Lightweight frame task container used for testing."""

    def __init__(self, frame_name: str, frame_type: str, **_: Any) -> None:
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.target = None

    def set_target(self, target: Any) -> None:
        self.target = target


class _StubPostureTask:
    """Lightweight posture task container used for testing."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.target = None

    def set_target_from_configuration(self, conf: _StubConfig) -> None:
        self.target = conf.q


class _StubCollisionLimit:
    """Placeholder collision limit."""

    def __init__(self, **_: Any) -> None:
        pass


def test_ik_multiple_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """IK should converge after re-running across attempts."""

    # Monkeypatch Mink bindings used inside Robot.ik
    stub_namespace = types.SimpleNamespace()

    def solve_ik(conf, tasks: List[Any], *_args, **_kwargs):
        """Return velocities, using a no-op on the first call to force a retry."""
        # First attempt does nothing, second moves toward target
        if solve_ik.calls == 0:
            solve_ik.calls += 1
            return np.zeros_like(conf.q)
        # Move directly to the task target position for convergence
        frame_task = [t for t in tasks if isinstance(t, _StubFrameTask)][0]
        target_pos = np.array(frame_task.target.t).flatten()
        return target_pos - conf.q

    solve_ik.calls = 0  # type: ignore[attr-defined]

    stub_namespace.solve_ik = solve_ik
    stub_namespace.Configuration = lambda model: _StubConfig(
        np.zeros(model.nq)
    )
    stub_namespace.FrameTask = _StubFrameTask
    stub_namespace.PostureTask = _StubPostureTask
    stub_namespace.ConfigurationLimit = lambda *_args, **_kwargs: None
    stub_namespace.CollisionAvoidanceLimit = _StubCollisionLimit
    monkeypatch.setattr(robot_module, "mink", stub_namespace)

    # Bypass mink conversions so we can pass SpatialMath directly.
    monkeypatch.setattr(robot_module, "sm_to_smx", lambda T: T)
    monkeypatch.setattr(robot_module, "smx_to_sm", lambda T: T)

    # Build a bare Robot instance without running __init__.
    robot = robot_module.Robot.__new__(robot_module.Robot)
    robot.model = types.SimpleNamespace(opt=types.SimpleNamespace(timestep=1.0), nq=3)
    robot.data = types.SimpleNamespace()
    robot._ik_conf = _StubConfig(np.zeros(3))
    robot._base = 0

    # Minimal info/state needed by ik()
    robot._info = types.SimpleNamespace(
        site_names=["ee"],
        joint_indxs=np.array([0, 1, 2]),
        _joint_ids=[0, 1, 2],
        n_joints=3,
        body_ids=[0],
        site_ids=[0],
        _actuator_ids=[],
        _dof_indxs=[0, 1, 2],
        actuator_limits=[-1, 1],
    )

    # Stub forward kinematics to reflect q directly into a pose translation.
    def fk_stub(q, site_name, base_frame=None):
        return sm.SE3(q[0], q[1], q[2])

    robot.fk = fk_stub  # type: ignore[assignment]

    target = sm.SE3(0.5, 0.0, 0.0)
    q_sol = robot_module.Robot.ik(
        robot,
        T_target=target,
        q0=np.zeros(3),
        site_names="ee",
        tolerance=1e-6,
        task_position_tolerance=1e-3,
        max_iterations=3,
        max_attempts=2,
    )

    assert np.allclose(q_sol[0], target.t[0], atol=1e-3)
    assert solve_ik.calls == 2  # first attempt was no-op, second moved to target
