"""
Inverse kinematics demonstration using a stubbed mink backend.

This mirrors the IK test by monkeypatching a lightweight mink stand-in so you
can see the control flow of `Robot.ik` without needing the real Mink bindings
or a MuJoCo model file. The goal pose is reached after a retry.
"""

import types
from typing import Any, List

import numpy as np
import spatialmath as sm

from mjsim.base import robot as robot_module


def _build_stub_robot() -> robot_module.Robot:
    """Construct a Robot instance without calling its __init__."""

    robot = robot_module.Robot.__new__(robot_module.Robot)
    object.__setattr__(
        robot, "_model", types.SimpleNamespace(opt=types.SimpleNamespace(timestep=1.0), nq=3)
    )
    object.__setattr__(robot, "_data", types.SimpleNamespace())
    robot._ik_conf = types.SimpleNamespace(
        q=np.zeros(3),
        update=lambda q: None,
        integrate=lambda vel, dt: q + vel * dt,
    )
    robot._base = 0
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

    # Forward kinematics: map q directly to translation.
    robot.fk = lambda q, site_name, base_frame=None: sm.SE3(q[0], q[1], q[2])  # type: ignore[assignment]
    return robot


def _install_stub_mink() -> None:
    """Monkeypatch a tiny mink interface onto the robot module."""
    stub_namespace = types.SimpleNamespace()

    def solve_ik(conf, tasks: List[Any], *_args, **_kwargs):
        if solve_ik.calls == 0:
            solve_ik.calls += 1
            return np.zeros_like(conf.q)  # force a retry
        frame_task = [t for t in tasks if hasattr(t, "target")][0]
        target_pos = np.array(frame_task.target.t).flatten()
        return target_pos - conf.q

    solve_ik.calls = 0  # type: ignore[attr-defined]

    stub_namespace.solve_ik = solve_ik
    stub_namespace.Configuration = lambda model: None
    stub_namespace.FrameTask = lambda **kwargs: types.SimpleNamespace(
        target=None,
        set_target=lambda target: setattr(types.SimpleNamespace(), "target", target),
    )
    stub_namespace.PostureTask = lambda *_, **__: types.SimpleNamespace(set_target_from_configuration=lambda *_: None)
    stub_namespace.ConfigurationLimit = lambda *_args, **_kwargs: None
    stub_namespace.CollisionAvoidanceLimit = lambda *_args, **_kwargs: None

    robot_module.mink = stub_namespace  # type: ignore[assignment]

    # Ensure conversions are pass-through for spatialmath SE3.
    robot_module.sm_to_smx = lambda T: T  # type: ignore[assignment]
    robot_module.smx_to_sm = lambda T: T  # type: ignore[assignment]


def main() -> None:
    _install_stub_mink()
    robot = _build_stub_robot()

    target = sm.SE3(0.3, 0.0, 0.0)
    q_sol = robot_module.Robot.ik(
        robot,
        T_target=target,
        q0=np.zeros(3),
        site_names="ee",
        tolerance=1e-6,
        task_position_tolerance=1e-3,
        max_iterations=3,
        max_attempts=2,
        verbose=False,
    )

    print("IK target:", target.t, "solution:", q_sol)


if __name__ == "__main__":
    main()
