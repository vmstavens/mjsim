import mujoco as mj
import numpy as np
import spatialmath as sm

from mjsim.base.robot import Robot


def _two_link_arm() -> tuple[mj.MjModel, mj.MjData]:
    spec = mj.MjSpec()
    spec.modelname = "ik_test"
    spec.option.gravity = [0, 0, -9.82]

    link1 = spec.worldbody.add_body(name="arm/link1", pos=[0, 0, 0.4])
    link1.add_joint(
        name="arm/j1",
        type=mj.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
    )
    link1.add_geom(
        name="arm/g1",
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, 0.3, 0, 0],
        size=[0.03],
        mass=1.0,
    )

    link2 = link1.add_body(name="arm/link2", pos=[0.3, 0, 0])
    link2.add_joint(
        name="arm/j2",
        type=mj.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
    )
    link2.add_geom(
        name="arm/g2",
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, 0.3, 0, 0],
        size=[0.03],
        mass=0.5,
    )
    link2.add_site(name="arm/ee", pos=[0.3, 0, 0])

    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


def _site_pose(model: mj.MjModel, data: mj.MjData, site_name: str) -> sm.SE3:
    site_id = model.site(site_name).id
    return sm.SE3.Rt(
        data.site_xmat[site_id].reshape(3, 3),
        data.site_xpos[site_id],
        check=False,
    )


def test_robot_q_mapping_does_not_mutate_reference_state() -> None:
    model, data = _two_link_arm()
    robot = Robot(model, data, "arm")

    data.qpos[:] = [0.1, 0.2]
    full_qpos = robot.robot_q_to_full_qpos(np.array([0.3, -0.4]))

    np.testing.assert_allclose(data.qpos, [0.1, 0.2])
    np.testing.assert_allclose(full_qpos, [0.3, -0.4])
    np.testing.assert_allclose(robot.full_qpos_to_robot_q(full_qpos), [0.3, -0.4])


def test_solve_ik_reaches_known_site_pose_without_mutating_data() -> None:
    model, data = _two_link_arm()
    robot = Robot(model, data, "arm")

    q_start = np.array([0.0, 0.0])
    q_goal = np.array([0.35, -0.45])

    data.qpos[:] = q_goal
    mj.mj_forward(model, data)
    target = _site_pose(model, data, "arm/ee")

    data.qpos[:] = q_start
    mj.mj_forward(model, data)

    result = robot.solve_ik(
        "ee",
        target,
        q0=q_start,
        orientation_cost=0.0,
        posture_cost=1e-4,
        max_iters=200,
        pos_tol=2e-3,
    )

    assert result.success
    assert result.position_error < 2e-3
    np.testing.assert_allclose(data.qpos, q_start)

    data.qpos[:] = result.qpos
    mj.mj_forward(model, data)
    reached = _site_pose(model, data, "arm/ee")
    assert np.linalg.norm(reached.t - target.t) < 2e-3


def test_ik_wrapper_returns_robot_local_q() -> None:
    model, data = _two_link_arm()
    robot = Robot(model, data, "arm")

    data.qpos[:] = [0.25, -0.2]
    mj.mj_forward(model, data)
    target = _site_pose(model, data, "arm/ee")

    data.qpos[:] = [0.0, 0.0]
    mj.mj_forward(model, data)

    q = robot.ik(
        target,
        q0=np.zeros(2),
        site_names="ee",
        orientation_cost=0.0,
        posture_cost=1e-4,
        max_iterations=200,
        task_position_tolerance=2e-3,
    )

    assert q.shape == (2,)
