import mujoco as mj
import numpy as np

from mjsim.base.robot import Robot


def _two_link_arm() -> tuple[mj.MjModel, mj.MjData]:
    spec = mj.MjSpec()
    spec.modelname = "mass_matrix_test"
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
    data.qpos[:] = [0.2, -0.3]
    mj.mj_forward(model, data)
    return model, data


def test_robot_mq_uses_dense_mujoco_inertia_matrix() -> None:
    model, data = _two_link_arm()
    robot = Robot(model, data, "arm")

    dense = np.zeros((model.nv, model.nv))
    mj.mju_sym2dense(dense, data.M, model.M_rownnz, model.M_rowadr, model.M_colind)
    dof_indices = np.ravel(robot.info._dof_indxs)

    assert np.allclose(robot.Mq, dense[np.ix_(dof_indices, dof_indices)])
