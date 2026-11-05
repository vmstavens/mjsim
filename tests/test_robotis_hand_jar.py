"""Physical contact, native tactile sensing, and IK for the Robotis jar copy."""

import importlib.util
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = load_example("robotis_hand_jar")
HandIK = load_example("robotis_hand_ik").HandIK


def test_scene_preserves_native_mesh_sensors_and_passive_lid():
    model, data = example.build_model()
    tactile = example.FingertipTactile(model, data)
    assert model.nu == 20
    assert model.njnt == 21
    assert model.nmocap == 5  # IK handles only; upstream test probes are removed.
    assert model.nsensor == 45 + 40 + 4
    assert data.ncon == 0
    np.testing.assert_array_equal(tactile.forces, np.zeros((5, 3, 3)))
    for finger in range(5):
        body = model.body(f"finger_{finger}_link_3_right").id
        geoms = np.flatnonzero(model.geom_bodyid == body)
        assert len(geoms) == 1
        assert model.geom_type[geoms[0]] == mj.mjtGeom.mjGEOM_MESH
    for site in tactile.site_ids.ravel():
        sensor = model.sensor(model.site(site).name)
        assert sensor.type[0] == mj.mjtSensor.mjSENS_TOUCH
        assert sensor.objid[0] == site
    lid = model.joint("lid_hinge").id
    assert lid not in model.actuator_trnid[:, 0]
    assert not model.jnt_limited[lid]
    assert model.dof_damping[model.jnt_dofadr[lid]] == 0.02
    copied = example.readings(model, data)
    copied["taxel_0_0_0"][:] = 999
    assert data.sensor("taxel_0_0_0").data[0] == 0


@pytest.mark.parametrize("cell", range(9))
def test_all_native_taxels_localize_contact_and_clear_after_release(cell):
    # Temporary test fixtures press the unchanged mesh at each grid cell.
    # The shipped jar scene contains no probes and requires real object contact.
    spec = mj.MjSpec.from_file(str(example.SCENE))
    for finger in range(5):
        body = spec.worldbody.add_body(
            name=f"probe_{finger}", mocap=True, pos=[0, 0, 1]
        )
        body.add_geom(
            type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.0025, 0, 0], contype=2, conaffinity=4
        )
    model = spec.compile()
    data = mj.MjData(model)
    tactile = example.FingertipTactile(model, data)
    mj.mj_forward(model, data)
    tips = [model.geom(f"finger_{f}_link_3_right_shell").id for f in range(5)]
    for _ in range(100):
        for finger, tip in enumerate(tips):
            site = tactile.site_ids[finger, cell // 3, cell % 3]
            normal = data.site_xmat[site].reshape(3, 3)[:, 2 if finger == 0 else 0]
            origin = data.site_xpos[site] + normal * 0.03
            distance = mj.mj_rayMesh(model, data, tip, origin, -normal)
            assert distance > 0
            probe = model.body(f"probe_{finger}").mocapid[0]
            data.mocap_pos[probe] = origin + normal * (0.002 - distance)
        mj.mj_step(model, data)
    mj.mj_forward(model, data)
    forces = tactile.forces
    assert (forces[:, cell // 3, cell % 3] > 0.01).all()
    totals = np.zeros(5)
    wrench = np.zeros(6)
    for idx, contact in enumerate(data.contact):
        mj.mj_contactForce(model, data, idx, wrench)
        for finger, tip in enumerate(tips):
            if tip in (contact.geom1, contact.geom2):
                totals[finger] += wrench[0]
    np.testing.assert_allclose(forces.sum(axis=(1, 2)), totals, rtol=1e-6, atol=1e-8)
    forces[:, cell // 3, cell % 3] = 0
    np.testing.assert_allclose(forces, 0, atol=1e-8)
    for finger in range(5):
        data.mocap_pos[model.body(f"probe_{finger}").mocapid[0]] = [1, 0, 1]
    for _ in range(100):
        mj.mj_step(model, data)
    mj.mj_forward(model, data)
    np.testing.assert_allclose(tactile.forces, 0, atol=1e-8)
    assert not np.any(data.warning.number)


def test_curl_contacts_real_lid_and_open_releases_it():
    model, data = example.build_model()
    tactile = example.FingertipTactile(model, data)
    example.curl_fingers(model, data, 1.0)
    for _ in range(100):
        example.step(model, data)
    assert (tactile.forces[1:].sum(axis=(1, 2)) > 0.1).all()
    lid = model.geom("lid_surface").id
    for finger in range(1, 5):
        tip = model.geom(f"finger_{finger}_link_3_right_shell").id
        assert any({c.geom1, c.geom2} == {lid, tip} for c in data.contact)
    example.curl_fingers(model, data, -1.0)
    for _ in range(100):
        example.step(model, data)
    np.testing.assert_allclose(tactile.forces, 0, atol=1e-8)
    assert not np.any(data.warning.number)


def test_ik_tracks_through_actuators_while_lid_remains_physical():
    model, data = example.build_model()
    ik = HandIK(model, data)
    target = data.site("taxel_1_1_1").xpos.copy() + [0, 0, -0.015]
    ik.set_target("first", target)
    lid = model.joint("lid_hinge")
    data.qvel[lid.dofadr[0]] = 1.0
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    ik.step(0.01)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    for _ in range(300):
        example.step(model, data, ik)
    assert np.linalg.norm(data.site("taxel_1_1_1").xpos - target) < 0.006
    assert data.qpos[lid.qposadr[0]] > 0
    assert abs(data.qvel[lid.dofadr[0]]) < 0.001
    assert np.all(data.ctrl >= model.actuator_ctrlrange[:, 0])
    assert np.all(data.ctrl <= model.actuator_ctrlrange[:, 1])
    assert not np.any(data.warning.number)


def test_reset_targets_and_bounded_manual_controls():
    model, data = example.build_model()
    ik = HandIK(model, data)
    for finger, position in (("unknown", [0, 0, 0]), ("first", [np.nan, 0, 0])):
        with pytest.raises(ValueError):
            ik.set_target(finger, position)
    with pytest.raises(ValueError):
        ik.step(0)
    model.body("hand_base_right").pos[2] += 0.01
    ik.reset()
    for code, finger in enumerate(("thumb", "first", "middle", "ring", "little")):
        target = model.body(f"{finger}_target").mocapid[0]
        np.testing.assert_allclose(
            data.mocap_pos[target], data.site(f"taxel_{code}_1_1").xpos
        )
    example.curl_fingers(model, data, 100)
    assert data.ctrl[model.actuator("finger_0_joint_1_right").id] < 0
    assert np.all(data.ctrl >= model.actuator_ctrlrange[:, 0])
    assert np.all(data.ctrl <= model.actuator_ctrlrange[:, 1])
    example.curl_fingers(model, data, -100)
    np.testing.assert_allclose(data.ctrl, 0)
