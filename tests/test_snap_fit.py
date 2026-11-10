"""Apply external test loads to the V-clip model."""

import importlib.util
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def example_import_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))


def load_model():
    path = Path(__file__).resolve().parents[1] / "examples" / "snap_fit.py"
    module_spec = importlib.util.spec_from_file_location("simple_snapfit", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = module.build_spec()
    assert isinstance(spec, mj.MjSpec)
    model = spec.compile()
    assert model.nu == 0
    return model, mj.MjData(model)


def run(model, data, seconds, press=0.0, torque=0.0):
    """Emulate viewer perturbations; none of these loads are in the example."""
    tab = model.body("press_tab").id
    lid = model.body("lid").id
    hinge = model.joint("lid_hinge").dofadr[0]
    initial_force = data.xfrc_applied[tab, 0]
    initial_torque = data.qfrc_applied[hinge]
    for i in range(round(seconds / model.opt.timestep)):
        blend = -np.expm1(-(i + 1) * model.opt.timestep / 0.05)
        direction = data.xmat[lid].reshape(3, 3)[:, 0]
        data.xfrc_applied[tab, :3] = (
            initial_force + blend * (press - initial_force)
        ) * direction
        data.qfrc_applied[hinge] = initial_torque + blend * (torque - initial_torque)
        mj.mj_step(model, data)
        assert not np.any(data.warning.number)


def test_external_press_releases_lid():
    model, data = load_model()
    run(model, data, 1)
    run(model, data, 20, torque=0.035)
    assert data.qpos[0] < 0.1
    run(model, data, 4, press=-0.15, torque=0.035)
    assert data.qpos[0] > 0.5


def test_squeezing_removes_pressure_against_right_wall():
    model, data = load_model()
    wall = model.geom("box_right").id

    def wall_force():
        force = np.zeros(6)
        normal = 0.0
        for i, contact in enumerate(data.contact):
            if wall in contact.geom:
                mj.mj_contactForce(model, data, i, force)
                normal += force[0]
        return normal

    peak = 0.0
    for _ in range(4000):
        mj.mj_step(model, data)
        peak = max(peak, wall_force())
        assert not np.any(data.warning.number)
    assert peak < 2.0, "Initial preload caused an excessive contact impulse"
    assert 0.01 < wall_force() < 0.1
    run(model, data, 2, press=-0.15)
    assert wall_force() < 0.001


def test_finger_load_cell_reads_contact_force_and_zero_when_clear():
    path = Path(__file__).resolve().parents[1] / "examples" / "snap_fit.py"
    module_spec = importlib.util.spec_from_file_location("finger_snapfit", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.build_spec().compile()
    data = mj.MjData(model)
    assert model.nmocap == 1
    for _ in range(4000):
        mj.mj_step(model, data)
    np.testing.assert_allclose(module.finger_force(model, data), 0, atol=1e-8)

    mocap = model.body("finger_mocap").mocapid[0]
    initial_x = data.mocap_pos[mocap, 0]
    for i in range(4000):
        fraction = min((i + 1) / 2000, 1)
        data.mocap_pos[mocap, 0] = initial_x + fraction * (0.128 - initial_x)
        mj.mj_step(model, data)
        assert not np.any(data.warning.number)

    # Check the load cell against independently summed fingertip contacts.
    fingertip = model.geom("fingertip").id
    contact_force = np.zeros(3)
    wrench = np.zeros(6)
    for i, contact in enumerate(data.contact):
        if fingertip in contact.geom:
            mj.mj_contactForce(model, data, i, wrench)
            sign = 1 if contact.geom[1] == fingertip else -1
            contact_force += sign * contact.frame.reshape(3, 3).T @ wrench[:3]
    measured = module.finger_force(model, data)
    assert 0.01 < np.linalg.norm(measured) < 1.0
    np.testing.assert_allclose(measured, -contact_force, atol=1e-7)


def test_opening_plan_releases_clip_and_opens_without_instability():
    path = Path(__file__).resolve().parents[1] / "examples" / "snap_fit.py"
    module_spec = importlib.util.spec_from_file_location("planned_snapfit", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.build_spec().compile()
    data = mj.MjData(model)
    plan = module.OpeningPlan(model)
    plan.start(data)
    peak = 0.0
    max_weld_gap = 0.0
    mount = model.site("finger_mount_site").id
    target = model.site("finger_target").id
    wall = model.geom("box_right").id
    wrench = np.zeros(6)
    for i in range(20000):
        plan.update(data)
        mj.mj_step(model, data)
        peak = max(peak, float(np.linalg.norm(module.finger_force(model, data))))
        max_weld_gap = max(
            max_weld_gap,
            float(np.linalg.norm(data.site_xpos[mount] - data.site_xpos[target])),
        )
        assert not np.any(data.warning.number)
        assert abs(data.qfrc_applied[plan.dof]) <= plan.torque_limit
        if i == 7800:  # Press has settled; opening torque has not started.
            normal = 0.0
            for j, contact in enumerate(data.contact):
                if wall in contact.geom:
                    mj.mj_contactForce(model, data, j, wrench)
                    normal += wrench[0]
            assert normal < 0.001
            assert data.qpos[plan.qpos] < 0.01
    assert plan.phase == "Open"
    assert data.qpos[plan.qpos] > 0.9
    assert 0.01 < peak < 0.2
    assert max_weld_gap < 0.0003
    plan.stop(data)
    assert data.qfrc_applied[plan.dof] == 0
    pose = data.mocap_pos.copy()
    plan.update(data)
    np.testing.assert_array_equal(data.mocap_pos, pose)
    plan.start(data)
    assert data.time == 0
    assert data.qpos[plan.qpos] == 0
    assert plan.phase == "Settling"
