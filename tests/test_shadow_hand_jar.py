"""Integration checks for fingertip IK driving a physical hand and passive lid."""

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


def test_ik_tracks_target_through_actuators_and_leaves_lid_physical():
    model, data = load_example("shadow_hand_jar").build_model()
    ik = load_example("shadow_hand_ik").HandIK(model, data)
    target = data.site("rh_ffdistal_tip").xpos.copy() + [0, 0, -0.015]
    ik.set_target("first", target)
    lid = model.joint("lid_hinge")
    data.qvel[lid.dofadr[0]] = 1.0
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    ik.step(0.01)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    for _ in range(300):
        ik.step(0.01)
        for _ in range(10):
            mj.mj_step(model, data)
    assert np.linalg.norm(data.site("rh_ffdistal_tip").xpos - target) < 0.006
    assert data.qpos[lid.qposadr[0]] > 0
    assert abs(data.qvel[lid.dofadr[0]]) < 0.001
    assert np.all(data.ctrl >= model.actuator_ctrlrange[:, 0])
    assert np.all(data.ctrl <= model.actuator_ctrlrange[:, 1])
    assert not np.any(data.warning.number)


def test_ik_reset_and_invalid_targets():
    model, data = load_example("shadow_hand_jar").build_model()
    ik = load_example("shadow_hand_ik").HandIK(model, data)
    with pytest.raises(ValueError):
        ik.set_target("first", [np.nan, 0, 0])
    with pytest.raises(ValueError):
        ik.set_target("unknown", [0, 0, 0])
    with pytest.raises(ValueError):
        ik.step(0)
    model.body("rh_forearm_base").pos[2] += 0.01
    ik.reset()
    for finger, code in load_example("shadow_hand_ik").FINGERS.items():
        idx = model.body(f"{finger}_target").mocapid[0]
        np.testing.assert_allclose(
            data.mocap_pos[idx], data.site(f"rh_{code}distal_tip").xpos
        )


def test_tactile_force_sign_shear_frames_and_sample_hold():
    # A slipping fingertip sphere presses on a plane. Markers deliberately lie
    # outside the contact, demonstrating there is no hard site-volume cutoff.
    sites = "".join(
        f'<site name="rh_ffdistal_{i}" pos="{(i - 8) * 0.001} 0 .01" '
        'size=".0025" quat=".70710678 0 0 .70710678"/>'
        for i in range(17)
    )
    model = mj.MjModel.from_xml_string(f"""<mujoco>
      <option timestep=".001"/>
      <worldbody>
        <geom type="plane" size="1 1 .01"/>
        <body name="rh_ffdistal" pos="0 0 .009">
          <freejoint/><geom type="sphere" size=".01" mass=".1"/>{sites}
        </body>
      </worldbody></mujoco>""")
    data = mj.MjData(model)
    data.qvel[0] = 0.1
    tactile = load_example("shadow_hand_tactile").FingertipTactile(
        model, data, fingers=("ff",)
    )
    assert tactile.update()
    values = tactile.forces["ff"].copy()
    rotations = data.site_xmat[tactile.site_ids["ff"]].reshape(-1, 3, 3)
    world = np.einsum("nij,nj->ni", rotations, values)
    force = np.zeros(6)
    mj.mj_contactForce(model, data, 0, force)
    expected = data.contact[0].frame.reshape(3, 3).T @ force[:3]
    np.testing.assert_allclose(world.sum(axis=0), expected, atol=1e-10)
    assert expected[2] > 0  # Upward normal force on sphere.
    assert expected[0] < 0  # Friction opposes its +x sliding velocity.
    assert np.linalg.norm(values[8]) > np.linalg.norm(values[0])
    assert not np.allclose(values, world)  # Rotated taxel frames matter.
    data.qpos[2] = 0.1  # Remove contact, but hold previous sample for 10 ms.
    data.time = 0.009
    assert not tactile.update()
    np.testing.assert_array_equal(tactile.forces["ff"], values)
    data.time = 0.01
    assert tactile.update()
    np.testing.assert_array_equal(tactile.forces["ff"], 0)
    mj.mj_resetData(model, data)
    assert tactile.update()  # Time rewind rebases sampling.
    assert tactile.sample_time == 0


def test_tactile_layout_and_readings():
    example = load_example("shadow_hand_jar")
    model, data = example.build_model()
    tactile = load_example("shadow_hand_tactile").FingertipTactile(model, data)
    assert tactile.update()
    assert len(tactile.readings()) == 85
    assert all(v.shape == (3,) for v in tactile.readings().values())
    for ids in tactile.site_ids.values():
        np.testing.assert_allclose(model.site_size[ids, 0], 0.0025)
    assert not any(
        model.sensor(i).name.startswith("touch_") for i in range(model.nsensor)
    )
    assert "stf_rh_ffdistal_0" in example.readings(model, data, tactile)
