"""Behavioral regressions for the MuJoCo 4 migration."""

import importlib.util
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from mjsim.utils.mj import set_state
from mjsim.utils.mjs import cable, empty_scene, pipe, replicate


def _example(name):
    path = Path(__file__).resolve().parents[1] / "examples"
    # rod_twist imports the shared example palette.
    import sys

    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_native_rod_torsion_replays_full_state():
    demo = _example("rod_twist")
    model = demo.build_model()
    data = demo.check_replay(model, steps=250)
    assert model.nplugin == 0
    assert model.joint("demo/rod_twist").type == mj.mjtJoint.mjJNT_HINGE
    assert abs(data.joint("demo/rod_twist").qpos[0]) > 0.1


@pytest.mark.parametrize("initial,pins", [("free", 0), ("ball", 1), ("none", 2)])
def test_native_cable_attachment_and_endpoint_sites(initial, pins):
    scene = empty_scene(integrator="discrete", solver="CG", noslip_iterations=0)
    rod = cable(model_name="rope", n_segments=5, length=0.4, initial=initial)
    scene.attach(rod, prefix="test/", frame=scene.worldbody.add_frame(pos=[0, 0, 1]))
    model = scene.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    np.testing.assert_allclose(data.site("test/rope:S_first").xpos, [0, 0, 1])
    np.testing.assert_allclose(data.site("test/rope:S_last").xpos, [0, 0.4, 1])
    vertices = model.flex_vertbodyid[:5]
    assert np.count_nonzero(vertices == 0) == pins
    if pins < 2:
        assert model.joint("test/rope_twist_0").id >= 0
    else:
        assert "test/rope_twist_0" not in [
            model.joint(i).name for i in range(model.njnt)
        ]


def test_curved_rod_reference_is_stress_free():
    theta = np.linspace(0, np.pi / 2, 9)
    points = np.column_stack((0.2 * np.cos(theta), 0.2 * np.sin(theta), np.zeros(9)))
    spec = cable(points=points)
    spec.option.gravity[:] = 0
    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_step(model, data, nstep=50)
    np.testing.assert_allclose(data.qvel, 0, atol=1e-8)
    assert data.time == pytest.approx(0.1)
    assert not np.any(data.warning.number)


def test_plastic_joint_keeps_permanent_set_and_restores():
    model = mj.MjModel.from_xml_string("""
      <mujoco><option gravity="0 0 0" integrator="implicitfast"/>
        <worldbody><body><joint name="spring" type="slide" stiffness="100" springyield="0.01" damping="2"/>
          <geom size=".02" mass="1"/></body></worldbody>
      </mujoco>
    """)
    data = mj.MjData(model)
    data.qfrc_applied[0] = 3
    mj.mj_step(model, data, nstep=300)
    permanent = data.plastic.copy()
    assert permanent[0] > 0.1
    set_state(model, data, plastic=permanent)
    np.testing.assert_array_equal(data.plastic, permanent)
    data.qfrc_applied[:] = 0
    mj.mj_step(model, data, nstep=2000)
    assert data.qpos[0] > 0.1
    with pytest.raises(ValueError, match="plastic has size"):
        set_state(model, data, plastic=[1, 2])
    mj.mj_resetData(model, data)
    np.testing.assert_array_equal(data.plastic, 0)


def test_speculative_contacts_prevent_tunneling():
    def position(enabled):
        spec = empty_scene(
            speculative=enabled,
            add_floor=False,
            gravity=(0, 0, 0),
            timestep=0.01,
            noslip_iterations=0,
        )
        spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_BOX, size=[0.001, 1, 1])
        body = spec.worldbody.add_body(pos=[-0.05, 0, 0])
        body.add_joint(type=mj.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
        body.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.005], mass=0.1)
        model = spec.compile()
        data = mj.MjData(model)
        data.qvel[0] = 10  # 10 cm per step, much larger than the wall.
        mj.mj_step(model, data)
        return data.qpos[0] - 0.05

    assert position(False) > 0
    assert position(True) < 0


def test_replicate_and_pipe_use_native_replicate():
    model = replicate(
        count=4, sep="0.1 0 0", geom_type="sphere", geom_size="0.01"
    ).compile()
    assert model.ngeom == 4
    np.testing.assert_allclose(model.geom_pos[:, 0], [0, 0.1, 0.2, 0.3])
    assert pipe(count=8).compile().ngeom == 8


@pytest.mark.parametrize("threads", [1, 3])
def test_batch_matches_scalar_simulation(threads):
    result = _example("batch_simulation").run(nsim=7, nstep=50, nthread=threads)
    assert result.shape == (7, 1)


def test_legacy_cable_damping_has_actionable_error():
    with pytest.raises(ValueError, match="use damping for Rayleigh"):
        cable(joint_damping=0.1)


def test_speculative_contacts_reject_disabled_nativeccd():
    with pytest.raises(ValueError, match="require nativeccd=True"):
        empty_scene(speculative=True, nativeccd=False)


def test_legacy_cloth_joint_damping_is_rejected():
    from mjsim.utils.mjs import dqo

    with pytest.raises(ValueError, match="use damping for Rayleigh"):
        dqo(
            model_name="cloth",
            prefix="cloth:",
            count=[3, 3, 1],
            size=0.2,
            initial="free",
            stretch=100,
            bend=10,
            shear=50,
            segment_size=0.001,
            segment_geom_type=mj.mjtGeom.mjGEOM_BOX,
            joint_damping=0.1,
        )
