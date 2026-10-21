"""The offline gallery must load, evolve, and keep styling out of dynamics."""

import importlib.util
import math
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
# Load the examples package without importing unrelated robotics dependencies.
spec = importlib.util.spec_from_file_location(
    "physics", EXAMPLES_DIR / "physics" / "__init__.py"
)
package = importlib.util.module_from_spec(spec)
sys.modules["physics"] = package
spec.loader.exec_module(package)

from physics.catalog import EXAMPLES, drive, load_example  # noqa: E402
from physics.palette import FLAME, OFF_WHITE, apply_palette  # noqa: E402


@pytest.mark.parametrize("name", EXAMPLES)
def test_gallery_dynamics_and_palette(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model, data = load_example(name)
    original, original_data = load_example(name, original_colors=True)
    assert np.all(model.geom_matid == -1)
    if model.nflex:
        np.testing.assert_allclose(
            model.flex_rgba[:, :3], np.tile(FLAME[:3], (model.nflex, 1))
        )
    np.testing.assert_array_equal(model.geom_size, original.geom_size)
    np.testing.assert_array_equal(model.body_mass, original.body_mass)
    np.testing.assert_array_equal(model.geom_friction, original.geom_friction)
    assert not np.array_equal(model.geom_rgba, original.geom_rgba)

    # Follow both models through impacts, flex deformation, and driven inputs.
    for _ in range(math.ceil(2 / model.opt.timestep)):
        drive(name, model, data)
        drive(name, original, original_data)
        mj.mj_step(model, data)
        mj.mj_step(original, original_data)
    assert data.time >= 2 - 1e-9
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert not any(warning.number for warning in data.warning)
    np.testing.assert_array_equal(data.qpos, original_data.qpos)
    np.testing.assert_array_equal(data.qvel, original_data.qvel)


def test_marker_roles_and_material_transparency():
    model = mj.MjModel.from_xml_string("""
      <mujoco>
        <asset><material name="glass" rgba="0 0 1 .35"/></asset>
        <worldbody>
          <geom name="payload" size=".1" material="glass"/>
          <site name="actual_frame" size=".01"/>
          <site name="target_frame" type="box" size=".02 .02 .02"/>
          <site name="hidden_sensor" size=".01" rgba="1 0 0 0"/>
        </worldbody>
      </mujoco>
    """)
    apply_palette(model, ("payload",))
    np.testing.assert_allclose(model.geom("payload").rgba, [*FLAME[:3], 0.35])
    np.testing.assert_allclose(model.site("actual_frame").rgba[:3], OFF_WHITE[:3])
    np.testing.assert_allclose(model.site("target_frame").rgba[:3], FLAME[:3])
    assert model.site("hidden_sensor").rgba[3] == 0


@pytest.mark.skipif(
    not hasattr(mj.MjModel, "geom_surfacevel"), reason="carousel requires MuJoCo 3.11+"
)
def test_carousel_contact_transport_and_controls():
    spec = importlib.util.spec_from_file_location(
        "example_conveyor", EXAMPLES_DIR / "conveyor_belt.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sim = module.Sim()
    assert sim.m.nu == 0  # The luggage must be moved by friction, not body actuators.
    initial_positions = sim.d.xpos[sim.bag_ids].copy()
    original_surfaces = sim.m.geom_surfacevel.copy()
    sim.reverse()
    np.testing.assert_array_equal(sim.m.geom_surfacevel, -original_surfaces)
    sim.key_callback(80)  # P stops the surfaces.
    assert not np.any(sim.m.geom_surfacevel)
    sim.key_callback(80)
    np.testing.assert_array_equal(sim.m.geom_surfacevel, -original_surfaces)
    sim.reset()
    np.testing.assert_array_equal(sim.m.geom_surfacevel, original_surfaces)
    sim.run_headless(10000)
    positions = sim.d.xpos[sim.bag_ids]
    assert np.all(np.linalg.norm(positions[:, :2], axis=1) < 2.2)
    assert np.all(positions[2:, 1] - initial_positions[2:, 1] > 2)
    assert not any(warning.number for warning in sim.d.warning)
