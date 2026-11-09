"""Check stiffness and shape retention through loads, impacts, and pusher cycles."""

import importlib.util
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest


@pytest.mark.skipif(
    not hasattr(mj.MjsBody, "make_flex"), reason="requires make_flex API"
)
def test_flex_catalogue_stays_stable(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("example_mk_flex", examples / "mk_flex.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sim = module.Sim()

    plate_ids = sim.flex_body_ids["cloth_2d"]
    torus_ids = sim.flex_body_ids["custom_mesh"]
    initial_torus = sim.d.xpos[torus_ids].copy()
    initial_torus -= initial_torus.mean(axis=0)
    initial_thickness = np.linalg.svd(initial_torus, compute_uv=False)[-1]

    # Twist the free plate with balanced opposing 2 N corner loads. This
    # excites bending without a net force or torque, before it hits the floor.
    plate = sim.d.xpos[plate_ids].copy()
    xy = plate[:, :2] - plate[:, :2].mean(axis=0)
    corners = np.flatnonzero(np.all(np.isclose(np.abs(xy), 0.1225), axis=1))
    assert len(corners) == 4
    for corner in corners:
        sim.d.xfrc_applied[plate_ids[corner], 2] = 2 * np.sign(np.prod(xy[corner]))
    for _ in range(200):
        sim.step()
    loaded_plate = sim.d.xpos[plate_ids]
    assert np.ptp(loaded_plate[:, 2]) < 0.002  # Less than 2 mm of bending.
    distances = np.linalg.norm(loaded_plate[:, None] - loaded_plate[None, :], axis=-1)
    original_distances = np.linalg.norm(plate[:, None] - plate[None, :], axis=-1)
    nonzero = original_distances > 0
    assert np.max(np.abs(distances[nonzero] / original_distances[nonzero] - 1)) < 0.005
    sim.d.xfrc_applied[:] = 0

    # MuJoCo automatically resets unstable simulations, so finite state alone
    # is insufficient: also check warnings and that time advances all the way.
    steps = 12000
    for _ in range(steps - 200):
        sim.step()
    assert sim.d.time == pytest.approx(steps * sim.m.opt.timestep)
    assert not np.any(sim.d.warning.number)
    assert np.isfinite(sim.d.qpos).all()
    assert np.isfinite(sim.d.qvel).all()

    # A flattened torus can remain numerically stable. Check its tube thickness
    # and shape after landing, allowing arbitrary translation and rotation.
    torus = sim.d.xpos[torus_ids].copy()
    torus -= torus.mean(axis=0)
    assert np.linalg.svd(torus, compute_uv=False)[-1] > 0.8 * initial_thickness
    u, _, vt = np.linalg.svd(initial_torus.T @ torus)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    shape_error = np.sqrt(np.mean(np.sum((initial_torus @ rotation - torus) ** 2, axis=1)))
    assert shape_error < 0.003  # Less than 3 mm RMS deformation of the ring.
