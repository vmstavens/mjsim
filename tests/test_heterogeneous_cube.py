"""Check the bonded material mesh and its compression/recovery response."""

import importlib.util
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest


def example():
    path = Path(__file__).resolve().parents[1] / "examples" / "heterogeneous_cube.py"
    spec = importlib.util.spec_from_file_location("heterogeneous_cube", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mesh_has_conforming_bond_and_correct_mass():
    sim = example().HeterogeneousCubeSim()
    assert np.all(sim.rest_volumes > 0)
    assert sim.rest_volumes.sum() == pytest.approx(sim.side**3)
    assert sim.vertex_masses.sum() == pytest.approx(sim.density * sim.side**3)
    assert sim.model.body_mass[sim.body_ids].sum() == pytest.approx(
        sim.density * sim.side**3
    )
    # Both flexes must reference the SAME 25 interface bodies, not coincident
    # but disconnected vertices. No equality constraint is needed for bonding.
    regions = []
    for name in ("stiff", "soft"):
        f = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_FLEX, name)
        start = sim.model.flex_vertadr[f]
        count = sim.model.flex_vertnum[f]
        regions.append(set(sim.model.flex_vertbodyid[start : start + count]))
    assert regions[0] & regions[1] == set(sim.body_ids[sim.layers[1]])
    assert sim.model.neq == 0
    # Every internal triangular face is shared twice; only the cube's exterior
    # can have a one-sided face (including across the material interface).
    faces = np.sort(
        sim.tetrahedra[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]], axis=2
    )
    _, counts = np.unique(faces.reshape(-1, 3), axis=0, return_counts=True)
    assert np.all((counts == 1) | (counts == 2))
    assert np.count_nonzero(counts == 1) == 6 * sim.cells**2 * 2


def test_soft_layer_compresses_more_and_recovers():
    module = example()
    sim = module.HeterogeneousCubeSim()
    fixed = sim.positions()[sim.layers[0]].copy()
    hold = module.run_headless(sim)
    assert hold[1] > 4 * hold[0]
    np.testing.assert_allclose(sim.positions()[sim.layers[0]], fixed, atol=1e-12)
    # Recovery within 0.1 mm, allowing small residual elastic vibrations.
    assert np.max(np.abs(sim.positions() - sim.points)) < 1e-4
    sim.reset()
    assert sim.steps == 0
    assert sim.data.time == 0
    np.testing.assert_allclose(sim.positions(), sim.points, atol=1e-12)


def test_equal_stiffness_control_does_not_show_fivefold_contrast():
    sim = example().HeterogeneousCubeSim

    class UniformCube(sim):
        soft_young = sim.stiff_young

    control = UniformCube()
    for _ in range(round(7 / control.model.opt.timestep)):
        control.step()
    stiff, soft = control.compressive_strains()
    # Clamping affects lateral expansion of the bottom layer; equal moduli
    # need not yield exactly equal strains, but must remove the large contrast.
    assert stiff > 0
    assert 0.8 < soft / stiff < 1.5
    assert not np.any(control.data.warning.number)
