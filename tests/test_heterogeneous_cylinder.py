"""Validate the concentric material interface and compression response."""

import importlib.util
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest


def example(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "heterogeneous_cylinder", directory / "heterogeneous_cylinder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concentric_mesh_has_solid_core_conforming_bond_and_correct_mass(monkeypatch):
    sim = example(monkeypatch).HeterogeneousCylinderSim()
    assert np.all(sim.rest_volumes > 0)
    # Exact volume of the faceted cylinder, not the slightly larger ideal circle.
    area_factor = sim.sectors / 2 * np.sin(2 * np.pi / sim.sectors)
    volume = area_factor * sim.outer_radius**2 * sim.side
    assert sim.rest_volumes.sum() == pytest.approx(volume)
    assert sim.rest_volumes[~sim.soft_elements].sum() == pytest.approx(
        area_factor * sim.core_radius**2 * sim.side
    )
    assert sim.vertex_masses.sum() == pytest.approx(sim.density * volume)
    assert sim.model.body_mass[sim.body_ids].sum() == pytest.approx(
        sim.density * volume
    )
    regions = []
    for name in ("stiff", "soft"):
        f = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_FLEX, name)
        start, count = sim.model.flex_vertadr[f], sim.model.flex_vertnum[f]
        regions.append(set(sim.model.flex_vertbodyid[start : start + count]))
    radii = np.linalg.norm(sim.points[:, :2], axis=1)
    interface = np.isclose(radii, sim.core_radius)
    assert interface.sum() == sim.sectors * (sim.axial_cells + 1)
    assert regions[0] & regions[1] == set(sim.body_ids[interface])
    assert np.all(np.isin(sim.body_ids[np.isclose(radii, 0)], list(regions[0])))
    assert not regions[1] & set(sim.body_ids[np.isclose(radii, 0)])
    assert sim.model.neq == 0
    # Every unmatched triangular face must be on the physical exterior, never
    # on the material interface or an internal prism diagonal.
    faces = np.sort(
        sim.tetrahedra[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]], axis=2
    )
    faces, counts = np.unique(faces.reshape(-1, 3), axis=0, return_counts=True)
    assert np.all((counts == 1) | (counts == 2))
    boundary = sim.points[faces[counts == 1]]
    bottom = np.all(np.isclose(boundary[:, :, 2], sim.base_height), axis=1)
    top = np.all(np.isclose(boundary[:, :, 2], sim.base_height + sim.side), axis=1)
    side = np.all(
        np.isclose(np.linalg.norm(boundary[:, :, :2], axis=2), sim.outer_radius), axis=1
    )
    assert np.all(bottom | top | side)


def test_concentric_compression_load_sharing_and_recovery(monkeypatch):
    module = example(monkeypatch)
    sim = module.HeterogeneousCylinderSim()
    fixed = sim.positions()[sim.layers[0]].copy()
    strains, stresses = module.run_headless(sim)
    # Equal height and the same plate impose comparable axial shortening.
    # The moduli change stress, rather than producing the cube's layer strains.
    assert strains[0] == pytest.approx(strains[1], abs=0.002)
    assert 4 < stresses[0] / stresses[1] < 6
    np.testing.assert_allclose(sim.positions()[sim.layers[0]], fixed, atol=1e-12)
    sim.reset()
    np.testing.assert_allclose(sim.positions(), sim.points, atol=1e-12)
    np.testing.assert_allclose(sim.axial_stresses(), 0, atol=1e-9)
    assert sim.steps == 0
    assert sim.data.time == 0


def test_stress_readout_matches_uniform_axial_compression(monkeypatch):
    sim = example(monkeypatch).HeterogeneousCylinderSim()
    # An affine deformation has an analytic StVK stress, independent of mesh
    # resolution. Check the stress reconstruction separately from contact.
    stretch = 0.95
    sim.data.xpos[sim.body_ids, 2] = sim.base_height + stretch * (
        sim.points[:, 2] - sim.base_height
    )
    nu = sim.poisson
    young = np.array([sim.stiff_young, sim.soft_young])
    expected = (
        -stretch * young * (1 - nu) / ((1 + nu) * (1 - 2 * nu)) * (stretch**2 - 1) / 2
    )
    np.testing.assert_allclose(sim.axial_stresses(), expected, rtol=1e-12)
    np.testing.assert_allclose(sim.compressive_strains(), 1 - stretch, atol=1e-12)
