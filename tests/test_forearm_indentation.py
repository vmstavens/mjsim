"""Geometry, material masses, bonded supports, and measured indentation."""

import csv
import importlib.util
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest


@pytest.fixture(scope="module")
def example():
    directory = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(directory))
    name = "example_forearm_indentation"
    spec = importlib.util.spec_from_file_location(
        name, directory / "forearm_indentation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves annotations through the module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[name]
    sys.path.remove(str(directory))


@pytest.fixture(scope="module")
def sim(example):
    return example.ForearmSim()


def test_three_tissue_regions_fill_forearm_around_two_bones(example, sim):
    assert np.all(sim.rest_volumes > 0)
    factor = 0.20 * (1 + 0.85 + 0.85**2) / 3
    outer_volume = (
        sim.sectors
        / 2
        * np.sin(2 * np.pi / sim.sectors)
        * np.prod(sim.outer_axes)
        * factor
    )
    assert sim.rest_volumes.sum() + sum(sim.bone_volumes) == pytest.approx(outer_volume)
    masses = (
        np.array([t.density for t in example.TISSUES.values()])[sim.material_ids]
        * sim.rest_volumes
    )
    assert sim.vertex_masses.sum() == pytest.approx(masses.sum())
    np.testing.assert_allclose(sim.model.body_mass[sim.body_ids], sim.vertex_masses)
    for name, volume in zip(example.BONES, sim.bone_volumes):
        assert sim.model.body(name).mass[0] == pytest.approx(
            volume * example.BONE_DENSITY, rel=1e-6
        )
    # No cracks on the tissue interfaces or the split prism faces; unmatched
    # faces can only occur at the outer skin, end caps, or bone cavities.
    faces = np.sort(
        sim.tetrahedra[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]], axis=2
    )
    faces, counts = np.unique(faces.reshape(-1, 3), axis=0, return_counts=True)
    assert np.all((counts == 1) | (counts == 2))
    boundary = faces[counts == 1]
    end = np.all(np.isclose(abs(sim.points[boundary, 1]), 0.10), axis=1)
    outer = np.all(
        boundary % len(sim.section) >= len(sim.section) - sim.sectors, axis=1
    )
    bone = np.logical_or.reduce(
        [np.all(sim.owners[boundary] == i, axis=1) for i in (0, 1)]
    )
    assert np.all(end | outer | bone)


def test_interfaces_share_vertices_and_bones_are_rigid(example, sim):
    regions = []
    for name in example.TISSUES:
        f = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_FLEX, name)
        start, count = sim.model.flex_vertadr[f], sim.model.flex_vertnum[f]
        regions.append(set(sim.model.flex_vertbodyid[start : start + count]))
    expected = sim.sectors * len(sim.longitudinal)
    assert len(regions[0] & regions[1]) == expected
    assert len(regions[1] & regions[2]) == expected
    assert not regions[0] & regions[2]
    for owner, name in enumerate(example.BONES):
        ids = sim.body_ids[sim.owners == owner]
        assert np.all(sim.model.body_dofnum[ids] == 0)
        assert np.all(sim.model.body_parentid[ids] == sim.model.body(name).id)
        assert set(ids) <= regions[0]
    np.testing.assert_array_equal(sim.model.opt.gravity, 0)
    assert sim.model.neq == 0
    assert sim.history[0][3:] == (0, 0)
    # Probe has a genuine clearance, and separate collision-calibrated touch
    # heights account for curved skin at the two measurement locations.
    assert sim.touch_z[0] != pytest.approx(sim.touch_z[1])
    assert sim.data.ncon == 0


def test_indentation_measures_two_sites_and_recovers(example, tmp_path):
    simulation = example.ForearmSim()
    reference = simulation.model.qpos_spring.copy()
    forces = example.run_headless(simulation)
    assert np.all(forces > 0.001)
    assert simulation.minimum_volume_ratio > 0.5
    np.testing.assert_allclose(
        simulation.positions()[simulation.fixed],
        simulation.points[simulation.fixed],
        atol=1e-12,
    )
    np.testing.assert_array_equal(simulation.model.qpos_spring, reference)
    assert (
        np.max(np.linalg.norm(simulation.positions() - simulation.points, axis=1))
        < 0.0002
    )
    for site in range(2):
        rows = [r for r in simulation.history if r[1] == site and r[2] == "Hold"]
        assert len(rows) >= 10
        np.testing.assert_allclose(
            [r[3] for r in rows], simulation.indentation, atol=1e-12
        )
        assert np.std([r[4] for r in rows[-5:]]) < 0.01 * forces[site]
    # The unloaded moving probe must not show a fabricated contact reaction.
    assert all(abs(r[4]) < 0.001 for r in simulation.history if r[1] == -1)
    figure = example.force_figure(simulation)
    assert np.all(figure.linepnt[:2] > 0)
    assert np.max(figure.linedata[0, : 2 * figure.linepnt[0] : 2]) == pytest.approx(
        1000 * simulation.indentation
    )
    path = tmp_path / "indentation.csv"
    simulation.write_csv(path)
    with path.open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == len(simulation.history)
    assert {r["site"] for r in rows} == {*simulation.site_names, "transfer"}
    assert max(float(r["probe_force_z_N"]) for r in rows) > 0
    simulation.reset()
    np.testing.assert_allclose(simulation.positions(), simulation.points, atol=1e-12)
    assert simulation.steps == 0
    assert len(simulation.history) == 1
    assert simulation.probe_force() == 0
