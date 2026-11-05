"""Check physical tendon steering, base motion, and the supplied OBJ/MTL."""

import importlib.util
import json
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest


@pytest.fixture
def example(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "bronchoscope", directory / "bronchoscope.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supplied_airway_keeps_material_texture_and_scale(example):
    sim = example.BronchoscopeSim(example.default_airway())
    model = sim.model
    geom = model.geom("airway")
    material = int(geom.matid[0])
    assert material >= 0
    np.testing.assert_allclose(model.mat_rgba[material], [0.4, 0.4, 0.4, 1])
    assert model.ntex == 1  # JPEG converted to PNG and loaded through the VFS
    assert np.any(model.mat_texid[material] >= 0)
    assert model.mesh_texcoord.shape[0] > 0
    assert geom.contype[0] == geom.conaffinity[0] == 0
    assert sim.wall_id >= 0
    assert model.flex_rigid[sim.wall_id]
    assert model.nv == 83  # The collision surface adds no degrees of freedom.
    assert 0 < model.nflexelem <= 5000
    # Check the actual compiled mesh in world coordinates, including MuJoCo's
    # mesh recentering/rotation, rather than trusting XML scale attributes.
    vertices = model.mesh_vert @ sim.data.geom_xmat[geom.id].reshape(3, 3).T
    vertices += sim.data.geom_xpos[geom.id]
    np.testing.assert_allclose(
        np.ptp(vertices, axis=0), [0.252699, 0.146496, 0.251257], atol=1e-6
    )
    assert vertices[:, 2].min() == pytest.approx(0.015, abs=1e-6)
    # The visible top is opened below the original sealed cap. Unused OBJ
    # vertices remain in the mesh arrays, so inspect referenced face vertices.
    active = np.unique(model.mesh_face)
    assert 0.260 < vertices[active, 2].max() < 0.261257

    tip = model.site("scope_tip").id
    entry_id = model.site("airway_entry").id
    entry = sim.data.site_xpos[entry_id].copy()
    frame = sim.data.site_xmat[entry_id].reshape(3, 3).copy()
    start = sim.data.site_xpos[tip].copy()
    np.testing.assert_allclose(
        start, entry + example.ENTRY_GAP * frame[:, 2], atol=1e-12
    )
    # A ray through the opened inlet must reach beyond the entire planned move.
    distance = mj.mj_rayMesh(model, sim.data, geom.id, start, -frame[:, 2])
    assert distance > sim.insertion_depth + 0.005
    assert sim.insertion_depth == pytest.approx(example.INSERTION_DEPTH)

    heights = []
    for step in range(2800):
        sim.demo()
        sim.step(10)
        assert sim.wall_contact_stats()[0] == 0  # Clear inlet, not a convex hull.
        if step % 10 == 0:
            heights.append(float(sim.data.joint("insertion").qpos[0]))
    assert np.max(np.diff(heights)) < 1e-7
    assert np.max(np.abs(np.diff(heights))) / 0.1 < 0.0048
    np.testing.assert_allclose(sim.data.ctrl[2:], 0)
    np.testing.assert_allclose(
        sim.data.site_xpos[tip], start - sim.insertion_depth * frame[:, 2], atol=1e-5
    )

    # Check every inserted capsule's actual center against the lumen walls.
    for i in range(model.ngeom):
        if not model.geom(i).name.startswith(("shaft_geom_", "tip_geom_")):
            continue
        point = sim.data.geom_xpos[i]
        if np.dot(entry - point, frame[:, 2]) < 0.004:
            continue
        distances = [
            mj.mj_rayMesh(
                model,
                sim.data,
                geom.id,
                point,
                np.cos(a) * frame[:, 0] + np.sin(a) * frame[:, 1],
            )
            for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ]
        assert min(distances) > model.geom_size[i, 0] + 0.001

    # A pulling tendon must now press against a wall, rather than passing
    # through it. Compare physical motion with contact switched off explicitly.
    contact_types = model.flex_contype.copy(), model.flex_conaffinity.copy()
    deflections = []
    for enabled in (True, False):
        model.flex_contype[:] = contact_types[0] if enabled else 0
        model.flex_conaffinity[:] = contact_types[1] if enabled else 0
        sim.reset()
        sim.data.joint("insertion").qpos[0] = -0.06
        sim.command(-0.06)
        mj.mj_forward(model, sim.data)
        rest = sim.data.site_xpos[tip].copy()
        peak_force, peak_penetration = 0.0, 0.0
        for sample in range(200):
            sim.command(-0.06, tensions=(0.6 * min(1, sample / 100), 0, 0, 0))
            sim.step(10)
            _, force, penetration = sim.wall_contact_stats()
            peak_force = max(peak_force, force)
            peak_penetration = max(peak_penetration, penetration)
        deflections.append(np.linalg.norm(sim.data.site_xpos[tip] - rest))
        if enabled:
            assert peak_force > 0.005
            assert peak_penetration < 0.0002
            assert sim.wall_contact_stats()[0] > 0
        else:
            assert sim.wall_contact_stats()[0] == 0
        sim.command(-0.06)
        sim.step(2000)
        assert np.linalg.norm(sim.data.site_xpos[tip] - rest) < 0.0015
    assert deflections[0] < 0.5 * deflections[1]
    model.flex_contype[:], model.flex_conaffinity[:] = contact_types

    sim.reset()
    np.testing.assert_allclose(sim.data.site_xpos[tip], start, atol=1e-12)


def test_tip_camera_tracks_steering_and_has_a_light(example):
    sim = example.BronchoscopeSim(None)
    model, data = sim.model, sim.data
    camera = model.camera(example.CAMERA_NAME)
    body = model.body("tip_7").id
    light = model.light("tip_light")
    assert camera.bodyid[0] == light.bodyid[0] == body
    assert camera.fovy[0] == 90
    assert model.vis.map.znear * model.stat.extent <= 0.0002
    sim.command(rotation=0.8, tensions=(0.5, 0, 0, 0))
    sim.step(3000)
    forward = -data.cam_xmat[camera.id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(
        forward, -data.xmat[body].reshape(3, 3)[:, 2], atol=1e-12
    )
    tip = data.site_xpos[model.site("scope_tip").id]
    np.testing.assert_allclose(
        data.cam_xpos[camera.id], tip + 0.0032 * forward, atol=1e-12
    )
    assert np.linalg.norm(forward[:2]) > 0.1


def test_priority_airway_clearance_and_surface_error(example):
    import open3d as o3d
    from validate_airway_collision import ray_scene

    reference = json.loads(
        (example.ASSETS / "airways/collision_priority_probes.json").read_text()
    )
    centers = np.array(reference["centers_mm"], dtype=np.float32)
    directions = np.array(reference["directions"], dtype=np.float32)
    collider = o3d.io.read_triangle_mesh(
        str(example.default_collision(example.default_airway()))
    )
    scene = ray_scene(collider)
    clearance = scene.compute_distance(o3d.core.Tensor(centers)).numpy()
    assert len(centers) == 300
    assert np.all(clearance >= 2.4)  # The thin tip fits at every reference center.
    origins = np.repeat(centers, len(directions), axis=0)
    rays = np.column_stack((origins, np.tile(directions, (len(centers), 1))))
    actual = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    expected = np.array(reference["reference_hits_mm"], dtype=float)
    selected = np.isfinite(expected) & (expected <= 20)
    assert np.isfinite(actual[selected]).all()
    assert np.quantile(np.abs(actual[selected] - expected[selected]), 0.95) < 0.4


@pytest.mark.parametrize(
    "tendon,axis,sign", [(0, 0, 1), (1, 0, -1), (2, 1, 1), (3, 1, -1)]
)
def test_tendons_pull_bend_and_release(example, tendon, axis, sign):
    sim = example.BronchoscopeSim(None)
    sim.step(1000)
    tip = sim.model.site("scope_tip").id
    rest = sim.data.site_xpos[tip].copy()
    lengths = sim.data.ten_length.copy()
    tensions = np.zeros(4)
    tensions[tendon] = 0.5
    sim.command(tensions=tensions)
    sim.step(2000)
    delta = sim.data.site_xpos[tip] - rest
    assert sign * delta[axis] > 0.01
    assert abs(delta[1 - axis]) < 1e-5
    assert sim.data.ten_length[tendon] < lengths[tendon] - 0.0005
    assert np.all(sim.data.actuator_force[2:] >= 0)
    sim.command()
    sim.step(2000)
    assert np.linalg.norm(sim.data.site_xpos[tip] - rest) < 0.0015


def test_base_motion_rotation_and_reset(example):
    sim = example.BronchoscopeSim(None)
    tip = sim.model.site("scope_tip").id
    np.testing.assert_allclose(
        sim.data.site_xpos[tip], example.BASE_POSITION - [0, 0, 0.34]
    )
    assert sim.model.geom("tip_geom_0").size[0] < sim.model.geom("shaft_geom_0").size[0]
    sim.command(height=-0.05, rotation=np.pi / 2, tensions=(0.5, 0, 0, 0))
    sim.step(4000)
    assert sim.data.joint("insertion").qpos[0] == pytest.approx(-0.05, abs=0.0002)
    assert sim.data.joint("rotation").qpos[0] == pytest.approx(np.pi / 2, abs=0.005)
    # Rotating the base rotates the local +X steering direction into world +Y.
    assert sim.data.site_xpos[tip, 1] > 0.01
    assert abs(sim.data.site_xpos[tip, 0] - example.BASE_POSITION[0]) < 0.002
    sim.command(1, 10, (-1, 100, 0, 0))
    np.testing.assert_allclose(
        sim.data.ctrl, [0.1, 3.1416, 0, example.MAX_TENSION, 0, 0]
    )
    # Manual insertion must travel beyond the former 100 mm actuator/joint cap.
    sim.command(height=-0.12)
    sim.step(1500)
    assert sim.data.joint("insertion").qpos[0] == pytest.approx(-0.12, abs=0.0002)
    with pytest.raises(ValueError, match="finite"):
        sim.command(tensions=(np.nan, 0, 0, 0))
    sim.reset()
    np.testing.assert_allclose(sim.data.qpos, sim.model.qpos0)
    np.testing.assert_allclose(sim.data.ctrl, 0)
    assert sim.data.time == 0
    assert not np.any(sim.data.warning.number)


def test_missing_texture_falls_back_to_mtl_color(example, tmp_path):
    import xml.etree.ElementTree as ET

    obj = tmp_path / "airway.obj"
    obj.write_text(
        "mtllib airway.mtl\nusemtl wall\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
    )
    (tmp_path / "airway.mtl").write_text(
        "newmtl wall\nKd 0.2 0.3 0.4\nmap_Kd absent.jpeg\n"
    )
    root = ET.fromstring("<mujoco><asset/><worldbody/></mujoco>")
    with pytest.warns(UserWarning, match="Missing MTL texture"):
        assets = example.add_airway(root, obj, 0.001)
    assert not assets
    assert root.find("asset/material").get("rgba") == "0.2 0.3 0.4 1.0"
