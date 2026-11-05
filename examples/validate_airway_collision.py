"""Measure airway simplification error, contact response, and simulation timing.

uv run python examples/validate_airway_collision.py --meshes \
    /tmp/airway-decimation/hollow_airway_collision_50000.obj \
    /tmp/airway-decimation/hollow_airway_collision_100000.obj \
    --report /tmp/airway-decimation/validation.json

No graphics context is needed. Surface-distance measurements are sampled, not
certified global bounds. Dynamic checks exercise the upper trachea only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco as mj
import numpy as np
import open3d as o3d
from bronchoscope import BronchoscopeSim, default_airway


def ray_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    # Geometry only; texture/material slots have no role in these queries.
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(mesh.triangles), dtype=o3d.core.Dtype.Int64),
    )
    scene.add_triangles(tensor_mesh)
    return scene


def distances_summary(values):
    return {
        "rms_mm": float(np.sqrt(np.mean(values**2))),
        "p95_mm": float(np.quantile(values, 0.95)),
        "p99_mm": float(np.quantile(values, 0.99)),
        "max_sample_mm": float(values.max()),
    }


def geometry_comparison(source, paths, samples, regions=None):
    original = o3d.io.read_triangle_mesh(str(source))
    vertices, triangles = np.asarray(original.vertices), np.asarray(original.triangles)
    cut_z = vertices[:, 2].max() - 5.0
    triangles = triangles[np.all(vertices[triangles, 2] <= cut_z, axis=1)]
    original.triangles = o3d.utility.Vector3iVector(triangles)
    original.remove_unreferenced_vertices()
    original_scene = ray_scene(original)
    inner_scene = priority_mesh = inner_mesh = priority_faces = None
    if regions is not None:
        region_points, region_faces = regions["vertices"], regions["triangles"]
        inner_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(region_points),
            o3d.utility.Vector3iVector(region_faces),
        )
        inner_scene = ray_scene(inner_mesh)
        priority_faces = regions["reachable"][region_faces].all(axis=1)
        priority_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(region_points),
            o3d.utility.Vector3iVector(region_faces[priority_faces]),
        )
    o3d.utility.random.seed(42)
    points = np.asarray(original.sample_points_uniformly(samples).points).astype(
        np.float32
    )
    results = []
    for path in paths:
        reduced = o3d.io.read_triangle_mesh(str(path))
        reduced_scene = ray_scene(reduced)
        o3d.utility.random.seed(42)
        reduced_points = np.asarray(
            reduced.sample_points_uniformly(samples).points
        ).astype(np.float32)
        result = {
            "mesh": path.name,
            "triangles": len(reduced.triangles),
            "vertices": len(reduced.vertices),
            "file_bytes": path.stat().st_size,
            "original_to_collision": distances_summary(
                reduced_scene.compute_distance(o3d.core.Tensor(points)).numpy()
            ),
            "collision_to_original": distances_summary(
                original_scene.compute_distance(o3d.core.Tensor(reduced_points)).numpy()
            ),
        }
        if regions is not None:
            o3d.utility.random.seed(42)
            priority_points = np.asarray(
                priority_mesh.sample_points_uniformly(samples).points
            ).astype(np.float32)
            closest = inner_scene.compute_closest_points(
                o3d.core.Tensor(reduced_points)
            )
            selected = priority_faces[closest["primitive_ids"].numpy()]
            back = inner_scene.compute_distance(
                o3d.core.Tensor(reduced_points[selected])
            ).numpy()
            centroids = (
                np.asarray(reduced.vertices)[np.asarray(reduced.triangles)]
                .mean(axis=1)
                .astype(np.float32)
            )
            face_classes = inner_scene.compute_closest_points(
                o3d.core.Tensor(centroids)
            )["primitive_ids"].numpy()
            directions = np.random.default_rng(42).normal(size=(16, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            origins = np.repeat(regions["probe_centers"], len(directions), axis=0)
            rays = o3d.core.Tensor(
                np.column_stack(
                    (origins, np.tile(directions, (len(regions["probe_centers"]), 1)))
                ),
                dtype=o3d.core.Dtype.Float32,
            )
            exact = inner_scene.cast_rays(rays)["t_hit"].numpy()
            measured = reduced_scene.cast_rays(rays)["t_hit"].numpy()
            local = np.isfinite(exact) & (exact <= 20)
            matched = local & np.isfinite(measured)
            center_clearance = reduced_scene.compute_distance(
                o3d.core.Tensor(regions["probe_centers"].astype(np.float32))
            ).numpy()
            result["priority_surface"] = {
                "reference": "Connected inner wall with estimated diameter >= priority threshold",
                "collision_triangles_nearest_priority": int(
                    priority_faces[face_classes].sum()
                ),
                "reference_to_collision": distances_summary(
                    reduced_scene.compute_distance(
                        o3d.core.Tensor(priority_points)
                    ).numpy()
                ),
                "collision_to_reference": distances_summary(back),
                "probe_rays_expected": int(local.sum()),
                "probe_rays_hit": int(matched.sum()),
                "probe_centers": len(center_clearance),
                "probe_centers_clear_for_tip": int((center_clearance >= 2.4).sum()),
                "probe_centers_clear_for_shaft": int((center_clearance >= 3.0).sum()),
                "minimum_probe_clearance_mm": float(center_clearance.min()),
                "probe_first_hit_error": distances_summary(
                    np.abs(exact[matched] - measured[matched])
                ),
            }
        results.append(result)
        print(json.dumps({"geometry": result}), flush=True)
    return results


def seated(sim):
    sim.reset()
    sim.data.joint("insertion").qpos[0] = -0.06
    sim.command(height=-0.06)
    mj.mj_forward(sim.model, sim.data)
    sim.step(500)


def exercise(sim):
    """Ramp each tendon, hold contact, then release; compare to a free rod."""
    tip_id = sim.model.site("scope_tip").id
    frame = sim.data.xmat[sim.model.body("base").id].reshape(3, 3).copy()
    directions = []
    for tendon in range(4):
        seated(sim)
        rest = sim.data.site_xpos[tip_id].copy()
        peak_force, peak_penetration, peak_contacts = 0.0, 0.0, 0
        start = time.perf_counter()
        # Controls at 100 Hz; physics remains at the model's 1 ms timestep.
        for sample in range(200):
            tensions = np.zeros(4)
            tensions[tendon] = 0.6 * min(1.0, sample / 100)
            sim.command(-0.06, tensions=tensions)
            sim.step(10)
            count, force, penetration = sim.wall_contact_stats()
            peak_contacts = max(peak_contacts, count)
            peak_force = max(peak_force, force)
            peak_penetration = max(peak_penetration, penetration)
        wall_time = time.perf_counter() - start
        deflection = frame.T @ (sim.data.site_xpos[tip_id] - rest)
        held = sim.wall_contact_stats()
        sim.command(-0.06)
        sim.step(2000)
        item = {
            "tendon": tendon,
            "local_tip_deflection_mm": (1000 * deflection).tolist(),
            "peak_contacts": peak_contacts,
            "peak_normal_force_N": peak_force,
            "peak_penetration_mm": 1000 * peak_penetration,
            "held_contacts": held[0],
            "held_normal_force_N": held[1],
            "release_error_mm": float(
                1000 * np.linalg.norm(sim.data.site_xpos[tip_id] - rest)
            ),
            "simulated_seconds": 2.0,
            "wall_seconds": wall_time,
            "real_time_factor": 2.0 / wall_time,
        }
        directions.append(item)
        print(json.dumps({"contact": item}), flush=True)
    return directions


def tracheal_clearance(sim, reduced_scene):
    """Compare original/reduced radial first-hit distances along the entry path."""
    entry = sim.data.site_xpos[sim.model.site("airway_entry").id]
    frame = sim.data.site_xmat[sim.model.site("airway_entry").id].reshape(3, 3)
    # Convert world coordinates back into the original OBJ's millimetres.
    # A visual mesh geom is recentered by MuJoCo; its asset frame is available
    # via mesh_pos/mesh_quat. Recover the original source-to-world transform.
    geom = sim.model.geom(sim.airway_id)
    mesh_id = int(geom.dataid[0])
    asset_rotation = np.empty(9)
    mj.mju_quat2Mat(asset_rotation, sim.model.mesh_quat[mesh_id])
    world_rotation = sim.data.geom_xmat[geom.id].reshape(3, 3)
    source_rotation = world_rotation @ asset_rotation.reshape(3, 3).T
    offset = sim.data.geom_xpos[geom.id] - source_rotation @ sim.model.mesh_pos[mesh_id]
    rays, exact = [], []
    for depth in np.linspace(0.004, 0.048, 45):
        point = entry - depth * frame[:, 2]
        for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
            direction = np.cos(angle) * frame[:, 0] + np.sin(angle) * frame[:, 1]
            exact.append(
                mj.mj_rayMesh(sim.model, sim.data, sim.airway_id, point, direction)
                * 1000
            )
            origin = source_rotation.T @ (point - offset) * 1000
            vector = source_rotation.T @ direction
            rays.append([*origin, *vector])
    hits = reduced_scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))[
        "t_hit"
    ].numpy()
    exact = np.asarray(exact)
    valid = np.isfinite(hits) & (exact >= 0)
    return {
        "ray_count": len(rays),
        "valid_rays": int(valid.sum()),
        "minimum_collision_radius_mm": float(hits[valid].min()),
        "radial_error": distances_summary(np.abs(hits[valid] - exact[valid])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_airway())
    parser.add_argument("--meshes", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument(
        "--regions",
        type=Path,
        help="inner-wall priority NPZ; report regional errors and branch probes",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("/tmp/airway-collision-validation.json")
    )
    args = parser.parse_args()
    regions = np.load(args.regions) if args.regions else None
    results = geometry_comparison(args.source, args.meshes, args.samples, regions)
    start = time.perf_counter()
    free = BronchoscopeSim(args.source, collisions=False)
    baseline = {
        "compile_seconds": time.perf_counter() - start,
        "directions": exercise(free),
    }
    for result, path in zip(results, args.meshes):
        print(f"Testing {path.name}", flush=True)
        start = time.perf_counter()
        sim = BronchoscopeSim(args.source, collision=path)
        result["compile_seconds"] = time.perf_counter() - start
        result["clearance"] = tracheal_clearance(
            sim, ray_scene(o3d.io.read_triangle_mesh(str(path)))
        )
        result["directions"] = exercise(sim)
        assert result["clearance"]["valid_rays"] == result["clearance"]["ray_count"]
        assert all(
            direction["held_normal_force_N"] > 0 for direction in result["directions"]
        )
        assert all(
            direction["peak_penetration_mm"] < 0.2 for direction in result["directions"]
        )
        assert all(
            direction["release_error_mm"] < 1.5 for direction in result["directions"]
        )
    report = {
        "mujoco": mj.__version__,
        "source": args.source.name,
        "samples_per_direction": args.samples,
        "baseline": baseline,
        "variants": results,
        "scope": "Sampled geometry and upper-tracheal contact only; not every bronchial branch",
        "priority_regions": json.loads(str(regions["metadata"]))
        if regions is not None
        else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validation report: {args.report}")


if __name__ == "__main__":
    main()
