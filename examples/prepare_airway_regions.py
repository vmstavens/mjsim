"""Prepare an inner-wall-only, region-weighted input for Blender decimation.

The local lumen width is estimated from inward normal/cone rays. A surface
graph retains high priority only for sufficiently wide regions connected to the
tracheal inlet. This is an allocation heuristic, not a scope motion planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from bronchoscope import default_airway
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from validate_airway_collision import ray_scene


def save_probe_reference(output, regions):
    """Save independent high-resolution inner-wall ray references for tests."""
    inner = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(regions["vertices"]),
        o3d.utility.Vector3iVector(regions["triangles"]),
    )
    directions = np.random.default_rng(42).normal(size=(16, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.repeat(regions["probe_centers"], len(directions), axis=0)
    rays = np.column_stack(
        (origins, np.tile(directions, (len(regions["probe_centers"]), 1)))
    )
    hits = (
        ray_scene(inner)
        .cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))["t_hit"]
        .numpy()
    )
    reference = {
        "metadata": json.loads(str(regions["metadata"])),
        "coordinate_frame": "original OBJ coordinates in millimetres",
        "centers_mm": regions["probe_centers"].tolist(),
        "directions": directions.tolist(),
        "reference_hits_mm": [
            float(value) if np.isfinite(value) else None for value in hits
        ],
    }
    output.write_text(json.dumps(reference, indent=2) + "\n")


def prepare(source: Path, output: Path, minimum_diameter: float = 6.5):
    if not np.isfinite(minimum_diameter) or not 0 < minimum_diameter < 12:
        raise ValueError("Priority diameter must be finite and between 0 and 12 mm")
    original = o3d.io.read_triangle_mesh(str(source))
    points, faces = np.asarray(original.vertices), np.asarray(original.triangles)
    cut_z = points[:, 2].max() - 5.0
    band = points[np.abs(points[:, 2] - (cut_z - 10.0)) < 0.75]
    seed = (band.min(axis=0) + band.max(axis=0)) / 2
    scene = ray_scene(original)
    hit = scene.cast_rays(
        o3d.core.Tensor([[*seed, 1, 0, 0]], dtype=o3d.core.Dtype.Float32)
    )
    face_id = int(hit["primitive_ids"].numpy()[0])
    labels, counts, _ = original.cluster_connected_triangles()
    labels = np.asarray(labels)
    inner_label = labels[face_id]
    inner_faces = faces[labels == inner_label]
    active, inverse = np.unique(inner_faces, return_inverse=True)
    points, faces = points[active].copy(), inverse.reshape(-1, 3)
    # Signed volume determines which way the closed cavity surface points.
    volume = (
        np.einsum(
            "ij,ij->i",
            points[faces[:, 0]],
            np.cross(points[faces[:, 1]], points[faces[:, 2]]),
        ).sum()
        / 6
    )
    if volume > 0:
        faces = faces[:, [0, 2, 1]]
    inner = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(faces)
    )
    inner.compute_vertex_normals()
    normals = np.asarray(inner.vertex_normals)
    inner_scene = ray_scene(inner)
    # The normal ray estimates a lumen chord. Cone rays reduce overestimates
    # near junctions and terminal caps; their lengths are cosine-corrected.
    origins = points + 0.03 * normals
    tangent = np.cross(normals, np.eye(3)[np.argmin(np.abs(normals), axis=1)])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    bitangent = np.cross(normals, tangent)
    width = np.full(len(points), np.inf)
    normal_hits = None
    for index, direction in enumerate(
        (
            normals,
            (normals + tangent) / np.sqrt(2),
            (normals - tangent) / np.sqrt(2),
            (normals + bitangent) / np.sqrt(2),
            (normals - bitangent) / np.sqrt(2),
        )
    ):
        rays = np.column_stack((origins, direction)).astype(np.float32)
        distances = inner_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        if index == 0:
            normal_hits = distances.copy()
        width = np.minimum(
            width, (distances + 0.03) / (1 if index == 0 else 1 / np.sqrt(2))
        )
    width[~np.isfinite(width)] = 0
    eligible = width >= minimum_diameter
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = edges[eligible[edges].all(axis=1)]
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(points), len(points)),
    ).tocsr()
    _, regions = connected_components(graph, directed=False)
    inlet = eligible & (points[:, 2] > cut_z - 8)
    if not inlet.any():
        raise ValueError("No wide cavity surface connected to the inlet")
    seed_label = np.bincount(regions[inlet]).argmax()
    reachable = eligible & (regions == seed_label)
    priority = np.clip((width - minimum_diameter) / (12.0 - minimum_diameter), 0, 1)
    priority[~reachable] = 0
    collapse_weights = 1.0 - 0.99 * priority

    # Independent geometric probes: normal-chord midpoints with enough actual
    # surface clearance for the priority diameter, rather than chord length alone.
    candidates = np.flatnonzero(
        reachable & np.isfinite(normal_hits) & (points[:, 2] < cut_z - 5)
    )
    rng = np.random.default_rng(42)
    candidates = rng.choice(candidates, min(3000, len(candidates)), replace=False)
    centers = origins[candidates] + normals[candidates] * (
        normal_hits[candidates, None] / 2
    )
    clearance = inner_scene.compute_distance(
        o3d.core.Tensor(centers.astype(np.float32))
    ).numpy()
    probes = centers[clearance >= minimum_diameter / 2]
    probes = probes[rng.choice(len(probes), min(300, len(probes)), replace=False)]

    keep = np.all(points[faces, 2] <= cut_z, axis=1)
    faces = faces[keep]
    active, inverse = np.unique(faces, return_inverse=True)
    metadata = {
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "inner_component": int(inner_label),
        "original_components": len(counts),
        "inner_triangles_before_opening": int(counts[inner_label]),
        "opened_inner_triangles": len(faces),
        "minimum_diameter_mm": minimum_diameter,
        "reachable_vertices": int(reachable[active].sum()),
        "probe_centers": len(probes),
        "inlet_cut_z_source_units": float(cut_z),
        "note": "Connected surface-width heuristic; no proof of full-device reachability. Inner wall only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices=points[active],
        triangles=inverse.reshape(-1, 3),
        collapse_weights=collapse_weights[active],
        reachable=reachable[active],
        lumen_width_mm=width[active],
        probe_centers=probes,
        metadata=np.array(json.dumps(metadata)),
    )
    with np.load(output) as regions:
        save_probe_reference(output.with_suffix(".probes.json"), regions)
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_airway())
    parser.add_argument("--output", type=Path, default=Path("/tmp/airway-regions.npz"))
    parser.add_argument(
        "--minimum-diameter",
        type=float,
        default=6.5,
        help="priority diameter in mm: default 6 mm shaft plus 0.5 mm clearance",
    )
    args = parser.parse_args()
    prepare(args.source, args.output, args.minimum_diameter)


if __name__ == "__main__":
    main()
