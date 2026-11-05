"""Create collision-only OBJ copies with Blender's Collapse decimator.

uv run python examples/prepare_airway_regions.py --output /tmp/airway-regions.npz
blender --background --factory-startup --python-exit-code 1 \
    --python examples/decimate_airway.py -- --regions /tmp/airway-regions.npz

Run through Blender, not the project's Python. Coordinates/units are preserved;
the capped inlet's top 5 source units (mm for this asset) are removed first.
With prepared regions, default to a 5k budget and a protected 38k intermediate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils.kdtree import KDTree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent / "assets" / "airways"
    parser.add_argument("--source", type=Path, default=directory / "hollow_airway.obj")
    parser.add_argument("--output-dir", type=Path, default=directory)
    parser.add_argument("--triangles", type=int, nargs="+", default=[5000])
    parser.add_argument("--inlet-trim", type=float, default=5.0)
    parser.add_argument(
        "--regions",
        type=Path,
        help="prepared inner-wall NPZ from prepare_airway_regions.py",
    )
    parser.add_argument("--weight-factor", type=float, default=10.0)
    parser.add_argument(
        "--protect-after",
        type=int,
        default=None,
        help="first simplify to this count, then freeze priority vertices and simplify distal regions",
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    if args.protect_after is None:
        args.protect_after = 38000 if args.regions else 0
    if min(args.triangles) < 100:
        parser.error("--triangles must be at least 100")
    if min(args.triangles) <= 5000 and args.regions is None:
        parser.error("For a 5k mesh, supply --regions from prepare_airway_regions.py")
    if args.protect_after and (not args.regions or len(args.triangles) != 1):
        parser.error("--protect-after requires --regions and a single target count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces = [], []
    with args.source.open() as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append(tuple(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                indices = [int(field.split("/")[0]) for field in line.split()[1:]]
                indices = [i - 1 if i > 0 else len(vertices) + i for i in indices]
                for i in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[i], indices[i + 1]))
    points, triangles = np.asarray(vertices), np.asarray(faces)
    del vertices, faces
    cut_z = points[:, 2].max() - args.inlet_trim
    triangles = triangles[np.all(points[triangles, 2] <= cut_z, axis=1)]
    active, inverse = np.unique(triangles, return_inverse=True)
    points, triangles = points[active], inverse.reshape(-1, 3)
    region_data = None
    if args.regions:
        region_data = np.load(args.regions)
        metadata = json.loads(str(region_data["metadata"]))
        if (
            metadata["source_sha256"]
            != hashlib.sha256(args.source.read_bytes()).hexdigest()
        ):
            raise ValueError("Prepared regions do not match the source OBJ")
        points, triangles = region_data["vertices"], region_data["triangles"]
        cut_z = metadata["inlet_cut_z_source_units"]

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    mesh = bpy.data.meshes.new("airway_source")
    mesh.from_pydata(points.tolist(), [], triangles.tolist())
    mesh.update()
    obj = bpy.data.objects.new("airway_collision", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    opened_triangles = len(triangles)
    staged_weights = None
    if args.protect_after:
        first = obj.modifiers.new("intermediate_decimate", "DECIMATE")
        first.ratio = min(1.0, args.protect_after / len(triangles))
        first.use_collapse_triangulate = True
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        intermediate = evaluated.to_mesh().copy()
        evaluated.to_mesh_clear()
        obj.modifiers.remove(first)
        obj.data = intermediate
        intermediate.calc_loop_triangles()
        points = np.array([vertex.co[:] for vertex in intermediate.vertices])
        triangles = np.array(
            [triangle.vertices[:] for triangle in intermediate.loop_triangles]
        )
        tree = KDTree(len(region_data["vertices"]))
        for i, point in enumerate(region_data["vertices"]):
            tree.insert(point, i)
        tree.balance()
        nearest = np.array([tree.find(point)[1] for point in points])
        frozen = region_data["reachable"][nearest] | (points[:, 2] > cut_z - 2.0)
        staged_weights = np.where(frozen, 0.0, 1.0)
    # Discourage collapsing the open inlet rim. Check its shape separately in
    # the geometric validation; this weighting is not a topology guarantee.
    weights = obj.vertex_groups.new(name="collapse_weights")
    if staged_weights is not None:
        weights.add(np.flatnonzero(staged_weights == 0).tolist(), 0.0, "REPLACE")
        weights.add(np.flatnonzero(staged_weights == 1).tolist(), 1.0, "REPLACE")
    elif region_data is None:
        collar = points[:, 2] > cut_z - 2.0
        weights.add(np.flatnonzero(~collar).tolist(), 1.0, "REPLACE")
        weights.add(np.flatnonzero(collar).tolist(), 0.0, "REPLACE")
    else:
        # Quantize the allocation weights to avoid hundreds of thousands of
        # individual Blender API calls. Low weights resist edge collapse.
        bins = np.rint(region_data["collapse_weights"] * 100).astype(int)
        for weight in np.unique(bins):
            weights.add(
                np.flatnonzero(bins == weight).tolist(), float(weight / 100), "REPLACE"
            )

    summary = {
        "source": args.source.name,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "blender": bpy.app.version_string,
        "source_triangles": sum(
            1 for line in args.source.open() if line.startswith("f ")
        ),
        "opened_triangles": opened_triangles,
        "inlet_cut_z_source_units": float(cut_z),
        "variants": [],
        "regions": json.loads(str(region_data["metadata"]))
        if region_data is not None
        else None,
        "weight_factor": args.weight_factor,
        "intermediate_triangles": len(triangles) if args.protect_after else None,
        "frozen_vertices": int(np.sum(staged_weights == 0))
        if staged_weights is not None
        else 0,
    }
    for target in args.triangles:
        start = time.perf_counter()
        modifier = obj.modifiers.new("collision_decimate", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = min(1.0, target / len(triangles))
        modifier.use_collapse_triangulate = True
        modifier.vertex_group = weights.name
        modifier.vertex_group_factor = args.weight_factor
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        result = evaluated.to_mesh()
        result.calc_loop_triangles()
        if len(result.loop_triangles) > target:
            raise RuntimeError(
                f"Frozen region prevents the requested budget: {len(result.loop_triangles)} > {target}"
            )
        suffix = f"_{target}" if len(args.triangles) > 1 else ""
        output = args.output_dir / f"{args.source.stem}_collision{suffix}.obj"
        with output.open("w") as stream:
            stream.write(
                "# Blender Collapse collision surface; source coordinates and units\n"
            )
            for vertex in result.vertices:
                stream.write("v {:.9f} {:.9f} {:.9f}\n".format(*vertex.co))
            for triangle in result.loop_triangles:
                stream.write("f {} {} {}\n".format(*(i + 1 for i in triangle.vertices)))
        item = {
            "file": output.name,
            "triangles": len(result.loop_triangles),
            "vertices": len(result.vertices),
            "seconds": time.perf_counter() - start,
            "bytes": output.stat().st_size,
        }
        summary["variants"].append(item)
        print(json.dumps(item), flush=True)
        evaluated.to_mesh_clear()
        obj.modifiers.remove(modifier)
    (args.output_dir / "decimation.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
