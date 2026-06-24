"""Simulate a flexible Velux sealing strip built from a repeated mesh profile."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import gettempdir
import time
from typing import Sequence

import mujoco as mj
import mujoco.viewer
import numpy as np

from mjsim.utils.mjs import empty_scene


ASSET_PATH = Path(__file__).parent / "assets" / "velux-sealing-element.obj"


def merged_obj_path(mesh_path: Path) -> Path:
    """Merge the profile OBJ's many objects into one MuJoCo mesh asset.

    MuJoCo's direct OBJ importer uses only the first object in this CAD export.
    Removing OBJ object/group/material boundaries preserves all its triangles in
    one mesh, while keeping the original positions, normals, and face indices.
    """
    source = mesh_path.resolve()
    output = Path(gettempdir()) / f"{source.stem}_mujoco_merged.obj"
    ignored_prefixes = ("o ", "g ", "mtllib ", "usemtl ")
    lines = [
        line
        for line in source.read_text(encoding="utf-8").splitlines()
        if not line.startswith(ignored_prefixes)
    ]
    output.write_text("\n".join([*lines, ""]), encoding="utf-8")
    return output


def add_velux_seal(
    scene: mj.MjSpec,
    *,
    name: str = "velux_seal",
    mesh_path: Path = ASSET_PATH,
    segment_count: int = 20,
    length: float = 1.0,
    pos: Sequence[float] = (0.0, 0.0, 0.40),
    profile_scale: Sequence[float] = (1.0, 7.0, 1.0),
    segment_mass: float = 0.00035,
    twist_modulus: float = 120_000.0,
    bend_modulus: float = 20_000_000.0,
    stretch_stiffness: float = 3.0,
    stretch_damping: float = 3.0,
) -> str:
    """Add a free, stretchable mesh-profile seal.

    The source OBJ is the cross-section used for every segment. Each segment has
    a slide and ball joint at its upstream boundary, so the chain can stretch
    along its length while bending and twisting like the old sealing model.

    Returns:
        Name of the final segment body.
    """
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Velux sealing mesh does not exist: {mesh_path}")
    if segment_count < 2:
        raise ValueError("segment_count must be at least 2")
    if length <= 0.0:
        raise ValueError("length must be positive")
    if stretch_stiffness < 0.0 or stretch_damping < 0.0:
        raise ValueError("stretch_stiffness and stretch_damping must be non-negative")
    if len(pos) != 3 or len(profile_scale) != 3:
        raise ValueError("pos and profile_scale must each contain three values")

    segment_length = length / segment_count
    mesh_name = f"{name}_profile"
    mesh = scene.add_mesh(
        name=mesh_name,
        file=str(merged_obj_path(mesh_path)),
        scale=[float(value) for value in profile_scale],
    )
    mesh.inertia = mj.mjtMeshInertia.mjMESH_INERTIA_SHELL

    # Approximate the previous cable-plugin stiffness from a 1 mm circular section.
    radius = 0.001
    polar_moment = np.pi * radius**4 / 2.0
    area_moment = np.pi * radius**4 / 4.0
    ball_stiffness = (
        polar_moment * twist_modulus
        + 2.0 * area_moment * bend_modulus
    ) / (3.0 * segment_length)

    root = scene.worldbody.add_body(
        name=f"{name}_root",
        pos=[float(value) for value in pos],
    )
    root.add_freejoint(name=f"{name}_free")

    parent = root
    last_body_name = ""
    for index in range(segment_count):
        body = parent.add_body(
            name=f"{name}_segment_{index}",
            pos=[0.0, 0.0, 0.0] if index == 0 else [0.0, segment_length, 0.0],
        )
        if index > 0:
            joint_pos = [0.0, -segment_length / 2.0, 0.0]
            stretch_joint = body.add_joint(
                name=f"{name}_stretch_{index}",
                type=mj.mjtJoint.mjJNT_SLIDE,
                pos=joint_pos,
                axis=[0.0, 1.0, 0.0],
            )
            stretch_joint.stiffness = [stretch_stiffness, 0.0, 0.0]
            stretch_joint.damping = [stretch_damping, 0.0, 0.0]

            joint = body.add_joint(
                name=f"{name}_joint_{index}",
                type=mj.mjtJoint.mjJNT_BALL,
                pos=joint_pos,
            )
            joint.damping = [0.01, 0.01, 0.01]
            joint.stiffness = [ball_stiffness, ball_stiffness, ball_stiffness]
            joint.armature = 0.001

        geom = body.add_geom(
            name=f"{name}_geom_{index}",
            type=mj.mjtGeom.mjGEOM_MESH,
            mass=segment_mass,
            rgba=[0.08, 0.08, 0.10, 1.0],
            friction=[0.6, 0.01, 0.001],
            condim=4,
            solref=[0.001, 3.0],
        )
        geom.meshname = mesh_name

        parent = body
        last_body_name = body.name

    parent.add_site(
        name=f"{name}_tip",
        size=[0.012],
        rgba=[1.0, 0.45, 0.05, 1.0],
    )
    return last_body_name


class Sim:
    def __init__(self) -> None:
        self.m, self.d, self.tip_body_name = self._build_model()
        mj.mj_forward(self.m, self.d)
        self.tip_body_id = self.m.body(self.tip_body_name).id

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData, str]:
        scene = empty_scene(
            sim_name="velux_sealing",
            memory="100M",
            solver="CG",
            timestep=0.001,
            tolerance=1e-8,
            iterations=100,
            nativeccd=True,
            statistic_center=(0.0, 0.45, 0.25),
            statistic_extent=0.85,
        )
        tip_body_name = add_velux_seal(scene)

        model = scene.compile()
        return model, mj.MjData(model), tip_body_name

    def step(self) -> None:
        mj.mj_step(self.m, self.d)

    def run_headless(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

        tip = self.d.xpos[self.tip_body_id]
        print(
            "Velux sealing strip: "
            f"time={self.d.time:.3f} s | "
            f"tip=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) m | "
            f"contacts={self.d.ncon}"
        )

    def run_viewer(self) -> None:
        print("Flexible Velux sealing strip with a repeated mesh profile.")
        print("Close the MuJoCo viewer to exit.")

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.cam.azimuth = 125
            viewer.cam.elevation = -22
            viewer.cam.lookat = [0.0, 0.45, 0.22]
            viewer.cam.distance = 1.35

            while viewer.is_running():
                step_start = time.perf_counter()
                self.step()
                viewer.sync()

                remaining = self.m.opt.timestep - (time.perf_counter() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()

    sim = Sim()
    if args.headless:
        sim.run_headless(args.steps)
    else:
        sim.run_viewer()


if __name__ == "__main__":
    main()
