from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import gettempdir

import mujoco as mj
import mujoco.viewer
import numpy as np

from mjsim.utils.mjs import cable, cloth, dlo, empty_scene, sponge


def custom_mesh_path() -> str:
    """Write a low-poly non-convex torus OBJ for the mesh flex example."""
    major_segments = 14
    minor_segments = 6
    major_radius = 0.075
    minor_radius = 0.025

    lines = [
        "# Low-poly torus generated for MuJoCo make_flex(type='mesh').",
        "# The center hole is real mesh topology, not a convex hull.",
        "o low_poly_torus",
    ]

    for i in range(major_segments):
        u = 2.0 * np.pi * i / major_segments
        for j in range(minor_segments):
            v = 2.0 * np.pi * j / minor_segments
            ring_radius = major_radius + minor_radius * np.cos(v)
            x = ring_radius * np.cos(u)
            y = ring_radius * np.sin(u)
            z = minor_radius * np.sin(v)
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

    for i in range(major_segments):
        for j in range(minor_segments):
            lines.append(f"vt {i / major_segments:.6f} {j / minor_segments:.6f}")

    for i in range(major_segments):
        next_i = (i + 1) % major_segments
        for j in range(minor_segments):
            next_j = (j + 1) % minor_segments
            a = i * minor_segments + j + 1
            b = next_i * minor_segments + j + 1
            c = next_i * minor_segments + next_j + 1
            d = i * minor_segments + next_j + 1
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
            lines.append(f"f {a}/{a} {c}/{c} {d}/{d}")

    path = Path(gettempdir()) / "mjsim_make_flex_custom_mesh.obj"
    path.write_text("\n".join([*lines, ""]), encoding="utf-8")
    return str(path)



class Sim:
    """Small deformable-object catalogue with a passive MuJoCo viewer."""

    def __init__(self) -> None:
        self.pusher_start = np.array([0.22, -0.36, 0.18])
        self.pusher_end = np.array([0.22, -0.05, 0.18])
        self.pusher_travel = float(self.pusher_end[1] - self.pusher_start[1])
        self.settle_time = 0.5
        self.push_period = 3.0
        self.flex_names = ["rope_1d", "cloth_2d", "soft_cube_3d", "custom_mesh"]

        self.m, self.d = self._build_model()
        mj.mj_forward(self.m, self.d)

        self.pusher_actuator_id = self.m.actuator("pusher_position").id
        self.flex_body_ids = {
            name: self._named_body_ids(f"{name}_") for name in self.flex_names
        }
        self.composite_cable_body_ids = self._body_ids_with_prefix(
            "composite_cable:B"
        )

        n_threads: int = 10
        mj.mju_threadpool(self.d, n_threads)

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData]:
        spec = empty_scene(
            sim_name="make_flex_demo",
            add_assets=False,
            add_floor=False,
            memory="100M",
            solver="CG",
            integrator="implicitfast",
            timestep=0.0005,
            tolerance=1e-8,
            iterations=200,
            statistic_center=[0.0, 0.0, 0.20],
            statistic_extent=0.80,
        )
        spec.memory = 100 * 1024 * 1024
        spec.visual.global_.azimuth = 115
        spec.visual.global_.elevation = -20

        spec.worldbody.add_light(
            pos=[0.0, 0.0, 1.5],
            dir=[0.0, 0.0, -1.0],
            type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        spec.worldbody.add_geom(
            name="floor",
            type=mj.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, 0.5],
            rgba=[0.25, 0.28, 0.30, 1.0],
            solref=[0.02, 1.0],
            solimp=[0.90, 0.95, 0.001, 0.5, 2.0],
        )
        spec.worldbody.add_geom(
            name="wall",
            type=mj.mjtGeom.mjGEOM_BOX,
            pos=[0.22, 0.12, 0.17],
            size=[0.18, 0.02, 0.20],
            rgba=[0.18, 0.18, 0.22, 1.0],
            friction=[1.0, 0.005, 0.0001],
        )

        pusher = spec.worldbody.add_body(
            name="pusher",
            pos=self.pusher_start.tolist(),
        )
        pusher.add_joint(
            name="pusher_slide",
            type=mj.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 1.0, 0.0],
            range=[0.0, self.pusher_travel],
            damping=12.0,
        )
        pusher.add_geom(
            name="pusher_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.16, 0.025, 0.16],
            rgba=[0.95, 0.25, 0.12, 1.0],
            friction=[1.0, 0.005, 0.0001],
        )
        spec.add_actuator(
            name="pusher_position",
            target="pusher_slide",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=[0.0, self.pusher_travel],
            forcerange=[-80.0, 80.0],
        ).set_to_position(kp=350.0, kv=35.0)

        rope = dlo(
            model_name="rope_1d",
            prefix="rope_1d:",
            n_segments=11,
            length=0.35,
            mass=0.08,
            radius=0.006,
            edge_damping=0.01,
            edge_stiffness=25.0,
            rgba=[0.98, 0.58, 0.12, 1.0],
            condim=3,
        )
        spec.attach(
            rope,
            prefix="",
            frame=spec.worldbody.add_frame(pos=[-0.38, -0.22, 0.42]),
        )

        composite_cable = cable(
            model_name="composite_cable",
            prefix="composite_cable:",
            curve="0 s 0",
            n_segments=11,
            length=0.35,
            mass=0.08,
            segment_size=0.006,
            geom_rgba=[0.72, 0.72, 0.78, 1.0],
            geom_condim=3,
        )
        spec.attach(
            composite_cable,
            prefix="",
            frame=spec.worldbody.add_frame(pos=[-0.62, -0.22, 0.42]),
        )

        sheet = cloth(
            model_name="cloth_2d",
            prefix="cloth_2d:",
            width_segments=8,
            height_segments=8,
            width=0.245,
            height=0.245,
            spacing=[0.035, 0.035, 0.035],
            mass=0.28,
            radius=0.006,
            pin_corner=False,
            rgba=[0.18, 0.78, 0.35, 0.82],
            young=800.0,
            poisson=0.2,
            damping=0.1,
            thickness=0.006,
            elastic2d="both",
            condim=3,
            friction=[0.6, 0.005, 0.0001],
            solref=[0.0001, 0.1],
            solimp=[0.90, 0.95, 0.001, 0.5, 2.0],
            selfcollide="none",
        )
        spec.attach(
            sheet,
            prefix="",
            frame=spec.worldbody.add_frame(pos=[-0.35, 0.24, 0.38]),
        )

        cube = sponge(
            model_name="soft_cube_3d",
            prefix="soft_cube_3d:",
            count=[6, 6, 6],
            spacing=[0.035, 0.035, 0.035],
            mass=0.35,
            radius=0.006,
            rgba=[0.05, 0.65, 0.95, 0.75],
            young=5_000.0,
            poisson=0.25,
            damping=0.002,
            condim=3,
            solref=None,
            solimp=None,
            selfcollide="none",
        )
        spec.attach(
            cube,
            prefix="",
            frame=spec.worldbody.add_frame(pos=[0.22, -0.13, 0.18]),
        )

        mesh_root = spec.worldbody.add_body(name="custom_mesh_root")
        mesh = mesh_root.make_flex(
            name="custom_mesh",
            type="mesh",
            file=custom_mesh_path(),
            # Surface flex: the torus hole remains part of the collision topology.
            dim=2,
            pos=[0.45, 0.24, 0.24],
            radius=0.005,
            mass=0.18,
            equality=1,
        )

        mesh.rgba = [0.988235294, 0.058823529, 0.752941176, 1.0]
        mesh.young = 12_000.0
        mesh.poisson = 0.2
        mesh.damping = 0.5
        mesh.thickness = 0.010
        mesh.elastic2d = 3
        mesh.condim = 3
        mesh.solref = [0.025, 1.0]
        mesh.solimp = [0.90, 0.95, 0.001, 0.5, 2.0]
        mesh.selfcollide = mj.mjtFlexSelf.mjFLEXSELF_NONE

        model = spec.compile()
        data = mj.MjData(model)
        return model, data

    def _named_body_ids(self, prefix: str) -> list[int]:
        return self._body_ids_with_prefix(prefix, numeric_suffix=True)

    def _body_ids_with_prefix(
        self,
        prefix: str,
        *,
        numeric_suffix: bool = False,
    ) -> list[int]:
        body_ids: list[int] = []
        for body_id in range(1, self.m.nbody):
            name = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, body_id)
            suffix = name.removeprefix(prefix) if name else ""
            if name and name.startswith(prefix) and (
                not numeric_suffix or suffix.isdigit()
            ):
                body_ids.append(body_id)
        return body_ids

    def _update_pusher(self) -> None:
        if self.d.time < self.settle_time:
            alpha = 0.0
        else:
            phase = (self.d.time - self.settle_time) / self.push_period
            alpha = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))

        self.d.ctrl[self.pusher_actuator_id] = alpha * self.pusher_travel

    def step(self) -> None:
        self._update_pusher()
        mj.mj_step(self.m, self.d)

    def flex_center(self, name: str) -> np.ndarray:
        return np.mean(self.d.xpos[self.flex_body_ids[name]], axis=0)

    def run_headless(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

        centers = {
            name: self.flex_center(name)
            for name, body_ids in self.flex_body_ids.items()
            if body_ids
        }
        print(
            "make_flex catalogue: "
            f"nflex={self.m.nflex}, nbody={self.m.nbody}, neq={self.m.neq}"
        )
        for name, center in centers.items():
            print(
                f"{name} center [m]: "
                f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
            )
        if self.composite_cable_body_ids:
            center = np.mean(self.d.xpos[self.composite_cable_body_ids], axis=0)
            print(
                "composite_cable center [m]: "
                f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
            )

    def run_viewer(self) -> None:
        print("MuJoCo 3.10 MjsBody.make_flex creates 1D, 2D, 3D, and mesh flexes.")
        print(
            "The gray cable uses the composite cable plugin for comparison. "
            "A force-limited slider pusher periodically presses the 3D cube. "
            "The cloth uses softer contact and extra damping to reduce jitter. "
            "Close the viewer to exit."
        )

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.cam.azimuth = 115
            viewer.cam.elevation = -20
            viewer.cam.lookat = [0.04, 0.02, 0.22]
            viewer.cam.distance = 0.95

            while viewer.is_running():
                self.step()
                viewer.sync()


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
