from __future__ import annotations

import argparse

import mujoco as mj
import mujoco.viewer
import numpy as np

from mjsim.utils.mjs import empty_scene


class Sim:
    """Container filled with damped free-sphere particles."""

    def __init__(self, particles_per_axis: tuple[int, int, int]) -> None:
        self.particles_per_axis = particles_per_axis
        self.radius = 0.012
        self.spacing = 2.25 * self.radius
        self.container_halfsize = np.array([0.28, 0.20, 0.16])
        self.paddle_start = -0.20
        self.paddle_travel = 0.40
        self.paddle_period = 3.5

        self.m, self.d = self._build_model()
        self.paddle_actuator_id = self.m.actuator("paddle_position").id
        self.particle_body_ids = self._particle_body_ids()
        mj.mj_forward(self.m, self.d)

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData]:
        spec = empty_scene(
            sim_name="particle_container",
            add_floor=False,
            memory="100M",
            solver="Newton",
            integrator="implicitfast",
            timestep=0.001,
            iterations=120,
            tolerance=1e-10,
            statistic_center=[0.0, 0.0, 0.15],
            statistic_extent=0.85,
            option_overrides={
                # Fluid drag helps the free particles behave more like a slurry
                # than dry marbles.
                "density": 80.0,
                "viscosity": 0.35,
            },
        )
        spec.visual.global_.azimuth = 135
        spec.visual.global_.elevation = -25

        self._add_container(spec)
        self._add_paddle(spec)
        self._add_particles(spec)

        model = spec.compile()
        data = mj.MjData(model)
        return model, data

    def _add_container(self, spec: mj.MjSpec) -> None:
        hx, hy, hz = self.container_halfsize
        thickness = 0.014
        wall_rgba = [0.65, 0.72, 0.78, 0.38]
        wall_friction = [0.12, 0.005, 0.0001]
        wall_solref = [0.006, 1.0]
        wall_solimp = [0.95, 0.995, 0.001, 0.5, 2.0]

        def add_wall(name: str, pos: list[float], size: list[float]) -> None:
            spec.worldbody.add_geom(
                name=name,
                type=mj.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=size,
                rgba=wall_rgba,
                friction=wall_friction,
                solref=wall_solref,
                solimp=wall_solimp,
            )

        add_wall("container_floor", [0.0, 0.0, -thickness], [hx, hy, thickness])
        add_wall("container_x_min", [-hx - thickness, 0.0, hz], [thickness, hy, hz])
        add_wall("container_x_max", [hx + thickness, 0.0, hz], [thickness, hy, hz])
        add_wall("container_y_min", [0.0, -hy - thickness, hz], [hx, thickness, hz])
        add_wall("container_y_max", [0.0, hy + thickness, hz], [hx, thickness, hz])

    def _add_paddle(self, spec: mj.MjSpec) -> None:
        paddle = spec.worldbody.add_body(
            name="paddle",
            pos=[self.paddle_start, 0.0, 0.075],
        )
        paddle.add_joint(
            name="paddle_slide",
            type=mj.mjtJoint.mjJNT_SLIDE,
            axis=[1.0, 0.0, 0.0],
            range=[0.0, self.paddle_travel],
            damping=30.0,
        )
        paddle.add_geom(
            name="paddle_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.012, 0.155, 0.055],
            rgba=[0.95, 0.28, 0.12, 1.0],
            friction=[0.4, 0.005, 0.0001],
            solref=[0.006, 1.0],
            solimp=[0.95, 0.995, 0.001, 0.5, 2.0],
        )
        spec.add_actuator(
            name="paddle_position",
            target="paddle_slide",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=[0.0, self.paddle_travel],
            forcerange=[-160.0, 160.0],
        ).set_to_position(kp=650.0, kv=70.0)

    def _add_particles(self, spec: mj.MjSpec) -> None:
        nx, ny, nz = self.particles_per_axis
        rng = np.random.default_rng(3)
        start = np.array(
            [
                -0.5 * (nx - 1) * self.spacing,
                -0.5 * (ny - 1) * self.spacing,
                self.radius + 0.020,
            ]
        )
        color_low = np.array([0.12, 0.58, 0.88, 0.95])
        color_high = np.array([0.22, 0.78, 0.96, 0.95])

        index = 0
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    jitter = rng.uniform(-0.15, 0.15, size=3) * self.radius
                    pos = start + np.array(
                        [
                            ix * self.spacing,
                            iy * self.spacing,
                            iz * self.spacing,
                        ]
                    )
                    pos += jitter
                    body = spec.worldbody.add_body(
                        name=f"particle_{index:03d}",
                        pos=pos.tolist(),
                    )
                    body.add_freejoint()
                    mix = iz / max(nz - 1, 1)
                    rgba = ((1.0 - mix) * color_low + mix * color_high).tolist()
                    body.add_geom(
                        name=f"particle_{index:03d}_geom",
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=[self.radius],
                        mass=0.004,
                        rgba=rgba,
                        friction=[0.04, 0.002, 0.0001],
                        solref=[0.004, 1.0],
                        solimp=[0.95, 0.995, 0.001, 0.5, 2.0],
                    )
                    index += 1

    def _particle_body_ids(self) -> list[int]:
        body_ids: list[int] = []
        for body_id in range(1, self.m.nbody):
            name = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, body_id)
            if name and name.startswith("particle_"):
                body_ids.append(body_id)
        return body_ids

    def _update_paddle(self) -> None:
        phase = self.d.time / self.paddle_period
        alpha = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
        self.d.ctrl[self.paddle_actuator_id] = alpha * self.paddle_travel

    def step(self) -> None:
        self._update_paddle()
        mj.mj_step(self.m, self.d)

    def particle_positions(self) -> np.ndarray:
        return self.d.xpos[self.particle_body_ids]

    def run_headless(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

        positions = self.particle_positions()
        center = np.mean(positions, axis=0)
        z_min = float(np.min(positions[:, 2]))
        z_max = float(np.max(positions[:, 2]))
        print(
            "particle container: "
            f"particles={len(self.particle_body_ids)}, contacts={self.d.ncon}"
        )
        print(f"center [m]: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"particle z range [m]: ({z_min:.3f}, {z_max:.3f})")

    def run_viewer(self) -> None:
        print(
            "Particle container demo. Blue free-sphere particles are damped and "
            "low-friction so the bed behaves like a semi-liquid slurry. "
            "The red paddle sweeps through the container. Close the viewer to exit."
        )
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.lookat = [0.0, 0.0, 0.11]
            viewer.cam.distance = 0.82
            while viewer.is_running():
                self.step()
                viewer.sync()


def particle_grid(total_hint: int) -> tuple[int, int, int]:
    if total_hint <= 0:
        msg = "--particles must be positive"
        raise ValueError(msg)
    nz = max(3, round(total_hint ** (1.0 / 3.0) * 0.65))
    nx = max(4, round((total_hint / nz) ** 0.5 * 1.25))
    ny = max(4, round(total_hint / (nx * nz)))
    return nx, ny, nz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--particles",
        type=int,
        default=320,
        help="Approximate particle count; the grid dimensions are rounded.",
    )
    args = parser.parse_args()

    sim = Sim(particle_grid(args.particles))
    if args.headless:
        sim.run_headless(args.steps)
    else:
        sim.run_viewer()


if __name__ == "__main__":
    main()
