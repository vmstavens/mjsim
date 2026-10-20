from __future__ import annotations

import argparse
import os
from collections import deque

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mjsim-matplotlib")

import mujoco as mj
import mujoco.viewer
import numpy as np

from mjsim.utils.mjs import empty_scene


def add_mocap_weld_sensor(scene, mocap_body, name: str, object_site: str) -> None:
    site = f"{name}_loadcell_site"

    loadcell = mocap_body.add_body(name=f"{name}_loadcell", pos=[0, 0, 0])
    loadcell.add_geom(
        name=f"{name}_loadcell_geom",
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[0.018],
        mass=1e-6,
        contype=0,
        conaffinity=0,
        rgba=[0.2, 0.2, 1, 1],
    )
    loadcell.add_site(name=site, pos=[0, 0, 0], size=[0.03], rgba=[0, 1, 0, 1])

    scene.add_equality(
        name=name,
        type=mj.mjtEq.mjEQ_WELD,
        objtype=mj.mjtObj.mjOBJ_SITE,
        name1=site,
        name2=object_site,
        solref=[0.001, 3],
    )
    scene.add_sensor(
        name=f"{name}_force",
        type=mj.mjtSensor.mjSENS_FORCE,
        objtype=mj.mjtObj.mjOBJ_SITE,
        objname=site,
        dim=3,
    )
    scene.add_sensor(
        name=f"{name}_torque",
        type=mj.mjtSensor.mjSENS_TORQUE,
        objtype=mj.mjtObj.mjOBJ_SITE,
        objname=site,
        dim=3,
    )


def weld_sensor(
    model: mj.MjModel,
    data: mj.MjData,
    name: str,
    site_name: str | None = None,
) -> np.ndarray:
    site_id = model.site(site_name or f"{name}_loadcell_site").id
    force_adr = model.sensor(f"{name}_force").adr[0]
    torque_adr = model.sensor(f"{name}_torque").adr[0]

    force = data.sensordata[force_adr : force_adr + 3].copy()
    torque = data.sensordata[torque_adr : torque_adr + 3].copy()
    site_rot = data.site_xmat[site_id].reshape(3, 3)

    return np.concatenate([site_rot @ force, site_rot @ torque])


class LiveForcePlot:
    def __init__(self, window: float = 5.0, max_points: int = 2000):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("Install matplotlib or run with --no-plot.") from exc

        self.window = window
        self.ts: deque[float] = deque(maxlen=max_points)
        self.fs: deque[np.ndarray] = deque(maxlen=max_points)

        plt.ion()
        self.fig, self.ax = plt.subplots(num="Mocap weld force")
        self.lines = {
            "Fx": self.ax.plot([], [], label="Fx", color="tab:red")[0],
            "Fy": self.ax.plot([], [], label="Fy", color="tab:green")[0],
            "Fz": self.ax.plot([], [], label="Fz", color="tab:blue")[0],
            "|F|": self.ax.plot([], [], label="|F|", color="black", linewidth=2)[0],
        }
        self.ax.set_xlabel("time [s]")
        self.ax.set_ylabel("force [N]")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper left")
        self.fig.tight_layout()
        self.fig.show()

    def update(self, t: float, force: np.ndarray) -> None:
        self.ts.append(t)
        self.fs.append(force.copy())

        ts = np.array(self.ts)
        forces = np.array(self.fs)
        magnitude = np.linalg.norm(forces, axis=1)

        self.lines["Fx"].set_data(ts, forces[:, 0])
        self.lines["Fy"].set_data(ts, forces[:, 1])
        self.lines["Fz"].set_data(ts, forces[:, 2])
        self.lines["|F|"].set_data(ts, magnitude)

        xmax = max(self.window, t)
        self.ax.set_xlim(max(0.0, xmax - self.window), xmax)

        visible = ts >= max(0.0, xmax - self.window)
        ymax = max(
            1.0,
            float(np.max(np.abs(forces[visible]))),
            float(np.max(magnitude[visible])),
        )
        self.ax.set_ylim(-1.1 * ymax, 1.1 * ymax)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


class Sim:
    def __init__(self):
        self.weld_name = "rod_weld"
        self.m, self.d = self._build_model()
        mj.mj_forward(self.m, self.d)

        self.mocap_id = int(self.m.body_mocapid[self.m.body("mocap").id])
        self.start_pos = np.array([0.0, 0.0, 0.45])
        self.stroke = 0.14
        self.push_duration = 3.0
        self.step_count = 0
        self.max_force = 0.0

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData]:
        scene = empty_scene()
        scene.option.timestep = 0.001
        scene.option.gravity = [0, 0, 0]

        mocap = scene.worldbody.add_body(name="mocap", mocap=True, pos=[0, 0, 0.45])
        mocap.add_geom(
            name="mocap_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.04, 0.04, 0.04],
            contype=0,
            conaffinity=0,
            rgba=[1, 0, 0, 1],
        )

        rod = scene.worldbody.add_body(name="rod", pos=[0.3, 0, 0.45])
        rod.add_freejoint(name="rod_freejoint")
        rod.add_geom(
            name="rod_geom",
            type=mj.mjtGeom.mjGEOM_CAPSULE,
            fromto=[-0.3, 0, 0, 0.3, 0, 0],
            size=[0.035],
            mass=0.35,
            rgba=[0.75, 0.75, 0.75, 1],
            friction=[1.0, 0.005, 0.0001],
            solref=[0.01, 1],
            solimp=[0.95, 0.99, 0.001, 0.5, 2],
        )
        rod.add_site(
            name="rod_mount_site",
            pos=[-0.3, 0, 0],
            size=[0.025],
            rgba=[1, 0.8, 0, 1],
        )

        scene.worldbody.add_geom(
            name="wall",
            type=mj.mjtGeom.mjGEOM_BOX,
            pos=[0.74, 0, 0.45],
            size=[0.04, 0.35, 0.35],
            rgba=[0.15, 0.15, 0.18, 1],
            friction=[1.0, 0.005, 0.0001],
            solref=[0.01, 1],
            solimp=[0.95, 0.99, 0.001, 0.5, 2],
        )

        add_mocap_weld_sensor(scene, mocap, self.weld_name, "rod_mount_site")

        m = scene.compile()
        d = mj.MjData(m)
        return m, d

    def update_mocap(self) -> None:
        phase = min(self.d.time / self.push_duration, 1.0)
        x = self.stroke * 0.5 * (1.0 - np.cos(np.pi * phase))
        self.d.mocap_pos[self.mocap_id] = self.start_pos + np.array([x, 0.0, 0.0])

    def step(self) -> np.ndarray:
        self.update_mocap()
        mj.mj_step(self.m, self.d)
        self.step_count += 1

        wrench = weld_sensor(self.m, self.d, self.weld_name)
        self.max_force = max(self.max_force, float(np.linalg.norm(wrench[:3])))
        return wrench

    def run_headless(self, steps: int) -> None:
        wrench = weld_sensor(self.m, self.d, self.weld_name)
        for _ in range(steps):
            wrench = self.step()

        print(
            "rod_weld force world [N]: "
            f"F=({wrench[0]: .4f}, {wrench[1]: .4f}, {wrench[2]: .4f}) | "
            f"|F|max={self.max_force:.4f}"
        )

    def run_viewer(self, show_plot: bool) -> None:
        plot = LiveForcePlot() if show_plot else None

        print("Mocap pushes the welded rod into the wall along +X.")
        print('API call: wrench = weld_sensor(sim.m, sim.d, "rod_weld")')

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            viewer.cam.lookat = [0.35, 0, 0.45]
            viewer.cam.distance = 1.4

            while viewer.is_running():
                wrench = self.step()

                if plot is not None and self.step_count % 10 == 0:
                    plot.update(self.d.time, wrench[:3])

                if self.step_count % 250 == 0:
                    print(
                        "rod_weld force world [N]: "
                        f"F=({wrench[0]: .4f}, {wrench[1]: .4f}, {wrench[2]: .4f}) | "
                        f"|F|={np.linalg.norm(wrench[:3]):.4f}"
                    )

                viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    sim = Sim()
    if args.headless:
        sim.run_headless(args.steps)
    else:
        sim.run_viewer(show_plot=True)
        # sim.run_viewer(show_plot=not args.no_plot)


if __name__ == "__main__":
    main()
