"""Three feeder belts deliver luggage onto a rotating carousel.

Adapted from MuJoCo's surfacevel/carousel.xml, matching the demonstration at
https://www.youtube.com/watch?v=PdSdrqhSiZA. Native surface velocities drive
contact friction; the bags have free joints and no actuators or scripted forces.
Requires MuJoCo 3.11 or newer. Assets are bundled for offline use.

Space reverses all surfaces, P pauses/resumes them, and Backspace resets the scene.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from queue import SimpleQueue

import mujoco as mj
import numpy as np
from physics.palette import (
    CHARCOAL,
    COOL_GRAY,
    FLAME,
    GRAPHITE,
    LIGHT_ORANGE,
    OFF_WHITE,
    RAISED,
    apply_palette,
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "physics"
    / "models"
    / "surfacevel"
    / "carousel.xml"
)


class Sim:
    """Contact-driven conveyors and carousel with six differently shaped bags."""

    def __init__(self, speed: float = 0.7) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        if not hasattr(mj.MjModel, "geom_surfacevel"):
            raise RuntimeError(
                "The luggage carousel requires MuJoCo >=3.11 for native surfacevel contacts."
            )
        self.speed = speed
        self.m = mj.MjModel.from_xml_path(str(MODEL_PATH))
        # Keep a dark transport surface and island, neutral rails, and distinct bags.
        apply_palette(
            self.m,
            geom_colors={
                "carousel": GRAPHITE,
                "belt": GRAPHITE,
                "rail": COOL_GRAY,
                "island": RAISED,
                "floor": CHARCOAL,
                "bag1": FLAME,
                "bag2": OFF_WHITE,
                "bag3": COOL_GRAY,
                "bag4": LIGHT_ORANGE,
                "bag5": OFF_WHITE,
                "bag6": FLAME,
            },
        )
        # Bag geom names are empty upstream, but apply_palette also matches bodies.
        self.base_surface_velocity = self.m.geom_surfacevel.copy() * (speed / 0.7)
        self.bag_ids = [self.m.body(f"bag{i}").id for i in range(1, 7)]
        self.d = mj.MjData(self.m)
        self.reset()

    def reset(self) -> None:
        mj.mj_resetData(self.m, self.d)
        self.direction = 1.0
        self.paused = False
        self._update_surfaces()
        mj.mj_forward(self.m, self.d)

    def _update_surfaces(self) -> None:
        self.m.geom_surfacevel[:] = self.base_surface_velocity * (
            0 if self.paused else self.direction
        )

    def reverse(self) -> None:
        self.direction *= -1
        self._update_surfaces()

    def step(self) -> None:
        mj.mj_step(self.m, self.d)
        if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():
            raise RuntimeError("Conveyor produced a non-finite state")
        if any(warning.number for warning in self.d.warning):
            raise RuntimeError("MuJoCo reported a conveyor simulation warning")

    def key_callback(self, key: int) -> None:
        if key == 32:
            self.reverse()
        elif key == 80:
            self.paused = not self.paused
            self._update_surfaces()
        elif key == 259:
            self.reset()

    def run_headless(self, steps: int) -> None:
        for _ in range(steps):
            self.step()
        positions = self.d.xpos[self.bag_ids]
        on_carousel = np.linalg.norm(positions[:, :2], axis=1) < 2.2
        print(
            f"luggage carousel: time={self.d.time:.2f}s, bags={len(self.bag_ids)}, "
            f"on_carousel={np.count_nonzero(on_carousel)}, contacts={self.d.ncon}, "
            f"actuators={self.m.nu}, no runtime warnings"
        )

    def configure_camera(self, camera: mj.MjvCamera) -> None:
        camera.lookat[:] = [-0.2, -1.5, 0.25]
        camera.distance = 9.4
        camera.azimuth = 145
        camera.elevation = -38

    def run_viewer(self) -> None:
        import mujoco.viewer

        print(
            "Space: reverse | P: stop/start surfaces | Backspace: reset | Close viewer: exit"
        )
        keys = SimpleQueue()
        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=keys.put
        ) as viewer:
            self.configure_camera(viewer.cam)
            while viewer.is_running():
                start = time.perf_counter()
                with viewer.lock():
                    # Apply UI input on the simulation thread, between steps.
                    while not keys.empty():
                        self.key_callback(keys.get_nowait())
                    for _ in range(round(1 / (60 * self.m.opt.timestep))):
                        self.step()
                viewer.sync()
                time.sleep(max(0, 1 / 60 - (time.perf_counter() - start)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="headless steps (default: 20 simulation seconds)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.7,
        help="feeder belt speed in m/s; also scales the carousel",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    sim = Sim(speed=args.speed)
    if args.headless:
        sim.run_headless(args.steps)
    else:
        sim.run_viewer()


if __name__ == "__main__":
    main()
