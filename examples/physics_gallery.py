"""Run `python examples/physics_gallery.py --list` to discover the gallery."""

from __future__ import annotations

import argparse
import math
import time

import mujoco as mj
import numpy as np
from physics.catalog import (
    EXAMPLES,
    configure_camera,
    drive,
    load_example,
    track_camera,
)


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", nargs="?", choices=EXAMPLES, default="newton_cradle")
    parser.add_argument(
        "--list", action="store_true", help="list the physical systems and exit"
    )
    parser.add_argument(
        "--headless", action="store_true", help="simulate without graphics"
    )
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=5,
        help="headless simulation seconds (default: 5)",
    )
    parser.add_argument(
        "--speed",
        type=positive_float,
        default=1,
        help="viewer simulation seconds per wall second",
    )
    parser.add_argument(
        "--manual", action="store_true", help="leave actuator inputs to viewer controls"
    )
    parser.add_argument(
        "--original-colors", action="store_true", help="use upstream colors"
    )
    args = parser.parse_args()
    if args.list:
        for name, example in EXAMPLES.items():
            print(f"{name:16} {example.description}")
        return

    model, data = load_example(args.example, original_colors=args.original_colors)
    print(EXAMPLES[args.example].description)
    print(EXAMPLES[args.example].observe)

    def step() -> None:
        if not args.manual:
            drive(args.example, model, data)
        mj.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Simulation produced a non-finite state")
        if any(warning.number for warning in data.warning):
            raise RuntimeError("MuJoCo reported a simulation warning")

    if args.headless:
        for _ in range(math.ceil(args.duration / model.opt.timestep)):
            step()
        print(
            f"Simulated {data.time:.3f} s; {model.nv} DoFs; {data.ncon} final contacts; no runtime warnings."
        )
        return

    # Import lazily: headless execution needs no windowing backend.
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_camera(args.example, model, viewer.cam)
        accumulated = 0.0
        while viewer.is_running():
            start = time.monotonic()
            accumulated += args.speed / 60
            steps = int(accumulated / model.opt.timestep)
            accumulated -= steps * model.opt.timestep
            with viewer.lock():
                for _ in range(steps):
                    step()
                track_camera(args.example, model, data, viewer.cam)
            viewer.sync()
            time.sleep(max(0, 1 / 60 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
