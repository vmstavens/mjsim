"""Render real simulation rollouts as the README's animated GIF previews.

On a headless Linux host, set MUJOCO_GL=egl before running this script.
Pillow is only needed for this exporter, not for the gallery launcher.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

import mujoco as mj
import numpy as np
from physics.catalog import EXAMPLES, configure_camera, drive, load_example
from PIL import Image

EXTRA_DURATIONS = {
    "conveyor_belt": 16,
    "mj_xml": 3,
    "particle_container": 7,
    "velux_sealing": 3,
    "mocap_gripper_control": 14,
}
PREVIEWS = (*EXAMPLES, *EXTRA_DURATIONS)


def render_gif(name: str, output: Path, duration: float, fps: int) -> None:
    camera = mj.MjvCamera()
    if name in EXTRA_DURATIONS:
        module = importlib.import_module(name)
        if name == "particle_container":
            # A lighter 120-particle grid keeps README regeneration practical.
            sim = module.Sim(module.particle_grid(128))
        elif name == "mocap_gripper_control":
            sim = module.Sim(manual=False)
        else:
            sim = module.Sim()
        model, data = (sim.model, sim.data) if name == "mj_xml" else (sim.m, sim.d)
        if name == "conveyor_belt":
            sim.configure_camera(camera)
        else:
            lookat, distance, azimuth, elevation = {
                "mj_xml": ([0, 0, 0.4], 2.1, 125, -25),
                "particle_container": ([0, 0, 0.11], 0.82, 135, -25),
                "velux_sealing": ([0, 0.45, 0.22], 1.35, 125, -22),
                "mocap_gripper_control": ([0, 0, 0.23], 0.9, 135, -20),
            }[name]
            camera.lookat[:] = lookat
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation

        def step():
            if name == "mj_xml":
                mj.mj_step(model, data)
            else:
                sim.step()
    else:
        model, data = load_example(name)
        configure_camera(name, model, camera)

        def step():
            drive(name, model, data)
            mj.mj_step(model, data)

    mj.mj_forward(model, data)
    frames = []
    with mj.Renderer(model, height=270, width=360) as renderer:
        for frame in range(math.ceil(duration * fps)):
            target_time = frame / fps
            while data.time < target_time:
                step()
                # MuJoCo can reset time after instability. Check immediately so
                # a reset cannot trap the exporter below the target timestamp.
                if any(warning.number for warning in data.warning):
                    raise RuntimeError(f"{name}: MuJoCo reported a runtime warning")
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(f"{name}: non-finite simulation state")
            if any(warning.number for warning in data.warning):
                raise RuntimeError(f"{name}: MuJoCo reported a runtime warning")
            renderer.update_scene(data, camera=camera)
            frames.append(Image.fromarray(renderer.render()))

    # Use one palette across the rollout to avoid color flicker between frames.
    samples = frames[:: max(1, len(frames) // 10)]
    strip = Image.new("RGB", (120 * len(samples), 90))
    for index, frame in enumerate(samples):
        strip.paste(frame.resize((120, 90)), (120 * index, 0))
    palette = strip.quantize(colors=256)
    frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
        disposal=2,
        optimize=True,
    )
    print(
        f"{name}: {len(frames)} frames, {output.stat().st_size / 1024:.0f} KiB -> {output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "examples", nargs="*", help="preview names; default: all documented previews"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="override simulation seconds; default: each preview's full sequence",
    )
    parser.add_argument("--fps", type=int, choices=(10, 20, 25, 50), default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "media"
    )
    args = parser.parse_args()
    if args.duration is not None and (
        not math.isfinite(args.duration) or args.duration <= 0
    ):
        parser.error("--duration must be finite and greater than zero")
    names = args.examples or list(PREVIEWS)
    if unknown := set(names) - set(PREVIEWS):
        parser.error(f"unknown examples: {', '.join(sorted(unknown))}")
    for name in names:
        duration = (
            args.duration if args.duration is not None else EXTRA_DURATIONS.get(name, 4)
        )
        render_gif(name, args.output_dir / f"{name}.gif", duration, args.fps)


if __name__ == "__main__":
    main()
