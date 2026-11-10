"""Render the snap-fit opening and a large force graph to an H.264 MP4.

MUJOCO_GL=egl uv run python examples/render_snap_fit.py
Requires ffmpeg on PATH. The rollout is the same OpeningPlan used by hotkey P.
"""

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import mujoco as mj
import numpy as np
from physics.palette import CARBON
from snap_fit import ForcePlot, OpeningPlan, apply_theme, build_spec, finger_force


def render_video(output: Path, duration: float = 11, fps: int = 30):
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        raise RuntimeError("ffmpeg is required to encode the video")
    width, height = 1280, 720
    spec = build_spec()
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    model = spec.compile()
    apply_theme(model)
    data = mj.MjData(model)
    plan = OpeningPlan(model)
    plan.start(data)
    plot = ForcePlot(window=12)
    camera = mj.MjvCamera()
    camera.lookat[:] = [0.005, 0, 0.18]
    camera.distance = 0.53
    camera.azimuth = 90
    camera.elevation = -8
    scene = mj.MjvScene(model, maxgeom=2000)
    option = mj.MjvOption()
    full = mj.MjrRect(0, 0, width, height)
    view = mj.MjrRect(0, 0, 740, height)
    chart = mj.MjrRect(740, 175, 540, 425)
    panel = mj.MjrRect(755, 10, 510, 155)
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    history = []
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            encoder,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    gl = mj.GLContext(width, height)
    gl.make_current()
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_200)
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, context)
    try:
        for frame in range(round(duration * fps)):
            target_step = round(frame / fps / model.opt.timestep)
            while data.time + model.opt.timestep / 2 < target_step * model.opt.timestep:
                plan.update(data)
                mj.mj_step(model, data)
                force = finger_force(model, data)
                plot.observe(force)
                history.append(
                    (data.time, *force, np.linalg.norm(force), data.qpos[plan.qpos])
                )
                if np.any(data.warning.number):
                    raise RuntimeError(f"Simulation warning at {data.time:.3f} s")
            plot.update(data.time, plan.phase)
            # Video uses elapsed time so the whole opening stays on the graph.
            for line in range(2):
                count = plot.figure.linepnt[line]
                plot.figure.linedata[line, : 2 * count : 2] += data.time
            plot.figure.range[0] = [0, duration]
            mj.mjv_updateScene(
                model, data, option, None, camera, mj.mjtCatBit.mjCAT_ALL, scene
            )
            mj.mjr_rectangle(full, *CARBON)
            mj.mjr_render(view, scene, context)
            mj.mjr_figure(chart, plot.figure, context)
            mj.mjr_overlay(
                mj.mjtFont.mjFONT_NORMAL,
                mj.mjtGridPos.mjGRID_TOPLEFT,
                view,
                f"Snap fit | {plan.phase}\nTime: {data.time:.2f} s",
                "",
                context,
            )
            mj.mjr_overlay(
                mj.mjtFont.mjFONT_NORMAL,
                mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                panel,
                f"Lid angle: {np.degrees(data.qpos[plan.qpos]):.1f} deg\n"
                "Measured finger load only\nHinge assist: max 0.045 Nm\n(not included in force graph)",
                "",
                context,
            )
            mj.mjr_readPixels(pixels, None, full, context)
            process.stdin.write(np.flipud(pixels).tobytes())
            if frame % fps == 0:
                print(f"Rendered {frame / fps:.0f}/{duration:.0f} s", flush=True)
    finally:
        process.stdin.close()
        returncode = process.wait()
        context.free()
        gl.free()
    if returncode:
        raise RuntimeError(f"ffmpeg exited with status {returncode}")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "fx_N", "fy_N", "fz_N", "force_N", "lid_angle_rad"])
        writer.writerows(history)
    print(f"Saved {output} and {csv_path}; peak finger force {plot.peak:.3f} N")


def render_gif(video: Path):
    """Use the model view for a readable 4:3 README thumbnail."""
    output = video.with_name("snap_fit.gif")
    filters = (
        "fps=12,crop=740:720:0:0,"
        "scale=360:270:force_original_aspect_ratio=decrease,"
        "pad=360:270:(ow-iw)/2:(oh-ih)/2:color=0x111214,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a"
    )
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-filter_complex",
            filters,
            "-loop",
            "0",
            str(output),
        ],
        check=True,
    )
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "media" / "snap_fit_opening.mp4",
    )
    args = parser.parse_args()
    render_video(args.output)
    render_gif(args.output)


if __name__ == "__main__":
    main()
