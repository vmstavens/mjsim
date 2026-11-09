"""MuJoCo 4 native rod torsion and full-state replay; no external assets.

Run with --headless for a numerical replay check, or without flags to watch the
clamped rod twist. The terminal twist hinge rotates the rod's material frame;
contact friction couples this rotation to the rod surface.
"""

import argparse
import time

import mujoco as mj
import numpy as np
from physics.palette import FLAME, apply_palette

from mjsim.utils.mjs import cable, empty_scene


def build_model() -> mj.MjModel:
    scene = empty_scene(
        integrator="discrete",
        solver="CG",
        noslip_iterations=0,
        timestep=0.001,
        gravity=(0, 0, 0),
        nativeccd=True,
        statistic_center=(0.2, 0, 0.3),
        statistic_extent=0.5,
    )
    rod = cable(
        model_name="rod",
        curve="s 0 0",
        n_segments=17,
        length=0.4,
        segment_size=0.006,
        mass=0.1,
        bend=2e6,
        twist=7e5,
        initial="none",
        geom_rgba=FLAME,
    )
    scene.attach(rod, prefix="demo/", frame=scene.worldbody.add_frame(pos=[0, 0, 0.3]))
    # Native rods expose per-edge material-frame hinges, including this tip hinge.
    scene.add_actuator(
        name="tip_twist", target="demo/rod_twist", trntype=mj.mjtTrn.mjTRN_JOINT
    ).set_to_position(kp=0.02, kv=0.0002)
    model = scene.compile()
    apply_palette(model)
    return model


def step(model: mj.MjModel, data: mj.MjData) -> None:
    data.ctrl[0] = np.pi * np.sin(2 * data.time)
    mj.mj_step(model, data)


def check_replay(model: mj.MjModel, steps: int = 500) -> mj.MjData:
    data = mj.MjData(model)
    for _ in range(steps):
        step(model, data)
    # qpos and qvel alone omit time-parallel rod frames stored in plastic.
    signature = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(model, signature))
    mj.mj_getState(model, data, state, signature)
    restored = mj.MjData(model)
    mj.mj_setState(model, restored, state, signature)
    for _ in range(100):
        step(model, data)
        step(model, restored)
    np.testing.assert_array_equal(data.qpos, restored.qpos)
    np.testing.assert_array_equal(data.plastic, restored.plastic)
    if np.any(data.warning.number):
        raise RuntimeError("Rod simulation reported a warning")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    model = build_model()
    if args.headless:
        data = check_replay(model)
        print(
            f"Rod replay matched exactly at {data.time:.3f} s; {model.nplastic} frame-history values."
        )
        return
    import mujoco.viewer

    data = mj.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.monotonic()
            with viewer.lock():
                for _ in range(16):
                    step(model, data)
            viewer.sync()
            time.sleep(max(0, 0.016 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
