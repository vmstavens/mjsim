"""Inspect the red fingertip hand with 85 STF-inspired 3D force taxels.

Run: python examples/shadow_hand_jar.py
Use --headless for a smoke check. The viewer's actuator sliders control individual
joints. C/O curl/open fingers, Up/Down move the mount by 2 mm, T toggles taxels,
P prints readings, and R resets. Ctrl + right-drag applies forces to the lid.
Add --ik for MINK fingertip tracking. Double-click a target sphere, then use
Ctrl + right-drag to move it (Shift selects the other translation plane).
I toggles IK; C/O and actuator sliders are for manual mode only.
"""

import argparse
from pathlib import Path
import time

import mujoco as mj
import numpy as np


SCENE = Path(__file__).parent / "assets" / "shadow_hand" / "scene.xml"


def build_model():
    model = mj.MjModel.from_xml_path(str(SCENE))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


def readings(model, data, tactile=None):
    """Copy native sensors and, if supplied, the latest 100 Hz STF sample."""
    result = {
        model.sensor(i).name: data.sensor(i).data.copy() for i in range(model.nsensor)
    }
    if tactile is not None:
        result.update(tactile.readings())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--ik", action="store_true", help="Enable MINK fingertip targets"
    )
    args = parser.parse_args()
    model, data = build_model()
    from shadow_hand_ik import HandIK
    from shadow_hand_tactile import FingertipTactile

    ik = HandIK(model, data)
    tactile = FingertipTactile(model, data)
    tactile.update()
    ik_enabled = args.ik
    if args.headless:
        for _ in range(100):
            if ik_enabled:
                ik.step(0.01)
            for _ in range(10):
                mj.mj_step(model, data)
            tactile.update()
        assert np.isfinite(data.qpos).all()
        print(
            f"Loaded {model.nu} actuators, {model.nsensor} native sensors, "
            f"85 three-axis taxels; t={data.time:.3f}s"
        )
        return

    import glfw
    import mujoco.viewer

    mount = model.body("rh_forearm_base")
    initial_pos = mount.pos.copy()
    pending = []
    print(__doc__)
    with mujoco.viewer.launch_passive(
        model, data, key_callback=pending.append
    ) as viewer:
        viewer.cam.lookat[:] = [0, 0.02, 0.2]
        viewer.cam.distance = 0.65
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -30
        viewer.opt.sitegroup[3] = 1
        viewer.opt.geomgroup[4] = int(ik_enabled)
        while viewer.is_running():
            start = time.monotonic()
            with viewer.lock():
                while pending:
                    key = pending.pop(0)
                    if key == glfw.KEY_R:
                        mj.mj_resetData(model, data)
                        mount.pos[:] = initial_pos
                        ik.reset()
                        tactile.reset()
                        tactile.update()
                    elif key == glfw.KEY_I:
                        ik_enabled = not ik_enabled
                        if ik_enabled:
                            ik.reset()
                        viewer.opt.geomgroup[4] = int(ik_enabled)
                        print(f"IK {'enabled' if ik_enabled else 'disabled'}")
                    elif key in (glfw.KEY_UP, glfw.KEY_DOWN):
                        mount.pos[2] += 0.002 if key == glfw.KEY_UP else -0.002
                        ik.reset()
                        tactile.reset()
                        tactile.update()
                    elif key == glfw.KEY_T:
                        viewer.opt.sitegroup[3] ^= 1
                    elif key in (glfw.KEY_C, glfw.KEY_O) and not ik_enabled:
                        for i in range(model.nu):
                            name = model.actuator(i).name
                            if any(
                                f"{finger}J{joint}" in name
                                for finger in ("FF", "MF", "RF", "LF")
                                for joint in (1, 2, 3)
                            ):
                                lo, hi = model.actuator_ctrlrange[i]
                                data.ctrl[i] = np.clip(
                                    data.ctrl[i] + (0.1 if key == glfw.KEY_C else -0.1),
                                    lo,
                                    hi,
                                )
                    elif key == glfw.KEY_P:
                        for finger in ("ff", "mf", "rf", "lf", "th"):
                            print(f"{finger} local xyz [N], t={tactile.sample_time}:")
                            print(np.round(tactile.forces[finger], 3))
                        print(
                            "lid angle/velocity:",
                            data.sensor("lid_angle").data,
                            data.sensor("lid_velocity").data,
                        )
                if ik_enabled:
                    ik.step(0.01)
                for _ in range(10):
                    mj.mj_step(model, data)
                tactile.update()
                for finger, ids in tactile.site_ids.items():
                    for site, force in zip(ids, tactile.forces[finger]):
                        active = np.linalg.norm(force) > 0.001
                        model.site_rgba[site] = (
                            [0.4, 0.65, 0.5, 0.45]
                            if active else [0.65, 0.67, 0.7, 0.15]
                        )
            viewer.sync()
            time.sleep(max(0, 0.01 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
