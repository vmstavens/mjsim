"""Robotis HX5-D20 hand over a passive jar lid, with native 3x3 tactile grids.

Run: python examples/robotis_hand_jar.py [--ik] [--headless]
C/O curl/open all five fingers; actuator sliders control individual joints.
Up/Down move the mount by 2 mm. T toggles sensing volumes, P prints readings,
and R resets. Ctrl + right-drag applies forces to the lid.
I toggles MINK fingertip IK. Double-click a target sphere, then Ctrl + right-drag
to move it (Shift selects the other translation plane). Manual controls are
overwritten while IK is enabled. The overlay shows normal forces in newtons.
"""

import argparse
import time
from pathlib import Path

import mujoco as mj
import numpy as np

SCENE = Path(__file__).parent / "assets" / "robotis_hand" / "scene.xml"
FINGERS = ("thumb", "index", "middle", "ring", "little")
CONTROL_DT = 0.01


def build_model():
    model = mj.MjModel.from_xml_path(str(SCENE))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


def readings(model, data):
    """Copy all native sensor readings, including taxels and lid sensors."""
    return {
        model.sensor(i).name: data.sensor(i).data.copy() for i in range(model.nsensor)
    }


class FingertipTactile:
    """Native normal-force readings in N, ordered thumb/index/middle/ring/little.

    Each 3x3 grid has distal-to-proximal rows. Columns increase along local X
    for the thumb and local Y for other fingers. The upstream sensing volumes
    are approximate surface zones, not calibrated hardware or shear sensors.
    """

    def __init__(self, model, data):
        self.model, self.data = model, data
        names = [
            f"taxel_{f}_{r}_{c}" for f in range(5) for r in range(3) for c in range(3)
        ]
        self.sensor_addresses = np.array([model.sensor(n).adr[0] for n in names])
        self.site_ids = np.array([model.site(n).id for n in names]).reshape(5, 3, 3)

    @property
    def forces(self):
        """Copy the current (5, 3, 3) normal-force array from sensordata."""
        return self.data.sensordata[self.sensor_addresses].reshape(5, 3, 3).copy()

    def color_sites(self):
        strength = np.clip(self.forces / 10.0, 0, 1).reshape(-1)
        self.model.site_rgba[self.site_ids.ravel()] = np.column_stack(
            (
                0.2 + 0.8 * strength,
                0.6 - 0.4 * strength,
                0.9 - 0.8 * strength,
                0.12 + 0.5 * strength,
            )
        )

    def text(self):
        lines = ["Tactile normal force [N]"]
        for finger, grid in zip(FINGERS, self.forces):
            lines.append(finger)
            lines.extend("  ".join(f"{v:5.2f}" for v in row) for row in grid)
        return "\n".join(lines)


def curl_fingers(model, data, increment):
    """Increment bounded flexion targets, respecting the thumb's reversed joint."""
    for finger in range(5):
        for joint in range(1, 4):
            actuator = model.actuator(f"finger_{finger}_joint_{joint}_right").id
            direction = -1 if finger == 0 and joint == 1 else 1
            scale = 0.65 if joint == 1 else 1.0
            data.ctrl[actuator] = np.clip(
                data.ctrl[actuator] + direction * scale * increment,
                *model.actuator_ctrlrange[actuator],
            )


def step(model, data, ik=None):
    """Advance one 10 ms control interval and refresh native contact sensors."""
    if ik is not None:
        ik.step(CONTROL_DT)
    for _ in range(round(CONTROL_DT / model.opt.timestep)):
        mj.mj_step(model, data)
    mj.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--ik", action="store_true", help="Enable MINK fingertip targets"
    )
    args = parser.parse_args()
    model, data = build_model()
    tactile = FingertipTactile(model, data)
    ik = None

    def enable_ik():
        from robotis_hand_ik import HandIK

        return HandIK(model, data)

    if args.ik:
        ik = enable_ik()
    if args.headless:
        for _ in range(100):
            step(model, data, ik)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.sensordata).all()
            or np.any(data.warning.number)
        ):
            raise RuntimeError("Non-finite state or MuJoCo warning during smoke check")
        print(
            f"Loaded {model.nu} actuators, 45 native tactile taxels "
            f"(5 x 3 x 3), {model.nsensor} total sensors; t={data.time:.3f}s"
        )
        return

    import glfw
    import mujoco.viewer

    mount = model.body("hand_base_right")
    initial_pos = mount.pos.copy()
    pending = []
    print(__doc__)
    with mujoco.viewer.launch_passive(
        model, data, key_callback=pending.append
    ) as viewer:
        viewer.cam.lookat[:] = [0.04, 0.02, 0.2]
        viewer.cam.distance = 0.6
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -30
        viewer.opt.sitegroup[3] = 0
        viewer.opt.geomgroup[4] = int(ik is not None)
        while viewer.is_running():
            start = time.monotonic()
            with viewer.lock():
                while pending:
                    key = pending.pop(0)
                    if key == glfw.KEY_R:
                        mj.mj_resetData(model, data)
                        mount.pos[:] = initial_pos
                        mj.mj_forward(model, data)
                        if ik is not None:
                            ik.reset()
                    elif key == glfw.KEY_I:
                        ik = enable_ik() if ik is None else None
                        viewer.opt.geomgroup[4] = int(ik is not None)
                        print(f"IK {'enabled' if ik is not None else 'disabled'}")
                    elif key in (glfw.KEY_UP, glfw.KEY_DOWN):
                        mount.pos[2] += 0.002 if key == glfw.KEY_UP else -0.002
                        mj.mj_forward(model, data)
                        if ik is not None:
                            ik.reset()
                    elif key == glfw.KEY_T:
                        viewer.opt.sitegroup[3] ^= 1
                    elif key in (glfw.KEY_C, glfw.KEY_O) and ik is None:
                        curl_fingers(model, data, 0.1 if key == glfw.KEY_C else -0.1)
                    elif key == glfw.KEY_P:
                        print(tactile.text())
                        print(
                            "lid angle/velocity:",
                            data.sensor("lid_angle").data,
                            data.sensor("lid_velocity").data,
                        )
                step(model, data, ik)
                tactile.color_sites()
                overlay = tactile.text()
                overlay += f"\nLid: {data.sensor('lid_angle').data[0]:.3f} rad"
                overlay += f"\nIK: {'on' if ik is not None else 'off'}"
            viewer.set_texts(
                (
                    mj.mjtFontScale.mjFONTSCALE_100,
                    mj.mjtGridPos.mjGRID_TOPLEFT,
                    overlay,
                    "",
                )
            )
            viewer.sync()
            time.sleep(max(0, CONTROL_DT - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
