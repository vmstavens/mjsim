"""
Minimal camera capture demo.

Builds a tiny MuJoCo scene with a ground plane and a camera, then grabs RGB and
depth images using `mjsim.sensors.camera.Camera`. Prints image shapes instead of
opening a viewer so it runs headless.
"""

from importlib import resources
from pathlib import Path

import numpy as np

try:
    import mujoco as mj
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    raise SystemExit("MuJoCo is not installed. Install `mujoco` to run this example.")

from mjsim.sensors.camera import Camera


def _build_model() -> tuple[mj.MjModel, mj.MjData]:
    """Create a minimal scene that includes a camera named 'front'."""
    xml = """
    <mujoco model="camera_demo">
      <option timestep="0.002" gravity="0 0 -9.81" />
      <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0" />
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.6 0.7 0.8" rgb2="0.2 0.3 0.4" width="512" height="512"/>
      </asset>
      <worldbody>
        <light pos="1 0 1" dir="-1 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.2" type="plane" rgba="0.9 0.9 0.9 1"/>
        <geom name="box" type="box" size="0.05 0.05 0.05" pos="0.2 0.0 0.05" rgba="0.8 0.2 0.2 1" />
        <camera name="front" pos="0.6 0 0.3" euler="0 0 3.14"/>
      </worldbody>
    </mujoco>
    """
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    return model, data


def main() -> None:
    model, data = _build_model()
    cam = Camera(
        model, data, cam_name="front", width=320, height=240, save_dir="tmp/cam/"
    )

    # Step once to ensure the scene is initialized.
    mj.mj_forward(model, data)

    rgb = cam.image
    depth = cam.depth_image

    print("RGB image shape:", rgb.shape, "dtype:", rgb.dtype)
    print("Depth image shape:", depth.shape, "dtype:", depth.dtype)


if __name__ == "__main__":
    main()
