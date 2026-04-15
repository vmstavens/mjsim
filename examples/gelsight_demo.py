"""
GelSight Mini tactile image demo.

Run the simulation, press G in the MuJoCo viewer, and the raw internal camera,
tactile image, and penetration depth image open in OpenCV windows.
"""

import cv2
import glfw
import mjsim as ms
import mujoco as mj
import numpy as np

from mjsim.sensors.gelsight_mini import GelSightMini


class Sim(ms.BaseSim):
    def __init__(self) -> None:
        super().__init__()
        self._model, self._data = self._init()
        self.gelsight = GelSightMini(
            self.model,
            self.data,
            cam_name="gelsight",
        )

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:
        scene = ms.empty_scene(
            sim_name="gelsight_demo",
            gravity=[0.0, 0.0, -9.82],
            statistic_center=[0.0, 0.0, 0.0],
            statistic_extent=0.12,
        )

        shell = ms.mesh(
            "examples/assets/gsmini_shell.obj",
            model_name="gelsight_shell",
            scale=0.001,
            decimation_method="none",
            collision=True,
            solref=[0.01,4]
        )

        b_gelsight = scene.worldbody.add_body(
            name="gelsight",
            pos=[0.0, 0.0, 0.0],
            euler=[3.14, 0.0, 0.0],
        )
        shell_frame = b_gelsight.add_frame()
        scene.attach(shell, prefix="gelsight_shell/", frame=shell_frame)

        b_gelsight.add_camera(
            name="gelsight",
            pos=[0.0, 0.0, 0],
            fovy=70.0,
            targetbody="indenter",
            mode=mj.mjtCamLight.mjCAMLIGHT_TARGETBODY,
        )

        b_indenter = scene.worldbody.add_body(name="indenter", pos=[0.0, 0.0, 0.1])
        b_indenter.add_joint(
            name="indenter",
            type=mj.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, 1.0],
        )
        b_indenter.add_geom(
            name="indenter",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.008],
            rgba=[0.9, 0.15, 0.1, 1.0],
        )

        model = scene.compile()
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        return model, data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def data(self) -> mj.MjData:
        return self._data

    def keyboard_callback(self, key: int) -> None:
        if key == glfw.KEY_G:
            rgb = self.gelsight.image
            tactile = self.gelsight.tactile_image
            depth = self.gelsight.tactile_depth_image

            depth_vis = depth.copy()
            depth_vis -= np.min(depth_vis)
            max_depth = np.max(depth_vis)
            if max_depth > 0.0:
                depth_vis /= max_depth
            depth_vis = (255.0 * depth_vis).astype(np.uint8)

            cv2.imshow("gelsight raw camera", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imshow(
                "gelsight tactile image",
                cv2.cvtColor(tactile, cv2.COLOR_RGB2BGR),
            )
            cv2.imshow("gelsight penetration depth", depth_vis)
            cv2.waitKey(1)
            print(
                "Captured GelSight images: "
                f"raw={rgb.shape} tactile={tactile.shape} depth={depth.shape}"
            )

        if key == glfw.KEY_Q:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sim = Sim()
    print("Press G in the MuJoCo viewer to capture GelSight images.")
    print("Press Q to close OpenCV image windows.")
    sim.run()
