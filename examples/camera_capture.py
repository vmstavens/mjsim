"""
Camera capture demo.

Run the simulation, press C in the MuJoCo viewer, and the camera image opens in
an OpenCV window.
"""

import cv2
import glfw
import mjsim as ms
import mujoco as mj

from mjsim.sensors.camera import Camera


class Sim(ms.BaseSim):
    def __init__(self) -> None:
        super().__init__()
        self._model, self._data = self._init()
        self.camera = Camera(
            self.model,
            self.data,
            cam_name="front",
            width=640,
            height=480,
            save_dir="tmp/cam/",
        )

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:
        scene = ms.empty_scene(sim_name="camera_demo")

        b_target = scene.worldbody.add_body(name="box")

        b_target.add_geom(
            name="box",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.05, 0.05, 0.05],
            pos=[0.2, 0.0, 0.05],
            rgba=[0.8, 0.2, 0.2, 1.0],
        )
        scene.worldbody.add_camera(
            name="front",
            pos=[0.6, 0.0, 0.3],
            euler=[0.0, 0.0, 3.14],
            fovy=45.0,
            targetbody="box",
            mode=mj.mjtCamLight.mjCAMLIGHT_TARGETBODY
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
        if key == glfw.KEY_C:
            rgb = self.camera.image
            cv2.imshow("front camera", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            print(f"Captured image from camera 'front': {rgb.shape} {rgb.dtype}")

        if key == glfw.KEY_Q:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sim = Sim()
    print("Press C in the MuJoCo viewer to capture the camera image.")
    print("Press Q to close OpenCV image windows.")
    sim.run()
