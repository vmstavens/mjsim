import glfw
import mjsim as ms
import mujoco as mj
import spatialmath as sm
from robot_descriptions import robotiq_2f85_mj_description


class Sim(ms.BaseSim):
    def __init__(self):
        self._model, self._data = self._init()

        self.gripper = ms.Robot(self.model, self.data, "gripper/")

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:

        scene = ms.empty_scene(memory="100M")

        # load gripper
        gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)
        f_gripper = scene.worldbody.add_frame()
        scene.attach(gripper, "gripper/", frame=f_gripper)

        # load cable
        cable = ms.cable()
        f_cable = scene.worldbody.add_frame(pos=[-0.4, 0, 0])
        scene.attach(cable, prefix="cable/", frame=f_cable)

        # load cloth
        cloth = ms.cloth(pin_corner=True)
        f_cloth = scene.worldbody.add_frame(pos=[-0.6, 0, 1])
        scene.attach(cloth, prefix="", frame=f_cloth)

        # load jello
        jello = ms.jello()
        f_jello = scene.worldbody.add_frame(pos=[-1, 0, 0.4])
        scene.attach(jello, prefix="jello/", frame=f_jello)

        # build object
        b_ball = scene.worldbody.add_body(name="ball", pos=[0.1, 0, 1])
        b_ball.add_geom(name="ball", size=[0.01])
        b_ball.add_freejoint()

        m = scene.compile()
        return m, mj.MjData(m)

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    def keyboard_callback(self, key):
        if key is glfw.KEY_SPACE:
            print("Placing ball at [0, 0, 1]...")
            ms.set_pose(self.model, self.data, "ball", ms.ObjType.BODY, sm.SE3.Tz(1))
        if key is glfw.KEY_PERIOD:
            print("Printing gripper information:")
            print(self.gripper.info)

    @ms.thread
    def see_me_run(self, ss: ms.SimSync):
        while True:
            ss.step()


if __name__ == "__main__":
    sim = Sim()

    sim.run()
