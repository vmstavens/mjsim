import glfw
import mjsim as ms
import mujoco as mj
from robot_descriptions import ur10e_mj_description, robotiq_2f85_mj_description
import spatialmath as sm

class Sim(ms.BaseSim):
    def __init__(self):
        self._model, self._data = self._init()

        self.ur = ms.Robot(self.model, self.data, "ur/")
        self.gripper = ms.Robot(self.model, self.data, "ur/gripper/")

        T = ms.get_pose(self.model, self.data, self.ur.info.site_names[0], ms.ObjType.SITE)

        self.ctrl = ms.OpSpace(self.ur, gravity_comp=True)

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:

        scene = ms.empty_scene()

        ur = mj.MjSpec.from_file(ur10e_mj_description.MJCF_PATH)

        gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)

        s_attachment = ur.site("attachment_site")

        s_attachment.attach_body(gripper.worldbody.first_body(),prefix="gripper/")

        f_ur = scene.worldbody.add_frame()

        scene.attach(ur, "ur/", frame=f_ur)

        b_ball = scene.worldbody.add_body(name="ball", pos=[0,0,1])
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