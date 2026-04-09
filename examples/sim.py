import mujoco as mj
import mujoco.viewer
from robot_descriptions import robotiq_2f85_mj_description, ur5e_mj_description

from mjsim.base.robot import Robot
from mjsim.utils.mjs import empty_scene


class Sim:
    def __init__(self):
        self.m, self.d = self._build_model()

        self.robot = Robot(self.m, self.d, "arm")
        self.gripper = Robot(self.m, self.d, "gripper")

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData]:
        scene = empty_scene()

        arm = mj.MjSpec.from_file(ur5e_mj_description.MJCF_PATH)

        gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)

        scene.worldbody.add_frame(name="arm").attach_body(
            arm.worldbody.first_body(), prefix="arm/"
        )

        s_attachment_site = arm.site("arm/attachment_site")

        s_attachment_site.attach_body(gripper.worldbody.first_body(), prefix="gripper")

        m = scene.compile()
        d = mj.MjData(m)

        return m, d

    def cb(self, key: int) -> None:
        pass

    def run(self) -> None:

        print(self.gripper.info)
        quit()

        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.cb
        ) as viewer:
            while viewer.is_running():
                mj.mj_step(self.m, self.d)

                viewer.sync()


if __name__ == "__main__":
    sim = Sim()
    sim.run()
