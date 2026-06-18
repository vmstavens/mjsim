import argparse

import mujoco as mj
import numpy as np
import spatialmath as sm
from robot_descriptions import robotiq_2f85_mj_description, ur10e_mj_description

import glfw
import mjsim as ms


class ForceTorqueSensor:
    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        site_name: str,
        force_sensor_name: str,
        torque_sensor_name: str,
    ):
        self.model = model
        self.data = data
        self.site_id = self.model.site(site_name).id
        self.force_adr = self.model.sensor(force_sensor_name).adr[0]
        self.torque_adr = self.model.sensor(torque_sensor_name).adr[0]

    def wrench(self, frame: str = "world") -> np.ndarray:
        if frame not in {"world", "site"}:
            raise ValueError("frame must be 'world' or 'site'")

        force = self.data.sensordata[self.force_adr : self.force_adr + 3].copy()
        torque = self.data.sensordata[self.torque_adr : self.torque_adr + 3].copy()

        if frame == "world":
            site_rot = self.data.site_xmat[self.site_id].reshape(3, 3)
            force = site_rot @ force
            torque = site_rot @ torque

        return np.concatenate([force, torque])


class Sim(ms.BaseSim):
    def __init__(self, with_gripper: bool = True, with_poker: bool = True):
        self.with_gripper = with_gripper
        self.with_poker = with_poker
        self._model, self._data = self._init(
            with_gripper=with_gripper,
            with_poker=with_poker,
        )
        mj.mj_forward(self.model, self.data)

        self.ur = ms.Robot(self.model, self.data, "ur/")
        self.gripper = (
            ms.Robot(self.model, self.data, "ur/gripper/") if with_gripper else None
        )
        self.ft = ForceTorqueSensor(
            self.model,
            self.data,
            site_name="ur/tool_loadcell_site",
            force_sensor_name="ur/tool_force",
            torque_sensor_name="ur/tool_torque",
        )
        self.tool_loadcell_body_id = self.model.body("ur/tool_loadcell").id
        self.poker_actuator_id = (
            self.model.actuator("poker_position").id if with_poker else None
        )
        self.poker_joint_id = self.model.joint("poker_slide").id if with_poker else None
        self.step_count = 0

        _ = ms.get_pose(
            self.model, self.data, self.ur.info.site_names[0], ms.ObjType.SITE
        )

        # self.ctrl = ms.OpSpace(self.ur, gravity_comp=True)

    def _init(
        self,
        with_gripper: bool,
        with_poker: bool,
    ) -> tuple[mj.MjModel, mj.MjData]:

        scene = ms.empty_scene()

        ur = mj.MjSpec.from_file(ur10e_mj_description.MJCF_PATH)

        bodies = ur.worldbody.find_all("body")
        wrist_3: mj.MjsBody = bodies[-1]
        s_attachment = ur.site("attachment_site")

        loadcell = wrist_3.add_body(name="tool_loadcell")
        loadcell.add_geom(
            name="tool_loadcell_geom",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            pos=s_attachment.pos,
            size=[0.01],
            mass=1e-6,
            contype=0,
            conaffinity=0,
            rgba=[0.2, 0.2, 1, 1],
        )
        s_loadcell = loadcell.add_site(
            name="tool_loadcell_site",
            pos=s_attachment.pos,
            quat=s_attachment.quat,
            size=[0.015],
            rgba=[0, 1, 0, 1],
        )
        ur.add_sensor(
            name="tool_force",
            type=mj.mjtSensor.mjSENS_FORCE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="tool_loadcell_site",
            dim=3,
        )
        ur.add_sensor(
            name="tool_torque",
            type=mj.mjtSensor.mjSENS_TORQUE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="tool_loadcell_site",
            dim=3,
        )

        if with_gripper:
            gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)
            s_loadcell.attach_body(gripper.worldbody.first_body(), prefix="gripper/")

        f_ur = scene.worldbody.add_frame()

        scene.attach(ur, "ur/", frame=f_ur)

        b_ball = scene.worldbody.add_body(name="ball", pos=[0, 0, 1])
        b_ball.add_geom(name="ball", size=[0.01])
        b_ball.add_freejoint()

        if with_poker:
            self._add_poking_finger(scene)

        m = scene.compile()
        d = mj.MjData(m)
        mj.mj_resetDataKeyframe(m, d, m.keyframe("ur/home").id)
        return m, d

    def _add_poking_finger(self, scene: mj.MjSpec) -> None:
        poker = scene.worldbody.add_body(name="poker", pos=[-0.127, 0.691, 0.46])
        poker.add_joint(
            name="poker_slide",
            type=mj.mjtJoint.mjJNT_SLIDE,
            axis=[0, 0, 1],
            range=[0.0, 0.3],
            damping=2.0,
        )
        poker.add_geom(
            name="poker_finger",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.018, 0.018, 0.04],
            mass=0.05,
            rgba=[0.9, 0.15, 0.1, 1],
            friction=[1.0, 0.005, 0.0001],
            solref=[0.01, 1],
            solimp=[0.95, 0.99, 0.001, 0.5, 2],
        )
        scene.add_actuator(
            name="poker_position",
            target="poker_slide",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=[0.0, 0.30],
            forcerange=[-80, 80],
        ).set_to_position(kp=500, kv=50)

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def expected_sensor_weight(self) -> float:
        return (
            self.model.body_subtreemass[self.tool_loadcell_body_id]
            * -self.model.opt.gravity[2]
        )

    def poker_target(self) -> float:
        if not self.with_poker:
            return 0.0

        if self.data.time < 1.0:
            return 0.0

        phase = (self.data.time - 1.0) % 4.0
        if phase < 1.0:
            return 0.08 * phase
        if phase < 2.0:
            return 0.08
        if phase < 3.0:
            return 0.08 * (3.0 - phase)
        return 0.0

    def update_poker_control(self) -> None:
        if self.poker_actuator_id is None:
            return
        self.data.ctrl[self.poker_actuator_id] = self.poker_target()

    def poker_position(self) -> float:
        if self.poker_joint_id is None:
            return 0.0
        qpos_adr = self.model.joint(self.poker_joint_id).qposadr[0]
        return float(self.data.qpos[qpos_adr])

    def control_loop(self):
        self.update_poker_control()
        self.step_count += 1
        if self.step_count % 100 != 0:
            return

        wrench = self.ft.wrench()
        print(
            "tool FT world [N, N*m]: "
            f"F=({wrench[0]: .4f}, {wrench[1]: .4f}, {wrench[2]: .4f}) | "
            f"T=({wrench[3]: .4f}, {wrench[4]: .4f}, {wrench[5]: .4f}) | "
            f"expected Fz from mounted mass ~= {self.expected_sensor_weight:.4f} | "
            f"poker q={self.poker_position(): .4f}"
        )

    def keyboard_callback(self, key):
        if key is glfw.KEY_SPACE:
            print("Placing ball at [0, 0, 1]...")
            ms.set_pose(self.model, self.data, "ball", ms.ObjType.BODY, sm.SE3.Tz(1))
        if key is glfw.KEY_PERIOD:
            if self.gripper is None:
                print("No gripper is attached. Run without --no-gripper to include it.")
            else:
                print("Printing gripper information:")
                print(self.gripper.info)
        if key is glfw.KEY_F:
            print("Printing tool force/torque sensor:")
            print(self.ft.wrench())
        if key is glfw.KEY_P:
            print(
                "Poking finger: "
                f"q={self.poker_position():.4f}, "
                f"target={self.poker_target():.4f}"
            )

    @ms.thread
    def see_me_run(self, ss: ms.SimSync):
        while True:
            ss.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Do not attach the Robotiq gripper; measure the load-cell-only baseline.",
    )
    parser.add_argument(
        "--no-poker",
        action="store_true",
        help="Do not add the actuated sliding box used to poke the gripper.",
    )
    args = parser.parse_args()

    sim = Sim(with_gripper=not args.no_gripper, with_poker=not args.no_poker)
    mode = "with gripper" if sim.with_gripper else "without gripper"
    print(f"Running UR FT sensor demo {mode}.")
    if sim.with_poker:
        print("Poking finger enabled. Press P to print slider state.")
    print(f"Expected sensor Fz from mounted mass ~= {sim.expected_sensor_weight:.6f} N")
    sim.run()
