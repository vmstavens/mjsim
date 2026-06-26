import glfw
import mjsim as ms
import mujoco as mj
import numpy as np
from robot_descriptions import ur10e_mj_description

from mjsim.utils.math import rotvec_to_quat


class Sim(ms.BaseSim):
    def __init__(self):
        self._model, self._data = self._init()

        self.ur = ms.Robot(self.model, self.data, "ur/")

        self.force_enabled = False
        self.load_cell_body_id = self.model.body("ur/load_cell").id
        self.flange_body_id = self.model.body("ur/flange").id
        self.flange_geom_id = self.model.geom("ur/flange_geom").id
        self.actual_site_id = self.model.site("ur/actual_frame").id
        self.target_site_id = self.model.site("target_frame").id
        self.target_body_id = self.model.body("target_frame").id
        self.target_mocap_id = self.model.body_mocapid[self.target_body_id]
        self.load_cell_site_id = self.model.site("ur/loadcell").id
        self.force_sensor_adr = self.model.sensor("ur/force").adr[0]
        self.torque_sensor_adr = self.model.sensor("ur/torque").adr[0]
        self.pusher_body_id = self.model.body("pusher").id
        self.pusher_mocap_id = self.model.body_mocapid[self.pusher_body_id]
        self.pusher_rest_pos = [-0.174, 0.691, 0.58]
        self.pusher_penetration_offset = 0.055
        self.admittance_mass = 8.0
        self.admittance_damping = 180.0
        self.admittance_stiffness = 2500.0
        self.admittance_rot_inertia = 0.4
        self.admittance_rot_damping = 8.0
        self.admittance_rot_stiffness = 35.0
        self.force_deadband = 3.0
        self.force_limit = 250.0
        self.torque_deadband = 0.05
        self.torque_limit = 8.0
        self.max_target_offset = 0.08
        self.max_target_rotvec = 0.35
        self.ik_pos_gain = 0.08
        self.ik_ori_gain = 0.04
        self.ik_damping = 0.04
        self.force_sign = -1.0
        self.torque_sign = -1.0
        self.admittance_offset = np.zeros(3)
        self.admittance_velocity = np.zeros(3)
        self.admittance_rotvec = np.zeros(3)
        self.admittance_angular_velocity = np.zeros(3)
        self.home_target_pos = self.data.site_xpos[self.actual_site_id].copy()
        self.home_target_quat = self.site_quat(self.actual_site_id)
        self.target_quat = self.home_target_quat.copy()
        self.q_target = self.ur.q.copy()
        self.update_target_frame()
        self.step_count = 0
        self.last_toggle_time = -1.0

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:

        scene = ms.empty_scene()

        ur = mj.MjSpec.from_file(ur10e_mj_description.MJCF_PATH)


        b_site_parent_body = ur.site("attachment_site").parent

        for geom in b_site_parent_body.geoms:
            if geom.type == mj.mjtGeom.mjGEOM_CYLINDER:
                geom.contype = 0
                geom.conaffinity = 0

        b_loadcell = b_site_parent_body.add_body(name="load_cell", mass=1e-6)
        s_loadcell = b_loadcell.add_site(name="loadcell", size=0.01)
        b_flange = b_loadcell.add_body(name="flange")
        b_flange.add_geom(
            name="flange_geom",
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            pos=[0, 0.097, 0],
            quat=[1, 1, 0, 0],
            size=[0.046, 0.02],
            mass=1e-4,
            contype=1,
            conaffinity=1,
            rgba=[0.5, 0.5, 0.5, 0.0],
        )
        b_flange.add_site(
            name="actual_frame",
            pos=[0, 0.097, 0],
            size=[0.018],
            rgba=[0.0, 1.0, 0.0, 1.0],
        )

        ur.add_sensor(
            name="force",
            type=mj.mjtSensor.mjSENS_FORCE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="loadcell",
            dim=3,
        )
        ur.add_sensor(
            name="torque",
            type=mj.mjtSensor.mjSENS_TORQUE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="actual_frame",
            dim=3,
        )


        key = ur.key("home")

        f_ur = scene.worldbody.add_frame()

        scene.attach(ur, "ur/", frame=f_ur)
        self._add_target_frame(scene)
        self._add_pusher(scene)

        m = scene.compile()

        d = mj.MjData(m)

        mj.mj_resetDataKeyframe(m, d, key.id)
        mj.mj_forward(m, d)

        return m, d

    def _add_target_frame(self, scene: mj.MjSpec) -> None:
        target = scene.worldbody.add_body(name="target_frame", mocap=True)
        target.add_site(
            name="target_frame",
            size=[0.022],
            rgba=[1.0, 0.0, 1.0, 1.0],
        )

    def _add_pusher(self, scene: mj.MjSpec) -> None:
        pusher = scene.worldbody.add_body(name="pusher", mocap=True, pos=[-0.174, 0.691, 0.58])
        pusher.add_geom(
            name="pusher_tip",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.04],
            rgba=[0.1, 0.7, 0.9, 1.0],
            friction=[1.0, 0.005, 0.0001],
            solref=[0.01, 1],
            solimp=[0.95, 0.99, 0.001, 0.5, 2],
        )

    @property
    def model(self):
        return self._model
    @property
    def data(self):
        return self._data

    def pusher_target(self):
        if not self.force_enabled:
            return self.pusher_rest_pos

        target = self.home_target_pos.copy()
        target[2] -= self.pusher_penetration_offset
        return target

    def pusher_position(self) -> float:
        return float(self.data.mocap_pos[self.pusher_mocap_id, 2])

    def update_pusher_control(self) -> None:
        self.data.mocap_pos[self.pusher_mocap_id] = self.pusher_target()

    def site_quat(self, site_id: int) -> np.ndarray:
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, self.data.site_xmat[site_id])
        return quat

    @staticmethod
    def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        out = np.zeros(4)
        mj.mju_mulQuat(out, q1, q2)
        return out

    @staticmethod
    def quat_error_rotvec(target_quat: np.ndarray, actual_quat: np.ndarray) -> np.ndarray:
        actual_conj = np.zeros(4)
        error_quat = np.zeros(4)
        error_rotvec = np.zeros(3)
        mj.mju_negQuat(actual_conj, actual_quat)
        mj.mju_mulQuat(error_quat, target_quat, actual_conj)
        if error_quat[0] < 0.0:
            error_quat *= -1.0
        mj.mju_quat2Vel(error_rotvec, error_quat, 1.0)
        return error_rotvec

    def update_target_frame(self) -> None:
        delta_quat = rotvec_to_quat(self.admittance_rotvec)
        self.target_quat = self.quat_mul(delta_quat, self.home_target_quat)
        self.data.mocap_pos[self.target_mocap_id] = (
            self.home_target_pos + self.admittance_offset
        )
        self.data.mocap_quat[self.target_mocap_id] = self.target_quat

    def admittance_force(self) -> np.ndarray:
        force = self.force_sign * self.force_world_frame()
        force_norm = np.linalg.norm(force)
        if force_norm < self.force_deadband:
            return np.zeros(3)
        if force_norm > self.force_limit:
            force = force * (self.force_limit / force_norm)
        return force

    def admittance_torque(self) -> np.ndarray:
        torque = self.torque_sign * self.torque_world_frame()
        torque_norm = np.linalg.norm(torque)
        if torque_norm < self.torque_deadband:
            return np.zeros(3)
        if torque_norm > self.torque_limit:
            torque = torque * (self.torque_limit / torque_norm)
        return torque

    def update_admittance_target(self) -> None:
        dt = self.model.opt.timestep
        force = self.admittance_force()
        torque = self.admittance_torque()
        acceleration = (
            force
            - self.admittance_damping * self.admittance_velocity
            - self.admittance_stiffness * self.admittance_offset
        ) / self.admittance_mass
        angular_acceleration = (
            torque
            - self.admittance_rot_damping * self.admittance_angular_velocity
            - self.admittance_rot_stiffness * self.admittance_rotvec
        ) / self.admittance_rot_inertia

        self.admittance_velocity += acceleration * dt
        self.admittance_offset += self.admittance_velocity * dt
        self.admittance_angular_velocity += angular_acceleration * dt
        self.admittance_rotvec += self.admittance_angular_velocity * dt

        offset_norm = np.linalg.norm(self.admittance_offset)
        if offset_norm > self.max_target_offset:
            self.admittance_offset *= self.max_target_offset / offset_norm
            self.admittance_velocity[:] = 0.0

        rotvec_norm = np.linalg.norm(self.admittance_rotvec)
        if rotvec_norm > self.max_target_rotvec:
            self.admittance_rotvec *= self.max_target_rotvec / rotvec_norm
            self.admittance_angular_velocity[:] = 0.0

        self.update_target_frame()

    def update_robot_control(self) -> None:
        target_pos = self.home_target_pos + self.admittance_offset
        actual_pos = self.data.site_xpos[self.actual_site_id]
        position_error = target_pos - actual_pos
        orientation_error = self.quat_error_rotvec(
            self.target_quat,
            self.site_quat(self.actual_site_id),
        )

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, jacp, jacr, self.actual_site_id)
        jac = np.vstack((jacp, jacr))[:, self.ur.robot_dof_indices]
        task_error = np.concatenate(
            (
                self.ik_pos_gain * position_error,
                self.ik_ori_gain * orientation_error,
            )
        )
        lhs = jac @ jac.T + self.ik_damping**2 * np.eye(6)
        dq = jac.T @ np.linalg.solve(lhs, task_error)

        self.q_target += dq
        lower, upper = self.ur.info.joint_limits
        self.q_target = np.clip(self.q_target, lower, upper)
        self.ur.set_ctrl(self.q_target)

    def step_admittance_controller(self) -> None:
        self.update_admittance_target()
        self.update_robot_control()

    def force_site_frame(self):
        return self.data.sensordata[self.force_sensor_adr : self.force_sensor_adr + 3]

    def torque_site_frame(self):
        return self.data.sensordata[self.torque_sensor_adr : self.torque_sensor_adr + 3]

    def force_world_frame(self):
        site_rot = self.data.site_xmat[self.load_cell_site_id].reshape(3, 3)
        return site_rot @ self.force_site_frame()

    def torque_world_frame(self):
        site_rot = self.data.site_xmat[self.actual_site_id].reshape(3, 3)
        return site_rot @ self.torque_site_frame()

    def print_force_reading(self) -> None:
        f_site = self.force_site_frame()
        f_world = self.force_world_frame()
        t_site = self.torque_site_frame()
        t_world = self.torque_world_frame()
        f_world_norm = (
            f_world[0] * f_world[0]
            + f_world[1] * f_world[1]
            + f_world[2] * f_world[2]
        ) ** 0.5
        t_world_norm = (
            t_world[0] * t_world[0]
            + t_world[1] * t_world[1]
            + t_world[2] * t_world[2]
        ) ** 0.5
        print(
            "wrench sensor: "
            f"enabled={self.force_enabled} | "
            f"F_site=({f_site[0]: .2f}, {f_site[1]: .2f}, {f_site[2]: .2f}) | "
            f"F_world=({f_world[0]: .2f}, {f_world[1]: .2f}, {f_world[2]: .2f}) | "
            f"|F|={f_world_norm: .2f} | "
            f"T_site=({t_site[0]: .2f}, {t_site[1]: .2f}, {t_site[2]: .2f}) | "
            f"T_world=({t_world[0]: .2f}, {t_world[1]: .2f}, {t_world[2]: .2f}) | "
            f"|T|={t_world_norm: .2f} | "
            f"target_offset=({self.admittance_offset[0]: .3f}, "
            f"{self.admittance_offset[1]: .3f}, {self.admittance_offset[2]: .3f}) | "
            f"target_rotvec=({self.admittance_rotvec[0]: .3f}, "
            f"{self.admittance_rotvec[1]: .3f}, {self.admittance_rotvec[2]: .3f}) | "
            f"pusher_z=({self.pusher_position(): .4f})",
            flush=True,
        )


    def keyboard_callback(self, key):
        if key == glfw.KEY_SPACE:
            if self.data.time - self.last_toggle_time < 0.25:
                return
            self.last_toggle_time = self.data.time
            self.force_enabled = not self.force_enabled
            self.update_pusher_control()
            print(
                f"Physical pusher "
                f"{'enabled' if self.force_enabled else 'disabled'}",
                flush=True,
            )
        elif key == glfw.KEY_F:
            self.print_force_reading()

    @ms.thread
    def see_me_run(self, ss: ms.SimSync):
        while True:
            self.update_pusher_control()
            self.step_admittance_controller()
            ss.step()
            self.step_count += 1
            if self.step_count % 100 == 0:
                self.print_force_reading()

if __name__ == "__main__":
    sim = Sim()
    print(
        "UR admittance demo. Press space to push the flange; green is actual, "
        "magenta is compliant target."
    )

    sim.run()
