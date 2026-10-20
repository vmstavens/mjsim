import argparse
import sys
from pathlib import Path

import glfw
import mujoco as mj
import numpy as np
from robot_descriptions import ur10e_mj_description

import mjsim as ms
from mjsim.utils.math import rotvec_to_quat

EXAMPLES_DIR = Path(__file__).resolve().parent
removed_paths = [
    path for path in sys.path if Path(path or ".").resolve() == EXAMPLES_DIR
]
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != EXAMPLES_DIR]
import viser  # noqa: E402
from mjviser import Viewer  # noqa: E402

sys.path[:0] = removed_paths


CONTROLLER_EQUATIONS = """
This GUI is intentionally written as a compact example of 6D admittance plus
inverse-dynamics torque control.

```text
Measured wrench:
F = clip_deadband(force_sign  * R_loadcell f_sensor)
T = clip_deadband(torque_sign * R_flange   t_sensor)

Admittance target:
M x_ddot + D x_dot + K x = F
I r_ddot + Dr r_dot + Kr r = T
p_target = p_home + x
q_target = quat(r) * q_home

Cartesian tracking:
e = [p_target - p_actual, rotvec(q_target * inverse(q_actual))]
xdot = J(q) qdot
xddot_des = Kp e - Kd xdot
Kd = 2 damping_ratio sqrt(Kp)

Damped least-squares acceleration:
qddot_des = J^T (J J^T + lambda^2 I)^-1 xddot_des

Inverse dynamics motor torque:
tau_raw = inverse_dynamics(q, qdot, qddot_des)
tau_cmd = clip(tau_raw, motor torque limits)
```
"""


def use_motor_actuators(robot_spec: mj.MjSpec) -> None:
    """Make UR actuator control inputs equal commanded motor torques."""
    for actuator in robot_spec.actuators:
        torque_range = list(actuator.forcerange)
        actuator.set_to_motor()
        actuator.ctrllimited = True
        actuator.ctrlrange = torque_range
        actuator.forcelimited = True
        actuator.forcerange = torque_range


class Sim(ms.BaseSim):
    def __init__(self):
        self._model, self._data = self._init()
        self.ur = ms.Robot(self.model, self.data, "ur/")

        self.force_enabled = False
        self.actual_site_id = self.model.site("ur/actual_frame").id
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
        self.tracking_pos_gain = 140.0
        self.tracking_ori_gain = 80.0
        self.tracking_damping_ratio = 1.0
        self.tracking_dls_damping = 0.04
        self.torque_limit_scale = 1.0
        self.force_sign = -1.0
        self.torque_sign = -1.0

        self.admittance_offset = np.zeros(3)
        self.admittance_velocity = np.zeros(3)
        self.admittance_rotvec = np.zeros(3)
        self.admittance_angular_velocity = np.zeros(3)
        self.home_target_pos = self.data.site_xpos[self.actual_site_id].copy()
        self.home_target_quat = self.site_quat(self.actual_site_id)
        self.target_quat = self.home_target_quat.copy()
        self.torque = np.zeros(self.ur.info.n_actuators)
        self.raw_torque = np.zeros(self.ur.info.n_actuators)
        self.torque_limits = self.ur.info.actuator_limits
        self.update_target_frame()
        self.step_count = 0
        self.last_toggle_time = -1.0

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:
        scene = ms.empty_scene()
        ur = mj.MjSpec.from_file(ur10e_mj_description.MJCF_PATH)
        use_motor_actuators(ur)

        wrist = ur.site("attachment_site").parent
        for geom in wrist.geoms:
            if geom.type == mj.mjtGeom.mjGEOM_CYLINDER:
                geom.contype = 0
                geom.conaffinity = 0

        loadcell = wrist.add_body(name="load_cell", mass=1e-6)
        loadcell.add_site(name="loadcell", size=0.01)
        flange = loadcell.add_body(name="flange")
        flange.add_geom(
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
        flange.add_site(
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

        scene.attach(ur, "ur/", frame=scene.worldbody.add_frame())
        self._add_target_frame(scene)
        self._add_pusher(scene)

        model = scene.compile()
        data = mj.MjData(model)
        mj.mj_resetDataKeyframe(model, data, ur.key("home").id)
        data.ctrl[:] = 0.0
        mj.mj_forward(model, data)
        return model, data

    def _add_target_frame(self, scene: mj.MjSpec) -> None:
        target = scene.worldbody.add_body(name="target_frame", mocap=True)
        target.add_site(
            name="target_frame",
            size=[0.022],
            rgba=[1.0, 0.0, 1.0, 1.0],
        )

    def _add_pusher(self, scene: mj.MjSpec) -> None:
        pusher = scene.worldbody.add_body(
            name="pusher",
            mocap=True,
            pos=[-0.174, 0.691, 0.58],
        )
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
    def quat_error_rotvec(
        target_quat: np.ndarray,
        actual_quat: np.ndarray,
    ) -> np.ndarray:
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

        # get timestep
        dt = self.model.opt.timestep
        # measure force and torque from sensor
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

    def desired_joint_acceleration(self) -> np.ndarray:
        position_error = (
            self.home_target_pos
            + self.admittance_offset
            - self.data.site_xpos[self.actual_site_id]
        )
        orientation_error = self.quat_error_rotvec(
            self.target_quat,
            self.site_quat(self.actual_site_id),
        )
        pose_error = np.concatenate((position_error, orientation_error))

        jac = self.ur.J(base_frame=0, site_frame=self.actual_site_id)
        site_velocity = jac @ self.ur.dq
        task_kp = np.array([self.tracking_pos_gain] * 3 + [self.tracking_ori_gain] * 3)
        task_kd = self.tracking_damping_ratio * 2.0 * np.sqrt(task_kp)
        task_acceleration = task_kp * pose_error - task_kd * site_velocity

        lhs = jac @ jac.T + self.tracking_dls_damping**2 * np.eye(6)
        return jac.T @ np.linalg.solve(lhs, task_acceleration)

    def inverse_dynamics_torque(self, qacc: np.ndarray) -> np.ndarray:
        saved_qacc = self.data.qacc.copy()
        self.data.qacc[:] = 0.0
        self.data.qacc[self.ur.robot_dof_indices] = qacc
        mj.mj_inverse(self.model, self.data)
        torque = self.data.qfrc_inverse[self.ur.robot_dof_indices].copy()
        self.data.qacc[:] = saved_qacc
        return torque

    def update_robot_control(self) -> None:
        qacc = self.desired_joint_acceleration()
        self.raw_torque = self.inverse_dynamics_torque(qacc)
        torque_min, torque_max = self.torque_limit_scale * self.torque_limits
        self.torque = np.clip(self.raw_torque, torque_min, torque_max)
        self.ur.set_ctrl(self.torque)

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
        force = self.force_world_frame()
        torque = self.torque_world_frame()
        print(
            "wrench sensor: "
            f"enabled={self.force_enabled} | "
            f"|F_world|={np.linalg.norm(force): .2f} N | "
            f"|T_world|={np.linalg.norm(torque): .2f} N m | "
            f"target_offset={self.admittance_offset} | "
            f"target_rotvec={self.admittance_rotvec} | "
            f"|tau_cmd|_inf={np.abs(self.torque).max(): .2f} N m | "
            f"pusher_z={self.pusher_position(): .4f}",
            flush=True,
        )

    def keyboard_callback(self, key):
        if key == glfw.KEY_SPACE:
            if self.data.time - self.last_toggle_time < 0.25:
                return
            self.last_toggle_time = self.data.time
            self.force_enabled = not self.force_enabled
            self.update_pusher_control()
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


def reset_sim(sim: Sim) -> None:
    mj.mj_resetDataKeyframe(sim.model, sim.data, sim.model.keyframe("ur/home").id)
    sim.data.ctrl[:] = 0.0
    mj.mj_forward(sim.model, sim.data)

    sim.force_enabled = False
    sim.admittance_offset[:] = 0.0
    sim.admittance_velocity[:] = 0.0
    sim.admittance_rotvec[:] = 0.0
    sim.admittance_angular_velocity[:] = 0.0
    sim.home_target_pos = sim.data.site_xpos[sim.actual_site_id].copy()
    sim.home_target_quat = sim.site_quat(sim.actual_site_id)
    sim.target_quat = sim.home_target_quat.copy()
    sim.raw_torque[:] = 0.0
    sim.torque[:] = 0.0
    sim.ur.set_ctrl(sim.torque)
    sim.update_target_frame()
    sim.update_pusher_control()


def add_float_control(
    server: viser.ViserServer,
    sim: Sim,
    label: str,
    attr: str,
    minimum: float,
    maximum: float,
    step: float,
):
    handle = server.gui.add_slider(
        label,
        min=minimum,
        max=maximum,
        step=step,
        initial_value=float(getattr(sim, attr)),
    )

    @handle.on_update
    def _(_event) -> None:
        setattr(sim, attr, float(handle.value))

    return handle


def add_bool_control(server: viser.ViserServer, sim: Sim, label: str, attr: str):
    handle = server.gui.add_checkbox(label, initial_value=bool(getattr(sim, attr)))

    @handle.on_update
    def _(_event) -> None:
        setattr(sim, attr, bool(handle.value))
        if attr == "force_enabled":
            sim.update_pusher_control()

    return handle


def add_gui(server: viser.ViserServer, sim: Sim):
    server.gui.set_panel_label("Admittance Controller")
    server.gui.add_markdown(
        "Tune the 6D admittance controller live. "
        "Green is the actual flange frame; magenta is the compliant target."
    )
    with server.gui.add_folder("Controller Equations", expand_by_default=True):
        server.gui.add_markdown(CONTROLLER_EQUATIONS)

    with server.gui.add_folder("Pusher", expand_by_default=True):
        pusher_enabled = add_bool_control(
            server,
            sim,
            "Enable physical pusher",
            "force_enabled",
        )
        add_float_control(
            server,
            sim,
            "Pusher penetration offset [m]",
            "pusher_penetration_offset",
            0.035,
            0.075,
            0.001,
        )

    with server.gui.add_folder("Translational Admittance", expand_by_default=True):
        add_float_control(server, sim, "Mass [kg]", "admittance_mass", 0.5, 30.0, 0.5)
        add_float_control(
            server,
            sim,
            "Damping [N s/m]",
            "admittance_damping",
            10.0,
            600.0,
            5.0,
        )
        add_float_control(
            server,
            sim,
            "Stiffness [N/m]",
            "admittance_stiffness",
            100.0,
            10000.0,
            50.0,
        )
        add_float_control(
            server,
            sim,
            "Max target offset [m]",
            "max_target_offset",
            0.005,
            0.25,
            0.005,
        )

    with server.gui.add_folder("Rotational Admittance", expand_by_default=True):
        add_float_control(
            server,
            sim,
            "Inertia [kg m^2]",
            "admittance_rot_inertia",
            0.02,
            3.0,
            0.02,
        )
        add_float_control(
            server,
            sim,
            "Rot damping [N m s/rad]",
            "admittance_rot_damping",
            0.2,
            40.0,
            0.2,
        )
        add_float_control(
            server,
            sim,
            "Rot stiffness [N m/rad]",
            "admittance_rot_stiffness",
            1.0,
            200.0,
            1.0,
        )
        add_float_control(
            server,
            sim,
            "Max target rotvec [rad]",
            "max_target_rotvec",
            0.02,
            1.0,
            0.01,
        )

    with server.gui.add_folder("Wrench Filtering", expand_by_default=False):
        add_float_control(
            server,
            sim,
            "Force deadband [N]",
            "force_deadband",
            0.0,
            50.0,
            0.5,
        )
        add_float_control(
            server,
            sim,
            "Force limit [N]",
            "force_limit",
            5.0,
            1000.0,
            5.0,
        )
        add_float_control(
            server,
            sim,
            "Torque deadband [N m]",
            "torque_deadband",
            0.0,
            2.0,
            0.01,
        )
        add_float_control(
            server,
            sim,
            "Torque limit [N m]",
            "torque_limit",
            0.1,
            50.0,
            0.1,
        )

    with server.gui.add_folder("Torque Tracking", expand_by_default=False):
        add_float_control(
            server,
            sim,
            "Position gain [1/s^2]",
            "tracking_pos_gain",
            0.0,
            500.0,
            5.0,
        )
        add_float_control(
            server,
            sim,
            "Orientation gain [1/s^2]",
            "tracking_ori_gain",
            0.0,
            300.0,
            5.0,
        )
        add_float_control(
            server,
            sim,
            "Damping ratio",
            "tracking_damping_ratio",
            0.0,
            3.0,
            0.05,
        )
        add_float_control(
            server,
            sim,
            "DLS damping",
            "tracking_dls_damping",
            0.001,
            0.2,
            0.001,
        )
        add_float_control(
            server,
            sim,
            "Torque limit scale",
            "torque_limit_scale",
            0.05,
            1.0,
            0.05,
        )
        add_float_control(server, sim, "Force sign", "force_sign", -1.0, 1.0, 2.0)
        add_float_control(server, sim, "Torque sign", "torque_sign", -1.0, 1.0, 2.0)

    readout = server.gui.add_text("State", "", multiline=True, disabled=True)
    reset_button = server.gui.add_button("Reset simulation", color="gray")

    @reset_button.on_click
    def _(_event) -> None:
        reset_sim(sim)
        pusher_enabled.value = sim.force_enabled

    return readout


def format_readout(sim: Sim) -> str:
    force = sim.force_world_frame()
    torque = sim.torque_world_frame()
    motor_torque = sim.torque
    raw_torque = sim.raw_torque
    return (
        f"force_enabled: {sim.force_enabled}\n"
        f"|F_world|: {(force @ force) ** 0.5: .3f} N\n"
        f"|T_world|: {(torque @ torque) ** 0.5: .3f} N m\n"
        f"target_offset: {sim.admittance_offset}\n"
        f"target_rotvec: {sim.admittance_rotvec}\n"
        f"raw motor torque: {raw_torque}\n"
        f"limited motor torque: {motor_torque}\n"
        f"pusher_z: {sim.pusher_position(): .4f} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    sim = Sim()
    server = viser.ViserServer(port=args.port)
    readout = add_gui(server, sim)
    step_count = 0

    def step(_model: mj.MjModel, _data: mj.MjData) -> None:
        nonlocal step_count
        sim.update_pusher_control()
        sim.step_admittance_controller()
        mj.mj_step(sim.model, sim.data)

        step_count += 1
        if step_count % 25 == 0:
            readout.value = format_readout(sim)

    def reset(_model: mj.MjModel, _data: mj.MjData) -> None:
        reset_sim(sim)

    print(f"Viser admittance controller GUI: http://localhost:{args.port}")
    viewer = Viewer(sim.model, sim.data, step_fn=step, reset_fn=reset, server=server)
    viewer.run()


if __name__ == "__main__":
    main()
