"""
Cartesian/task-space planning demo.

Places a UR10e arm in a home configuration, generates a Cartesian end-effector
path around an obstacle with ``mjsim.utils.ompl.xplan``, solves position IK for the waypoints, and plays the
motion back kinematically. No dynamics step is used; the viewer is run with
``dyn=False`` and the playback thread refreshes poses with ``mj_forward``.
"""

import time

import cv2
import glfw
import mjsim as ms
import mujoco as mj
import numpy as np
from robot_descriptions import ur10e_mj_description

from mjsim.sensors.camera import Camera
from mjsim.utils.ompl import xplan


HOME_Q = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
OBSTACLE_RADIUS = 0.11
N_PLAYBACK_SAMPLES = 220


class Sim(ms.BaseSim):
    def __init__(self) -> None:
        super().__init__()
        self._model, self._data = self._init()
        self.ur = ms.Robot(self.model, self.data, "ur/")
        self.site_name = self.ur.info.site_names[0]
        self.site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self.site_name)
        self.joint_qpos_ids = np.array(self.ur.info.joint_indxs)
        self.joint_dof_ids = np.array(self.ur.info.dof_indxs)
        self.joint_limits = np.array(
            [self.model.joint(name).range for name in self.ur.info.joint_names]
        )

        self.obstacle_pos = np.zeros(3)
        self.cartesian_path = self._plan_cartesian_path()
        self.q_waypoints = self._solve_path_ik(self.cartesian_path)
        self.q_path = self._densify_joint_path(self.q_waypoints, N_PLAYBACK_SAMPLES)
        self.path_index = 0
        self.playing = True

        self._add_path_visuals()
        mj.mj_forward(self.model, self.data)

        self.camera = Camera(
            self.model,
            self.data,
            cam_name="front",
            width=640,
            height=480,
            save_dir="tmp/cartesian_planning/",
        )

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:
        scene = ms.empty_scene(
            sim_name="cartesian_planning_demo",
            statistic_center=(-0.45, 0.0, 0.45),
            statistic_extent=1.2,
        )

        ur = mj.MjSpec.from_file(ur10e_mj_description.MJCF_PATH)
        scene.attach(ur, "ur/", frame=scene.worldbody.add_frame())

        scene.worldbody.add_camera(
            name="front",
            pos=[0.25, -1.35, 0.95],
            euler=[0.95, 0.0, 0.25],
            fovy=45.0,
        )

        model = scene.compile()
        data = mj.MjData(model)
        data.qpos[: len(HOME_Q)] = HOME_Q
        mj.mj_forward(model, data)
        return model, data

    def _home_site_position(self) -> np.ndarray:
        self.data.qpos[self.joint_qpos_ids] = HOME_Q
        self.data.qvel[self.joint_dof_ids] = 0.0
        mj.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy()

    def _plan_cartesian_path(self) -> np.ndarray:
        start = self._home_site_position()
        goal = start + np.array([0.0, 0.45, -0.05])
        self.obstacle_pos = 0.5 * (start + goal)

        start_pose = np.r_[start, [0.0, 0.0, 0.0, 1.0]]
        goal_pose = np.r_[goal, [0.0, 0.0, 0.0, 1.0]]

        ok, path = xplan(
            robot=self.ur,
            start_pose=start_pose,
            goal_pose=goal_pose,
            validity_check_fn=self._xplan_validity_check,
            max_step_size=0.08,
            simplify=True,
            timeout=2.0,
        )
        if not ok or path is None:
            msg = "xplan failed to find a Cartesian path around the obstacle."
            raise RuntimeError(msg)
        return path

    def _xplan_validity_check(self, state) -> bool:
        point = np.array([state.getX(), state.getY(), state.getZ()])
        clears_obstacle = np.linalg.norm(point - self.obstacle_pos) > (
            OBSTACLE_RADIUS + 0.03
        )
        above_floor = point[2] > 0.2
        return bool(clears_obstacle and above_floor)

    def _solve_path_ik(self, path: np.ndarray) -> np.ndarray:
        q = HOME_Q.copy()
        q_path = []
        for target_pose in path:
            q = self._solve_position_ik(target_pose[:3], q)
            q_path.append(q.copy())
        return np.array(q_path)

    def _solve_position_ik(self, target: np.ndarray, q0: np.ndarray) -> np.ndarray:
        q = q0.copy()
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]

        for _ in range(120):
            self.data.qpos[self.joint_qpos_ids] = q
            self.data.qvel[self.joint_dof_ids] = 0.0
            mj.mj_forward(self.model, self.data)

            error = target - self.data.site_xpos[self.site_id]
            if np.linalg.norm(error) < 2e-3:
                return q

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mj.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jac = jacp[:, self.joint_dof_ids]

            damping = 1e-2
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(3), error)
            dq_norm = np.linalg.norm(dq)
            if dq_norm > 0.08:
                dq *= 0.08 / dq_norm
            q = np.clip(q + dq, lo, hi)

        return q

    @staticmethod
    def _densify_joint_path(q_waypoints: np.ndarray, n_samples: int) -> np.ndarray:
        source = np.linspace(0.0, 1.0, len(q_waypoints))
        target = np.linspace(0.0, 1.0, n_samples)
        return np.column_stack(
            [np.interp(target, source, q_waypoints[:, i]) for i in range(q_waypoints.shape[1])]
        )

    def _add_path_visuals(self) -> None:
        # Visual-only geoms can be added through the model after compile by using user
        # scene geoms, but keeping them in the compiled model makes camera captures
        # include the planned path. Rebuild once with the solved path markers attached.
        scene = ms.empty_scene(
            sim_name="cartesian_planning_demo",
            statistic_center=(-0.45, 0.0, 0.45),
            statistic_extent=1.2,
        )
        ur = mj.MjSpec.from_file(ur10e_mj_description.MJCF_PATH)
        scene.attach(ur, "ur/", frame=scene.worldbody.add_frame())
        scene.worldbody.add_geom(
            name="obstacle",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[OBSTACLE_RADIUS],
            pos=self.obstacle_pos.tolist(),
            rgba=[0.9, 0.1, 0.1, 0.35],
            contype=0,
            conaffinity=0,
        )
        scene.worldbody.add_geom(
            name="start",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.035],
            pos=self.cartesian_path[0, :3].tolist(),
            rgba=[0.1, 0.7, 0.2, 1.0],
            contype=0,
            conaffinity=0,
        )
        scene.worldbody.add_geom(
            name="goal",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.035],
            pos=self.cartesian_path[-1, :3].tolist(),
            rgba=[0.1, 0.3, 0.9, 1.0],
            contype=0,
            conaffinity=0,
        )
        for i, (a, b) in enumerate(zip(self.cartesian_path[:-1], self.cartesian_path[1:])):
            scene.worldbody.add_geom(
                name=f"path_segment_{i}",
                type=mj.mjtGeom.mjGEOM_CAPSULE,
                fromto=[*a[:3], *b[:3]],
                size=[0.01],
                rgba=[1.0, 0.75, 0.1, 1.0],
                contype=0,
                conaffinity=0,
            )
        scene.worldbody.add_camera(
            name="front",
            pos=[0.25, -1.35, 0.95],
            euler=[0.95, 0.0, 0.25],
            fovy=45.0,
        )

        self._model = scene.compile()
        self._data = mj.MjData(self._model)
        self._data.qpos[: len(HOME_Q)] = HOME_Q
        mj.mj_forward(self._model, self._data)
        self.ur = ms.Robot(self.model, self.data, "ur/")
        self.site_name = self.ur.info.site_names[0]
        self.site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self.site_name)
        self.joint_qpos_ids = np.array(self.ur.info.joint_indxs)
        self.joint_dof_ids = np.array(self.ur.info.dof_indxs)

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def data(self) -> mj.MjData:
        return self._data

    def keyboard_callback(self, key: int) -> None:
        if key == glfw.KEY_C:
            rgb = self.camera.image
            cv2.imshow("cartesian planning camera", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            print(f"Captured image from camera 'front': {rgb.shape} {rgb.dtype}")

        if key == glfw.KEY_P:
            self.playing = not self.playing
            print("Playback:", "running" if self.playing else "paused")

        if key == glfw.KEY_R:
            self.path_index = 0
            self._set_robot_q(self.q_path[0])
            print("Reset kinematic playback.")

        if key == glfw.KEY_Q:
            cv2.destroyAllWindows()

    def _set_robot_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.joint_qpos_ids] = q
        self.data.qvel[self.joint_dof_ids] = 0.0
        mj.mj_forward(self.model, self.data)

    @ms.thread
    def play_kinematic_path(self, ss: ms.SimSync) -> None:
        while True:
            if self.playing:
                self._set_robot_q(self.q_path[self.path_index])
                self.path_index = (self.path_index + 1) % len(self.q_path)
            ss.step()
            time.sleep(0.01)


if __name__ == "__main__":
    sim = Sim()
    print(f"Generated {len(sim.q_waypoints)} Cartesian IK waypoints.")
    print("Press C in the MuJoCo viewer to capture the camera image.")
    print("Press P to pause/resume kinematic playback.")
    print("Press R to reset the path.")
    print("Press Q to close OpenCV image windows.")
    sim.run(dyn=False)
