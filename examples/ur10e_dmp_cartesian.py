"""Train a Cartesian DMP from a CSV demonstration and play it on a UR10e.

The CSV is expected to contain UR-style TCP pose columns:
``actual_TCP_pose_0..5`` or ``target_TCP_pose_0..5``. The first three columns
are position in meters and the last three are a rotation vector in radians.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mjsim-matplotlib")

import glfw
import mujoco as mj
import numpy as np
import spatialmath as sm
import spatialmath.base as smb
from robot_descriptions import ur10e_mj_description
from scipy.spatial.transform import Rotation

import mjsim as ms
from mjsim.base.robot import IKResult
from mjsim.ctrl import DMPCartesian

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = EXAMPLES_DIR / "assets" / "demonstration.csv"
DEFAULT_PLOT = Path("tmp") / "ur10e_dmp_tracking.png"
DEFAULT_HOME_Q = np.array(
    [-0.33877165, -1.27340336, -2.21872163, -1.2085238, 1.6245724, -0.03443128]
)


@dataclass(frozen=True)
class Demonstration:
    ts: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    q0: np.ndarray
    tcp_to_attachment: np.ndarray


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float).copy()
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Quaternion trajectory contains a zero quaternion.")
    quaternions /= norms

    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i], quaternions[i - 1]) < 0.0:
            quaternions[i] *= -1.0
    return quaternions


def _load_csv(
    path: Path,
    pose_prefix: str,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = data.reshape(1)

    required = ["timestamp", *[f"{pose_prefix}_{i}" for i in range(6)]]
    missing = [name for name in required if name not in data.dtype.names]
    if missing:
        msg = f"{path} is missing required columns: {missing}"
        raise ValueError(msg)

    ts = np.asarray(data["timestamp"], dtype=float)
    ts = ts - ts[0]
    poses = np.column_stack([data[f"{pose_prefix}_{i}"] for i in range(6)])

    finite = np.isfinite(ts) & np.all(np.isfinite(poses), axis=1)
    ts = ts[finite]
    poses = poses[finite]
    if len(ts) < 2:
        msg = f"{path} must contain at least two finite pose samples"
        raise ValueError(msg)

    if max_samples > 0 and len(ts) > max_samples:
        indices = np.linspace(0, len(ts) - 1, max_samples).astype(int)
        ts = ts[indices]
        poses = poses[indices]

    return ts, poses


def _rotvec_poses_to_demo(
    path: Path,
    pose_prefix: str,
    max_samples: int,
    model: mj.MjModel,
    data: mj.MjData,
    robot: ms.Robot,
) -> Demonstration:
    ts, poses = _load_csv(path, pose_prefix, max_samples)
    csv = np.genfromtxt(path, delimiter=",", names=True, max_rows=1)

    q_cols = [f"actual_q_{i}" for i in range(6)]
    if csv.dtype.names is not None and all(name in csv.dtype.names for name in q_cols):
        q0 = np.array([float(csv[name]) for name in q_cols])
    else:
        q0 = DEFAULT_HOME_Q.copy()

    data.qpos[robot.info.joint_indxs] = q0
    data.qvel[robot.info.dof_indxs] = 0.0
    mj.mj_forward(model, data)

    positions = poses[:, :3]
    rotations = Rotation.from_rotvec(poses[:, 3:])
    quats_xyzw = rotations.as_quat()
    quaternions = np.column_stack([quats_xyzw[:, 3], quats_xyzw[:, :3]])
    quaternions = _continuous_quaternions(quaternions)

    # The UR10e MJCF exposes ``attachment_site``. The recording's TCP pose is
    # offset from that site by a constant tool offset, expressed in TCP frame.
    attachment_id = robot.info.site_ids[0]
    tcp_rotation0 = rotations[0].as_matrix()
    tcp_to_attachment = tcp_rotation0.T @ (data.site_xpos[attachment_id] - positions[0])

    return Demonstration(
        ts=ts,
        positions=positions,
        quaternions=quaternions,
        q0=q0,
        tcp_to_attachment=tcp_to_attachment,
    )


def _pose_from_tcp(
    position: np.ndarray,
    quaternion: np.ndarray,
    tcp_to_attachment: np.ndarray,
) -> sm.SE3:
    rotation = smb.q2r(np.asarray(quaternion, dtype=float))
    attachment_pos = position + rotation @ tcp_to_attachment
    return sm.SE3.Rt(rotation, attachment_pos, check=False)


class Sim(ms.BaseSim):
    def __init__(
        self,
        *,
        csv_path: Path = DEFAULT_CSV,
        pose_prefix: str = "actual_TCP_pose",
        train_samples: int = 700,
        rollout_samples: int = 260,
        n_bfs: int = 80,
        playback_dt: float = 0.02,
        loop: bool = False,
        plot_on_complete: bool = True,
        plot_path: Path = DEFAULT_PLOT,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.pose_prefix = pose_prefix
        self.train_samples = train_samples
        self.rollout_samples = rollout_samples
        self.n_bfs = n_bfs
        self.playback_dt = playback_dt
        self.playing = True
        self.path_index = 0
        self.ik_failures = 0
        self.loop = loop
        self.plot_on_complete = plot_on_complete
        self.plot_path = plot_path
        self.tracking_plotted = False
        self.target_position_log: list[np.ndarray] = []
        self.target_quaternion_log: list[np.ndarray] = []
        self.actual_position_log: list[np.ndarray] = []
        self.actual_quaternion_log: list[np.ndarray] = []

        self._model, self._data = self._build_scene()
        self.ur = ms.Robot(self.model, self.data, "ur/")
        self.site_name = self.ur.info.site_names[0]
        self.joint_qpos_ids = np.array(self.ur.info.joint_indxs)
        self.joint_dof_ids = np.array(self.ur.info.dof_indxs)

        self.demo = _rotvec_poses_to_demo(
            self.csv_path,
            self.pose_prefix,
            self.train_samples,
            self.model,
            self.data,
            self.ur,
        )
        self.dmp_positions, self.dmp_quaternions = self._train_and_rollout_dmp()
        self.q_path, self.ik_errors = self._solve_dmp_ik()
        self._rebuild_scene_with_path_visuals()
        self._set_robot_q(self.q_path[0])

    def _build_scene(
        self,
        *,
        add_path_visuals: bool = False,
    ) -> tuple[mj.MjModel, mj.MjData]:
        scene = ms.empty_scene(
            sim_name="ur10e_cartesian_dmp_demo",
            statistic_center=(0.25, -0.25, 0.45),
            statistic_extent=1.25,
        )
        ur = mj.MjSpec.from_file(ur10e_mj_description.MJCF_PATH)
        scene.attach(ur, "ur/", frame=scene.worldbody.add_frame())
        scene.worldbody.add_camera(
            name="front",
            pos=[0.75, -1.25, 0.85],
            euler=[0.85, 0.0, 0.45],
            fovy=45.0,
        )

        if add_path_visuals:
            self._add_path_visuals(scene)

        model = scene.compile()
        data = mj.MjData(model)
        return model, data

    def _train_and_rollout_dmp(self) -> tuple[np.ndarray, np.ndarray]:
        dmp = DMPCartesian(n_bfs=self.n_bfs)
        tau = float(self.demo.ts[-1])
        dmp.train(self.demo.positions, self.demo.quaternions, self.demo.ts, tau)
        rollout_ts = np.linspace(0.0, tau, self.rollout_samples)
        positions, _, _, quaternions, _, _ = dmp.rollout(rollout_ts, tau)
        quaternions = _continuous_quaternions(quaternions)
        return positions, quaternions

    def _solve_dmp_ik(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.demo.q0.copy()
        q_path = []
        errors = []

        for position, quaternion in zip(self.dmp_positions, self.dmp_quaternions):
            target = _pose_from_tcp(position, quaternion, self.demo.tcp_to_attachment)
            result = self._solve_target_ik(target, q)
            q = result.q
            q_path.append(q.copy())
            errors.append([result.position_error, result.orientation_error])

        return np.asarray(q_path), np.asarray(errors)

    def _solve_target_ik(self, target: sm.SE3, q0: np.ndarray) -> IKResult:
        try:
            return self.ur.solve_ik(
                self.site_name,
                target,
                q0=q0,
                position_cost=1.0,
                orientation_cost=0.4,
                posture_cost=1e-3,
                lm_damping=1e-2,
                max_iters=100,
                pos_tol=5e-3,
                ori_tol=5e-2,
            )
        except Exception:
            try:
                return self.ur.solve_ik(
                    self.site_name,
                    target,
                    q0=q0,
                    position_cost=1.0,
                    orientation_cost=0.0,
                    posture_cost=1e-3,
                    lm_damping=1e-2,
                    max_iters=100,
                    pos_tol=5e-3,
                    ori_tol=np.inf,
                )
            except Exception:
                self.ik_failures += 1
                return self._held_pose_result(target, q0)

    def _held_pose_result(self, target: sm.SE3, q: np.ndarray) -> IKResult:
        qpos_prev = self.data.qpos.copy()
        self.data.qpos[self.joint_qpos_ids] = q
        mj.mj_forward(self.model, self.data)

        site_id = self.ur.info.site_ids[0]
        pos_error = np.linalg.norm(target.t - self.data.site_xpos[site_id])
        actual_rot = self.data.site_xmat[site_id].reshape(3, 3)
        rot_error = Rotation.from_matrix(target.R.T @ actual_rot).magnitude()

        self.data.qpos[:] = qpos_prev
        mj.mj_forward(self.model, self.data)

        return IKResult(
            q=q.copy(),
            success=False,
            iterations=0,
            position_error=float(pos_error),
            orientation_error=float(rot_error),
            qpos=self.ur.robot_q_to_full_qpos(q),
            message="held previous waypoint after IK failure",
        )

    def _add_path_visuals(self, scene: mj.MjSpec) -> None:
        stride = max(1, len(self.demo.positions) // 90)
        for i, point in enumerate(self.demo.positions[::stride]):
            scene.worldbody.add_geom(
                name=f"demo_point_{i:03d}",
                type=mj.mjtGeom.mjGEOM_SPHERE,
                pos=point.tolist(),
                size=[0.006],
                rgba=[0.2, 0.2, 0.2, 0.22],
                contype=0,
                conaffinity=0,
            )

        for i, (a, b) in enumerate(
            zip(self.dmp_positions[:-1], self.dmp_positions[1:])
        ):
            scene.worldbody.add_geom(
                name=f"dmp_path_{i:03d}",
                type=mj.mjtGeom.mjGEOM_CAPSULE,
                fromto=[*a.tolist(), *b.tolist()],
                size=[0.006],
                rgba=[0.95, 0.45, 0.08, 0.92],
                contype=0,
                conaffinity=0,
            )

        scene.worldbody.add_geom(
            name="dmp_start",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            pos=self.dmp_positions[0].tolist(),
            size=[0.025],
            rgba=[0.1, 0.7, 0.2, 1.0],
            contype=0,
            conaffinity=0,
        )
        scene.worldbody.add_geom(
            name="dmp_goal",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            pos=self.dmp_positions[-1].tolist(),
            size=[0.025],
            rgba=[0.1, 0.25, 0.9, 1.0],
            contype=0,
            conaffinity=0,
        )

    def _rebuild_scene_with_path_visuals(self) -> None:
        self._model, self._data = self._build_scene(add_path_visuals=True)
        self.ur = ms.Robot(self.model, self.data, "ur/")
        self.site_name = self.ur.info.site_names[0]
        self.joint_qpos_ids = np.array(self.ur.info.joint_indxs)
        self.joint_dof_ids = np.array(self.ur.info.dof_indxs)

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def data(self) -> mj.MjData:
        return self._data

    def _set_robot_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.joint_qpos_ids] = q
        self.data.qvel[self.joint_dof_ids] = 0.0
        mj.mj_forward(self.model, self.data)

    def _actual_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        site_id = self.ur.info.site_ids[0]
        rotation = self.data.site_xmat[site_id].reshape(3, 3).copy()
        position = self.data.site_xpos[site_id] - rotation @ self.demo.tcp_to_attachment
        quat_xyzw = Rotation.from_matrix(rotation).as_quat()
        quaternion = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return position.copy(), quaternion

    def _record_tracking_sample(self, index: int) -> None:
        actual_position, actual_quaternion = self._actual_tcp_pose()
        self.target_position_log.append(self.dmp_positions[index].copy())
        self.target_quaternion_log.append(self.dmp_quaternions[index].copy())
        self.actual_position_log.append(actual_position)
        self.actual_quaternion_log.append(actual_quaternion)

    def _reset_tracking_log(self) -> None:
        self.target_position_log.clear()
        self.target_quaternion_log.clear()
        self.actual_position_log.clear()
        self.actual_quaternion_log.clear()
        self.tracking_plotted = False

    @staticmethod
    def _relative_rotvecs(quaternions: np.ndarray, reference: Rotation) -> np.ndarray:
        quaternions = _continuous_quaternions(quaternions)
        rotations = Rotation.from_quat(
            np.column_stack([quaternions[:, 1:], quaternions[:, 0]])
        )
        return (rotations * reference.inv()).as_rotvec()

    def plot_tracking(self) -> None:
        if not self.target_position_log:
            print("No tracking samples recorded; skipping plot.")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skipping tracking plot.")
            return

        target_p = np.asarray(self.target_position_log)
        actual_p = np.asarray(self.actual_position_log)
        target_q = _continuous_quaternions(np.asarray(self.target_quaternion_log))
        actual_q = _continuous_quaternions(np.asarray(self.actual_quaternion_log))

        t = np.arange(len(target_p)) * self.playback_dt
        reference = Rotation.from_quat(
            [target_q[0, 1], target_q[0, 2], target_q[0, 3], target_q[0, 0]]
        )
        target_r = self._relative_rotvecs(target_q, reference)
        actual_r = self._relative_rotvecs(actual_q, reference)

        pos_error = np.linalg.norm(actual_p - target_p, axis=1)
        target_rotations = Rotation.from_quat(
            np.column_stack([target_q[:, 1:], target_q[:, 0]])
        )
        actual_rotations = Rotation.from_quat(
            np.column_stack([actual_q[:, 1:], actual_q[:, 0]])
        )
        ori_error = (target_rotations.inv() * actual_rotations).magnitude()

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        labels = ("x", "y", "z")
        for axis, label in enumerate(labels):
            axes[0].plot(t, target_p[:, axis], "--", label=f"target {label}")
            axes[0].plot(t, actual_p[:, axis], label=f"actual {label}")
        axes[0].set_ylabel("position [m]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(ncol=3, fontsize="small")

        for axis, label in enumerate(labels):
            axes[1].plot(t, target_r[:, axis], "--", label=f"target r{label}")
            axes[1].plot(t, actual_r[:, axis], label=f"actual r{label}")
        axes[1].set_ylabel("relative rotvec [rad]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(ncol=3, fontsize="small")

        axes[2].plot(t, pos_error, label="position error [m]")
        axes[2].plot(t, ori_error, label="orientation error [rad]")
        axes[2].set_xlabel("time [s]")
        axes[2].set_ylabel("tracking error")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.suptitle("UR10e Cartesian DMP Tracking")
        fig.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.plot_path, dpi=180)
        print(f"Saved tracking plot to {self.plot_path}")
        print(
            "Tracking error: "
            f"max position={np.max(pos_error):.4f} m, "
            f"rms position={np.sqrt(np.mean(pos_error**2)):.4f} m, "
            f"max orientation={np.max(ori_error):.4f} rad, "
            f"rms orientation={np.sqrt(np.mean(ori_error**2)):.4f} rad"
        )

        if "agg" not in plt.get_backend().lower():
            plt.ion()
            fig.show()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def keyboard_callback(self, key: int) -> None:
        if key == glfw.KEY_P:
            self.playing = not self.playing
            print("Playback:", "running" if self.playing else "paused")
        elif key == glfw.KEY_R:
            self.path_index = 0
            self._reset_tracking_log()
            self._set_robot_q(self.q_path[0])
            print("Reset DMP playback.")
        elif key == glfw.KEY_T:
            self.plot_tracking()

    @ms.thread
    def play_dmp_path(self, ss: ms.SimSync) -> None:
        while True:
            if self.playing:
                index = self.path_index
                self._set_robot_q(self.q_path[index])
                self._record_tracking_sample(index)
                self.path_index += 1

                if self.path_index >= len(self.q_path):
                    if self.loop:
                        self.path_index = 0
                        self._reset_tracking_log()
                    else:
                        self.playing = False
                        self.path_index = len(self.q_path) - 1
                        if self.plot_on_complete and not self.tracking_plotted:
                            self.tracking_plotted = True
                            self.plot_tracking()
            ss.step()
            time.sleep(self.playback_dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--pose-prefix",
        choices=["actual_TCP_pose", "target_TCP_pose"],
        default="actual_TCP_pose",
    )
    parser.add_argument("--train-samples", type=int, default=700)
    parser.add_argument("--rollout-samples", type=int, default=260)
    parser.add_argument("--n-bfs", type=int, default=80)
    parser.add_argument("--playback-dt", type=float, default=0.02)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sim = Sim(
        csv_path=args.csv,
        pose_prefix=args.pose_prefix,
        train_samples=args.train_samples,
        rollout_samples=args.rollout_samples,
        n_bfs=args.n_bfs,
        playback_dt=args.playback_dt,
        loop=args.loop,
        plot_on_complete=not args.no_plot,
        plot_path=args.plot_path,
    )
    max_pos_err, max_ori_err = np.max(sim.ik_errors, axis=0)
    print(f"Loaded demonstration from {args.csv}")
    print(f"Trained Cartesian DMP on {len(sim.demo.ts)} samples.")
    print(f"Generated {len(sim.q_path)} UR10e joint waypoints.")
    if sim.ik_failures:
        print(f"Held previous joint pose for {sim.ik_failures} unsolved waypoints.")
    print(
        f"Max IK error: position={max_pos_err:.4f} m, orientation={max_ori_err:.4f} rad"
    )
    print("Press P to pause/resume playback.")
    print("Press R to restart playback.")
    print("Press T to plot the current tracking log.")
    sim.run(dyn=False)
