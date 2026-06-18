from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mink
import mujoco as mj
import numpy as np
import spatialmath as sm

from mjsim.utils.mj import (
    ObjType,
    RobotInfo,
    get_joint_ddq,
    get_joint_dq,
    get_joint_q,
    get_pose,
    name2id,
)


def _dense_mass_matrix(model: mj.MjModel, data: mj.MjData) -> np.ndarray:
    """Return the full dense generalized inertia matrix for the current state.

    MuJoCo 3.8 introduced ``mjData.M`` in CSR-like sparse symmetric storage and
    ``mju_sym2dense`` as the forward-compatible extraction path. The legacy
    ``mjData.qM`` storage and ``mj_fullM(model, dst, data.qM)`` signature are
    scheduled for removal; upcoming MuJoCo versions change ``mj_fullM`` to
    ``mj_fullM(model, data, dst)``.

    This helper prefers the new ``data.M`` path, then falls back to the future
    ``mj_fullM`` signature, then to the legacy ``qM`` signature.
    """
    dense = np.zeros((model.nv, model.nv))

    if hasattr(data, "M") and hasattr(mj, "mju_sym2dense"):
        try:
            mj.mju_sym2dense(
                dense,
                data.M,
                model.M_rownnz,
                model.M_rowadr,
                model.M_colind,
            )
            return dense
        except TypeError:
            try:
                # Some C-level documentation includes nv explicitly. Keep this
                # branch for bindings that expose that spelling.
                mj.mju_sym2dense(
                    dense,
                    data.M,
                    model.nv,
                    model.M_rownnz,
                    model.M_rowadr,
                    model.M_colind,
                )
                return dense
            except TypeError:
                pass

    try:
        mj.mj_fullM(model, data, dense)
        return dense
    except TypeError:
        if not hasattr(data, "qM"):
            raise

    mj.mj_fullM(model, dense, data.qM)
    return dense


def sm_to_smx(T: sm.SE3):
    """Convert a SpatialMath SE3 pose to a Mink SE3 pose."""
    return mink.SE3.from_matrix(np.array(T.A))


def smx_to_sm(T) -> sm.SE3:
    """Convert a Mink SE3 pose to a SpatialMath SE3 pose."""
    return sm.SE3(np.array(T.as_matrix()))


@dataclass(frozen=True)
class IKResult:
    """Result from a Robot inverse-kinematics solve."""

    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float
    qpos: np.ndarray
    message: str = ""


class Robot:
    """Base class for robot simulation in MuJoCo."""

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        namespace: str,
        base_identifier: Optional[Union[int, str]] = None,
    ):
        """Initialize a robot wrapper.

        Args:
            model: MuJoCo model.
            data: MuJoCo data.
            namespace: Namespace/prefix used for this robot in the MJCF.
            base_identifier: Optional base body id or name.
        """
        self._model = model
        self._data = data
        self._name = namespace
        self._info = RobotInfo(self._model, namespace)
        self._base = 0 if base_identifier is None else base_identifier

        self._ik_conf = mink.Configuration(self.model)

    @property
    def name(self) -> str:
        """
        Get the name of the robot.

        This property returns the name of the robot as a string. The name is typically a unique identifier
        used to distinguish between different robots in the simulation environment.

        Returns
        -------
        str
            The name of the robot as a string.
        """
        return self._name

    @property
    def data(self) -> mj.MjData:
        """
        Access the current simulation data.

        This property provides access to an instance of the `MjData` class, which contains the dynamic
        simulation state. This includes quantities such as joint positions, velocities,
        actuator forces, and sensory information. The `MjData` object is updated at each simulation step
        and can be used to inspect the real-time state of the robot during the simulation.

        Returns
        -------
        mj.MjData
            An object representing the current dynamic state of the simulation.
        """
        return self._data

    @property
    def model(self) -> mj.MjModel:
        """
        Access the model of the MuJoCo simulation.

        This property returns an instance of the `MjModel` class, which describes the physical and
        mechanical properties of the simulation. The `MjModel` object contains static information about the
        robot such as its kinematic tree, inertial properties, joint and actuator definitions, and geometry
        configurations. It is used to define the robot's structure and behavior within the simulation.

        Returns
        -------
        mj.MjModel
            An object representing the static model of the robot and overall MuJoCo simulation.
        """
        return self._model

    @property
    def info(self) -> RobotInfo:
        """
        Get detailed information about the robot.

        This property returns an instance of the `RobotInfo` class, which provides comprehensive
        details about the robot's structure and components. This includes information on the robot's
        bodies, joints, actuators, and geometries, among other attributes. The `RobotInfo` instance
        can be used to access various properties such as the number of joints, actuator limits, joint
        limits, and more.

        Returns
        -------
        RobotInfo
            An object containing detailed information about the robot's configuration and components.
        """
        return self._info

    def set_ctrl(self, x: Union[list, np.ndarray]) -> None:
        """
        This function sends the control signal to the simulated robot.

        Args
        ----------
                x (Union[list, np.ndarray]): control signal
        """
        assert len(x) == self._info.n_actuators, (
            f"Number of actuators and control input does not match in dimensions, number of actuators {self._info.n_actuators} and length of control input {len(x)}"
        )
        for i, xi in enumerate(x):
            self.data.actuator(self.info.actuator_ids[i]).ctrl = xi

    @property
    def ctrl(self) -> np.ndarray:
        """
        The control signal sent to the robot's actuator(s).
        """
        return np.array(
            [self.data.actuator(aid).ctrl for aid in self._info._actuator_ids]
        ).flatten()

    def Jp(
        self, base_frame: Union[str, int] = None, site_frame: Union[str, int] = None
    ) -> np.ndarray:
        """
        Get the position (linear velocity) Jacobian expressed in the specified base frame.

        The position Jacobian relates joint velocities to linear velocity of the site frame:
        v = Jp @ q_dot, where v is the 3D linear velocity vector.

        Parameters
        ----------
        base_frame : Union[str, int], optional
            The reference frame in which the Jacobian is expressed.
            If None, uses the robot's base body frame.
        site_frame : Union[str, int], optional
            The site frame for which to compute the Jacobian.
            If None, uses the first site in the robot.

        Returns
        -------
        np.ndarray
            Position Jacobian matrix of shape (3 x nv), where nv is the number of
            degrees of freedom. This matrix maps joint velocities to linear velocity
            components in the specified base frame.
        """
        return self.J(base_frame=base_frame, site_frame=site_frame)[:3, :]

    def Jo(
        self, base_frame: Union[str, int] = None, site_frame: Union[str, int] = None
    ) -> np.ndarray:
        """
        Get the orientation (angular velocity) Jacobian expressed in the specified base frame.

        The orientation Jacobian relates joint velocities to angular velocity of the site frame:
        ω = Jo @ q_dot, where ω is the 3D angular velocity vector.

        Parameters
        ----------
        base_frame : Union[str, int], optional
            The reference frame in which the Jacobian is expressed.
            If None, uses the robot's base body frame.
        site_frame : Union[str, int], optional
            The site frame for which to compute the Jacobian.
            If None, uses the first site in the robot.

        Returns
        -------
        np.ndarray
            Orientation Jacobian matrix of shape (3 x nv), where nv is the number of
            degrees of freedom. This matrix maps joint velocities to angular velocity
            components in the specified base frame.
        """
        return self.J(base_frame=base_frame, site_frame=site_frame)[3:, :]

    def J(
        self, base_frame: Union[str, int] = None, site_frame: Union[str, int] = None
    ) -> np.ndarray:
        """Get the full Jacobian for a site expressed in the chosen base frame.

        Args:
            base_frame: Body frame name or id to express the Jacobian in. Defaults
                to the robot base.
            site_frame: Site name or id to compute the Jacobian for. Defaults to
                the first site.

        Returns:
            ``(6, nv)`` Jacobian where rows 0..2 are linear velocity and 3..5
            are angular velocity components for the site.
        """

        # Set default frames if not provided
        if base_frame is None:
            base_frame = self._info.body_ids[0]
        if site_frame is None:
            site_frame = self._info.site_ids[0]

        # Convert frame names to IDs if strings are provided
        if isinstance(base_frame, str):
            base_frame = name2id(self.model, f"{self.name}/{base_frame}", ObjType.BODY)

        if isinstance(site_frame, str):
            site_frame = name2id(self.model, f"{self.name}/{site_frame}", ObjType.SITE)

        # Initialize Jacobian matrix
        sys_J = np.zeros((6, self.model.nv))

        # Compute Jacobian using MuJoCo function
        # First 3 rows: linear velocity Jacobian, last 3 rows: angular velocity Jacobian
        mj.mj_jacSite(
            self.model,
            self.data,
            sys_J[:3],  # Jacobian for linear velocity
            sys_J[3:],  # Jacobian for angular velocity
            site_frame,  # Use the specified site frame
        )

        # Extract only the DOFs relevant to this robot
        sys_J = sys_J[:, self.info._dof_indxs]

        # If base_frame is not world frame (body_id 0), transform Jacobian to base frame
        if base_frame != 0:
            # Get rotation matrix from world frame to base frame
            base_pose = get_pose(self.model, self.data, base_frame, ObjType.BODY)
            R_base_world = base_pose.R

            # Transform both linear and angular Jacobians to base frame
            # For angular velocity Jacobian: J_ω_base = R_base_world * J_ω_world
            # For linear velocity Jacobian: J_v_base = R_base_world * J_v_world
            sys_J[3:, :] = R_base_world @ sys_J[3:, :]  # Angular part
            sys_J[:3, :] = R_base_world @ sys_J[:3, :]  # Linear part

        return sys_J

    @property
    def c(self) -> np.ndarray:
        """
        bias force: Coriolis, centrifugal, gravitational
        """
        return self.data.qfrc_bias[np.ravel(self.info._dof_indxs)]

    @property
    def Mq(self) -> np.ndarray:
        """Joint-space inertia matrix."""
        sys_Mq = _dense_mass_matrix(self.model, self.data)
        dof_indices = self.robot_dof_indices
        self._Mq = sys_Mq[np.ix_(dof_indices, dof_indices)]
        return self._Mq

    def Mx(
        self, base_frame: Union[str, int] = None, site_frame: Union[str, int] = None
    ) -> np.ndarray:
        """Task-space inertia matrix for a site and base frame.

        Args:
            base_frame: Body frame to express inertia in. Defaults to robot base.
            site_frame: Site to evaluate. Defaults to first site.

        Returns:
            ``(6, 6)`` task-space inertia matrix.
        """
        Mx_inv = (
            self.J(base_frame=base_frame, site_frame=site_frame)
            @ np.linalg.inv(self.Mq)
            @ self.J(base_frame=base_frame, site_frame=site_frame).T
        )

        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            self._Mx = np.linalg.inv(Mx_inv)
        else:
            self._Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        return self._Mx

    @property
    def q(self) -> np.ndarray:
        """Joint positions."""
        return np.array(
            [get_joint_q(self.data, self.model, jn) for jn in self.info._joint_ids]
        ).flatten()

    @property
    def dq(self) -> np.ndarray:
        """Joint velocities."""
        return np.array(
            [get_joint_dq(self.data, self.model, jn) for jn in self.info._joint_ids]
        ).flatten()

    @property
    def ddq(self) -> np.ndarray:
        """Joint accelerations."""
        return np.array(
            [get_joint_ddq(self.data, self.model, jn) for jn in self.info._joint_ids]
        ).flatten()

    @property
    def robot_qpos_indices(self) -> np.ndarray:
        """Full-model qpos indices controlled by this robot wrapper."""
        indices: list[int] = []
        for joint_id in self.info._joint_ids:
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == int(mj.mjtJoint.mjJNT_FREE):
                width = 7
            elif joint_type == int(mj.mjtJoint.mjJNT_BALL):
                width = 4
            else:
                width = 1
            indices.extend(range(qpos_adr, qpos_adr + width))
        return np.array(indices, dtype=int)

    @property
    def robot_dof_indices(self) -> np.ndarray:
        """Full-model qvel/qfrc DOF indices controlled by this robot wrapper."""
        indices: list[int] = []
        for joint_id in self.info._joint_ids:
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == int(mj.mjtJoint.mjJNT_FREE):
                width = 6
            elif joint_type == int(mj.mjtJoint.mjJNT_BALL):
                width = 3
            else:
                width = 1
            indices.extend(range(dof_adr, dof_adr + width))
        return np.array(indices, dtype=int)

    def full_qpos_to_robot_q(self, qpos_full: np.ndarray) -> np.ndarray:
        """Extract this robot's qpos block from a full MuJoCo qpos vector."""
        qpos_full = np.asarray(qpos_full, dtype=float).reshape(-1)
        if qpos_full.size != self.model.nq:
            msg = f"qpos_full has size {qpos_full.size}, expected {self.model.nq}"
            raise ValueError(msg)
        return qpos_full[self.robot_qpos_indices].copy()

    def robot_q_to_full_qpos(
        self, q_robot: np.ndarray, qpos_ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Insert a robot-local q vector into a full MuJoCo qpos vector."""
        q_robot = np.asarray(q_robot, dtype=float).reshape(-1)
        if q_robot.size == self.model.nq:
            return q_robot.copy()

        qpos_indices = self.robot_qpos_indices
        if q_robot.size != qpos_indices.size:
            msg = (
                f"q_robot has size {q_robot.size}, expected {qpos_indices.size} "
                f"for robot {self.name!r} or {self.model.nq} for full qpos"
            )
            raise ValueError(msg)

        qpos_full = (
            self.data.qpos.copy()
            if qpos_ref is None
            else np.asarray(qpos_ref, dtype=float).reshape(-1).copy()
        )
        if qpos_full.size != self.model.nq:
            msg = f"qpos_ref has size {qpos_full.size}, expected {self.model.nq}"
            raise ValueError(msg)
        qpos_full[qpos_indices] = q_robot
        return qpos_full

    def _resolve_frame_name(self, frame: Union[str, int], frame_type: str) -> str:
        if frame_type not in {"site", "body", "geom"}:
            msg = f"Unsupported frame_type {frame_type!r}; expected site, body, or geom"
            raise ValueError(msg)

        obj_type = {
            "site": mj.mjtObj.mjOBJ_SITE,
            "body": mj.mjtObj.mjOBJ_BODY,
            "geom": mj.mjtObj.mjOBJ_GEOM,
        }[frame_type]

        if isinstance(frame, int):
            name = mj.mj_id2name(self.model, obj_type, frame)
            if name is None:
                msg = f"No {frame_type} with id {frame}"
                raise ValueError(msg)
            return name

        candidates = [frame]
        namespace = self.name.strip("/")
        if namespace and not frame.startswith(f"{namespace}/"):
            candidates.append(f"{namespace}/{frame}")

        for candidate in candidates:
            if mj.mj_name2id(self.model, obj_type, candidate) >= 0:
                return candidate

        msg = f"No {frame_type} named {frame!r}; tried {candidates}"
        raise ValueError(msg)

    def _target_to_mink(
        self,
        target: Union[sm.SE3, np.ndarray, object],
        base_frame: Optional[Union[sm.SE3, str, int]],
        qpos_full: np.ndarray,
    ):
        if isinstance(target, sm.SE3):
            target_world = sm_to_smx(target)
        elif isinstance(target, np.ndarray):
            target_world = mink.SE3.from_matrix(target)
        elif hasattr(target, "as_matrix"):
            target_world = target
        else:
            msg = f"Unsupported IK target type: {type(target)}"
            raise TypeError(msg)

        if base_frame is None:
            return target_world

        if isinstance(base_frame, sm.SE3):
            T_world_base = sm_to_smx(base_frame)
        else:
            frame_name = self._resolve_frame_name(base_frame, "body")
            conf = mink.Configuration(self.model, q=qpos_full)
            T_world_base = conf.get_transform_frame_to_world(frame_name, "body")

        return T_world_base @ target_world

    @staticmethod
    def _task_error_norms(
        task, configuration: mink.Configuration
    ) -> tuple[float, float]:
        error = np.asarray(task.compute_error(configuration), dtype=float).reshape(-1)
        if error.size < 6:
            return float(np.linalg.norm(error)), 0.0
        return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:6]))

    def solve_ik(
        self,
        frame: Union[str, int],
        target: Union[sm.SE3, np.ndarray, object],
        *,
        frame_type: str = "site",
        q0: Optional[np.ndarray] = None,
        base_frame: Optional[Union[sm.SE3, str, int]] = None,
        position_cost: Union[float, np.ndarray] = 1.0,
        orientation_cost: Union[float, np.ndarray] = 1.0,
        posture_cost: Union[float, np.ndarray] = 1e-3,
        damping: float = 1e-6,
        lm_damping: float = 1e-3,
        solver: str = "daqp",
        max_iters: int = 100,
        pos_tol: float = 1e-3,
        ori_tol: float = 1e-2,
        dt: Optional[float] = None,
        inplace: bool = False,
        collision_avoidance_pairs: Optional[List[Tuple[List[str], List[str]]]] = None,
        min_collision_distance: float = 0.05,
        collision_detection_distance: float = 0.1,
        verbose: bool = False,
    ) -> IKResult:
        """Solve single-frame inverse kinematics using Mink.

        ``target`` is interpreted in world coordinates unless ``base_frame`` is
        provided. The returned ``IKResult.q`` is robot-local and matches
        ``robot.q``; ``IKResult.qpos`` is the full MuJoCo qpos solution.
        """
        frame_name = self._resolve_frame_name(frame, frame_type)
        qpos_initial = (
            self.data.qpos.copy()
            if q0 is None
            else self.robot_q_to_full_qpos(q0, self.data.qpos)
        )
        configuration = mink.Configuration(self.model, q=qpos_initial)
        target_world = self._target_to_mink(target, base_frame, qpos_initial)

        frame_task = mink.FrameTask(
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping,
        )
        frame_task.set_target(target_world)

        posture_task = mink.PostureTask(
            self.model,
            cost=posture_cost,
            lm_damping=lm_damping,
        )
        posture_task.set_target(qpos_initial)

        tasks = [frame_task, posture_task]
        limits = [mink.ConfigurationLimit(self.model)]
        if collision_avoidance_pairs:
            limits.append(
                mink.CollisionAvoidanceLimit(
                    model=self.model,
                    geom_pairs=collision_avoidance_pairs,
                    minimum_distance_from_collisions=min_collision_distance,
                    collision_detection_distance=collision_detection_distance,
                )
            )

        dt = self.model.opt.timestep if dt is None else float(dt)
        position_error, orientation_error = self._task_error_norms(
            frame_task, configuration
        )
        success = position_error <= pos_tol and orientation_error <= ori_tol
        iterations = 0

        for iterations in range(1, max_iters + 1):
            if success:
                break

            velocity = mink.solve_ik(
                configuration,
                tasks,
                dt,
                solver,
                damping,
                limits=limits,
            )
            configuration.update(configuration.integrate(velocity, dt))
            position_error, orientation_error = self._task_error_norms(
                frame_task, configuration
            )
            success = position_error <= pos_tol and orientation_error <= ori_tol

            if verbose:
                print(
                    f"IK iter {iterations}: "
                    f"pos_err={position_error:.3e}, "
                    f"ori_err={orientation_error:.3e}"
                )

        qpos_solution = configuration.q.copy()
        q_solution = self.full_qpos_to_robot_q(qpos_solution)

        if inplace:
            self.data.qpos[:] = qpos_solution
            mj.mj_forward(self.model, self.data)

        message = "converged" if success else "maximum iterations reached"
        return IKResult(
            q=q_solution,
            success=success,
            iterations=iterations,
            position_error=position_error,
            orientation_error=orientation_error,
            qpos=qpos_solution,
            message=message,
        )

    def ik(
        self,
        T_target: Union[sm.SE3, List[sm.SE3]],
        q0: Optional[np.ndarray] = None,
        site_names: Optional[Union[List[str], str]] = None,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        lm_damping: float = 1.0,
        posture_cost: float = 1e-2,
        solver: str = "daqp",
        regularization: float = 1e-5,
        collision_avoidance_pairs: Optional[List[Tuple[List[str], List[str]]]] = None,
        min_collision_distance: float = 0.05,
        collision_detection_distance: float = 0.1,
        max_iterations: int = 10,
        max_attempts: int = 3,
        tolerance: float = 1e-4,
        task_position_tolerance: float = 1e-3,
        task_orientation_tolerance: float = 0.01,  # radians
        verbose: bool = False,
    ) -> np.ndarray:
        """Solve inverse kinematics for target poses.

        Args:
            T_target: Target pose(s) as SpatialMath SE3 or mink SE3 list.
            q0: Initial joint configuration; defaults to the current config.
            site_names: Site(s) to control; defaults to all robot sites.
            position_cost: Cost weight for position tracking.
            orientation_cost: Cost weight for orientation tracking.
            lm_damping: Levenberg–Marquardt damping.
            posture_cost: Cost weight for posture regularization.
            solver: Optimization solver name (e.g. ``"daqp"``).
            regularization: Regularization term for the solver.
            collision_avoidance_pairs: Optional collision pairs to avoid.
            min_collision_distance: Minimum allowed distance for collisions.
            collision_detection_distance: Distance where collision detection starts.
            max_iterations: Iterations per attempt before retrying.
            max_attempts: Number of attempts to retry before giving up.
            tolerance: Joint-space convergence tolerance on ``q`` change.
            task_position_tolerance: Position error tolerance in meters.
            task_orientation_tolerance: Orientation error tolerance in radians.
            verbose: Whether to print convergence diagnostics.

        Returns:
            Joint configuration that achieves the target(s) within tolerance.
        """
        _ = tolerance
        if site_names is None:
            if not self._info.site_names:
                msg = "site_names must be provided because this robot has no sites"
                raise ValueError(msg)
            frame = self._info.site_names[0]
        elif isinstance(site_names, list):
            if len(site_names) != 1:
                msg = "Robot.ik supports one site; call solve_ik per target"
                raise NotImplementedError(msg)
            frame = site_names[0]
        else:
            frame = site_names

        if isinstance(T_target, list):
            if len(T_target) != 1:
                msg = "Robot.ik supports one target; call solve_ik per target"
                raise NotImplementedError(msg)
            target = T_target[0]
        else:
            target = T_target

        result = self.solve_ik(
            frame=frame,
            target=target,
            frame_type="site",
            q0=q0,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            posture_cost=posture_cost,
            damping=regularization,
            lm_damping=lm_damping,
            solver=solver,
            max_iters=max_iterations * max_attempts,
            pos_tol=task_position_tolerance,
            ori_tol=task_orientation_tolerance,
            collision_avoidance_pairs=collision_avoidance_pairs,
            min_collision_distance=min_collision_distance,
            collision_detection_distance=collision_detection_distance,
            verbose=verbose,
        )
        if verbose and not result.success:
            print(
                f"IK did not converge: pos_err={result.position_error:.3e}, "
                f"ori_err={result.orientation_error:.3e}"
            )
        return result.q

    def fk(
        self,
        q: np.ndarray,
        sites: Optional[Union[str, int, list[str], list[int]]] = None,
        base_frame: Optional[Union[sm.SE3, int, str]] = None,
    ) -> Union[sm.SE3, list[sm.SE3]]:
        """Forward kinematics for one or multiple sites.

        Args:
            q: Joint configuration.
            sites: Site name/id or list thereof; defaults to all robot sites.
            base_frame: Base frame to express poses in (body id/name or SE3).

        Returns:
            SpatialMath SE3 pose or list of poses for the requested sites.
        """
        assert len(q) == self._info.n_joints, (
            f"To compute the forward kinematics, the length of q must be equal to the number of joints in the robot, {len(q)=}, {self._info.n_joints=}"
        )

        base_frame = self._base if base_frame is None else base_frame

        # Validate base_frame
        if isinstance(base_frame, (int, str)):
            # Convert to body ID and check if it exists in the robot

            if isinstance(base_frame, str):
                base_body_id = name2id(self.model, base_frame, ObjType.BODY)
            else:
                base_body_id = base_frame

            assert base_body_id in self._info.body_ids, (
                f"base_frame '{base_frame}' (body ID {base_body_id}) is not part of the robot. "
                f"Available body IDs: {self._info.body_ids}"
            )
        elif isinstance(base_frame, sm.SE3):
            # External base frame is always valid
            pass
        else:
            raise ValueError(f"Invalid base_frame type: {type(base_frame)}")

        # Handle sites parameter
        if sites is None:
            sites = self._info.site_ids
        elif isinstance(sites, (str, int)):
            sites = [sites]

        # Convert all sites to IDs and validate they belong to the robot
        site_ids = []
        for site in sites:
            if isinstance(site, str):
                site_id = name2id(self.model, site, ObjType.SITE)
            else:
                site_id = site
            assert site_id in self._info.site_ids, (
                f"Site '{site}' (site ID {site_id}) is not part of the robot. "
                f"Available site IDs: {self._info.site_ids}"
            )
            site_ids.append(site_id)

        _data_q = self.data.qpos[self._info.joint_indxs]
        # overwrite qpos
        self.data.qpos[self._info.joint_indxs] = q
        mj.mj_forward(self.model, self.data)

        T = [
            get_pose(self.model, self.data, base_frame, ObjType.BODY).inv()
            @ get_pose(self.model, self.data, sid, ObjType.SITE)
            for sid in site_ids
        ]

        self.data.qpos[self._info.joint_indxs] = _data_q
        return T if len(T) > 1 else T[0]
