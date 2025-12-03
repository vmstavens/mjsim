from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import mink
import mujoco as mj
import numpy as np
import spatialmath as sm

from mjsim.utils.jax import sm_to_smx, smx_to_sm
from mjsim.utils.mj import (
    ObjType,
    RobotInfo,
    get_joint_ddq,
    get_joint_dq,
    get_joint_q,
    get_pose,
    name2id,
)


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
        sys_Mq_inv = np.zeros((self.model.nv, self.model.nv))

        mj.mj_solveM(self.model, self.data, sys_Mq_inv, np.eye(self.model.nv))

        dof_indices = np.ravel(self.info._dof_indxs)  # Flatten to 1D if not already
        Mq_inv = sys_Mq_inv[np.ix_(dof_indices, dof_indices)]

        if abs(np.linalg.det(Mq_inv)) >= 1e-2:
            self._Mq = np.linalg.inv(Mq_inv)
        else:
            self._Mq = np.linalg.pinv(Mq_inv, rcond=1e-2)
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
        # Handle site_names parameter
        if site_names is None:
            site_names = self._info.site_names
        elif isinstance(site_names, str):
            site_names = [site_names]

        from mink import SE3

        # Handle T_target parameter and keep both mink and spatialmath variants for error checks
        if isinstance(T_target, sm.SE3):
            targets_smx = [sm_to_smx(T_target)]
            targets_sm = [T_target]
        elif (
            isinstance(T_target, list)
            and len(T_target) > 0
            and isinstance(T_target[0], sm.SE3)
        ):
            targets_smx = [sm_to_smx(T) for T in T_target]
            targets_sm = T_target
        else:
            targets_smx = T_target
            targets_sm = [smx_to_sm(T) for T in T_target]

        assert len(targets_smx) == len(site_names), (
            f"Number of target poses ({len(targets_smx)}) must match number of sites ({len(site_names)})"
        )

        # Set initial configuration if provided
        current_q = self._ik_conf.q.copy()
        if q0 is not None:
            self._ik_conf.update(q0)
            q = q0.copy()
        else:
            q = current_q.copy()

        # Store original configuration to restore later
        original_config = self._ik_conf.q.copy()
        q_change = np.inf

        converged = False
        attempts = 0
        while attempts < max_attempts and not converged:
            for iteration in range(max_iterations):
                # Update configuration for this iteration
                self._ik_conf.update(q)

                # Create tasks
                tasks = []

                # Posture task for regularization
                posture_task = mink.PostureTask(self.model, cost=posture_cost)
                posture_task.set_target_from_configuration(self._ik_conf)
                tasks.append(posture_task)

                # Frame tasks for each site
                frame_tasks = []
                for i, site_name in enumerate(site_names):
                    task = mink.FrameTask(
                        frame_name=site_name,
                        frame_type="site",
                        position_cost=position_cost,
                        orientation_cost=orientation_cost,
                        lm_damping=lm_damping,
                    )
                    task.set_target(targets_smx[i])
                    frame_tasks.append(task)
                    tasks.append(task)

                # Create limits (constraints)
                limits = [mink.ConfigurationLimit(self.model)]

                # Add collision avoidance if specified
                if collision_avoidance_pairs:
                    collision_limit = mink.CollisionAvoidanceLimit(
                        model=self.model,
                        geom_pairs=collision_avoidance_pairs,
                        minimum_distance_from_collisions=min_collision_distance,
                        collision_detection_distance=collision_detection_distance,
                    )
                    limits.append(collision_limit)

                # Solve IK for this iteration
                vel = mink.solve_ik(
                    self._ik_conf,
                    tasks,
                    self.model.opt.timestep,
                    solver,
                    regularization,
                    limits=limits,
                )

                # Integrate to get new configuration
                q_new = self._ik_conf.integrate(vel, self.model.opt.timestep)

                # Check joint space convergence
                q_change = np.linalg.norm(q_new - q)
                joint_converged = q_change < tolerance

                # Check task space convergence every iteration
                task_converged = True
                max_position_error = 0.0
                max_orientation_error = 0.0

                # Update configuration to compute FK
                self._ik_conf.update(q_new)

                def _translation(pose: sm.SE3) -> np.ndarray:
                    """Return translation for both spatialmath and mink SE3 types."""
                    if hasattr(pose, "translation"):
                        return np.array(pose.translation()).reshape(-1)
                    if hasattr(pose, "t"):
                        return np.array(pose.t).reshape(-1)
                    raise AttributeError("Pose does not expose translation or t")

                def _rotation_matrix(pose: sm.SE3) -> np.ndarray:
                    """Return rotation matrix for both spatialmath and mink SE3 types."""
                    if hasattr(pose, "R"):
                        return np.array(pose.R)
                    if hasattr(pose, "rotation"):
                        rot = pose.rotation()
                        if hasattr(rot, "matrix"):
                            return np.array(rot.matrix())
                        if hasattr(rot, "as_matrix"):
                            return np.array(rot.as_matrix())
                    raise AttributeError("Pose does not expose R or rotation matrix")

                for i, site_name in enumerate(site_names):
                    current_pose = self.fk(q_new, site_name, base_frame=self._base)
                    target_pose = targets_sm[i]

                    # Position error
                    pos_error = np.linalg.norm(
                        _translation(current_pose) - _translation(target_pose)
                    )
                    max_position_error = max(max_position_error, pos_error)
                    if pos_error > task_position_tolerance:
                        task_converged = False

                    # Orientation error (angle in radians)
                    cur_R = _rotation_matrix(current_pose)
                    tgt_R = _rotation_matrix(target_pose)
                    rot_error = cur_R.T @ tgt_R
                    angle_error = np.arccos(np.clip((np.trace(rot_error) - 1) / 2, -1, 1))
                    max_orientation_error = max(max_orientation_error, angle_error)
                    if angle_error > task_orientation_tolerance:
                        task_converged = False

                if verbose and (
                    iteration % 3 == 0 or joint_converged or task_converged
                ):
                    print(
                        f"IK iter {iteration + 1}: "
                        f"Δq = {q_change:.2e}, "
                        f"max_pos_err = {max_position_error:.2e}, "
                        f"max_rot_err = {max_orientation_error:.2e}"
                    )

                # Check if we should stop
                if joint_converged and task_converged:
                    converged = True
                    if verbose:
                        if joint_converged:
                            print(
                                f"IK converged in joint space after {iteration + 1} iterations (Δq = {q_change:.2e})"
                            )
                        else:
                            print(
                                f"IK converged in task space after {iteration + 1} iterations "
                                f"(pos_err ≤ {task_position_tolerance:.1e}, rot_err ≤ {task_orientation_tolerance:.1e})"
                            )
                    q = q_new
                    break

                q = q_new

            attempts += 1

        if not converged and verbose:
            # Compute final errors for reporting
            self._ik_conf.update(q)
            final_position_errors = []
            final_orientation_errors = []

            for i, site_name in enumerate(site_names):
                current_pose = self.fk(q, site_name, base_frame=self._base)
                target_pose = targets_sm[i]

                pos_error = np.linalg.norm(
                    current_pose.translation() - target_pose.translation()
                )
                final_position_errors.append(pos_error)

                rot_error = current_pose.rotation().inv() * target_pose.rotation()
                angle_error = np.arccos(
                    np.clip((np.trace(rot_error.matrix()) - 1) / 2, -1, 1)
                )
                final_orientation_errors.append(angle_error)

            max_final_pos = max(final_position_errors) if final_position_errors else 0
            max_final_rot = (
                max(final_orientation_errors) if final_orientation_errors else 0
            )

            print(
                f"IK reached maximum attempts ({max_attempts}) "
                f"and iterations ({max_iterations}) without converging"
            )
            print(
                f"Final errors - Δq: {q_change:.2e}, "
                f"max_pos: {max_final_pos:.2e}, max_rot: {max_final_rot:.2e}"
            )

        # Restore original configuration state
        self._ik_conf.update(original_config)

        return q

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
