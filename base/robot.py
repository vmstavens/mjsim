from abc import ABC, abstractmethod
from typing import List, Union

import mujoco as mj
import numpy as np

from utils.mj import (
    ObjType,
    RobotInfo,
    get_joint_ddq,
    get_joint_dq,
    get_joint_q,
    get_pose,
    name2id,
)


class Robot:
    """
    Base class for robot simulation in MuJoCo.

    This class provides a framework for simulating robots in MuJoCo environments. It defines
    key properties and methods that should be implemented in child classes, including access
    to the robot's model, data, and control mechanisms.
    """

    def __init__(self, model: mj.MjModel, data: mj.MjData, namespace: str):
        self._model = model
        self._data = data
        self._name = namespace
        self._info = RobotInfo(self._model, namespace)

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
    def ctrl(self) -> List[float]:
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
        """
        Get the full Jacobian for the specified site frame expressed in the specified base frame.

        Parameters
        ----------
        base_frame : Union[str, int], optional
            The reference frame in which the Jacobian is expressed.
            If None, uses the robot's base body.
        site_frame : Union[str, int], optional
            The site for which to compute the Jacobian.
            If None, uses the first site in the robot.

        Returns
        -------
        np.ndarray
            Full Jacobian (6 x nv) where first 3 rows are linear velocity,
            last 3 rows are angular velocity components.
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
        """
        Getter property for the inertia matrix M(q) in joint space.

        Returns
        ----------
        - numpy.ndarray: Symmetric inertia matrix in joint space.
        """
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
        """
        Compute the task-space inertia matrix for the specified site and base frames.

        The task-space inertia matrix Mx(q) maps task-space accelerations to task-space forces:
        F = Mx(q) @ x_ddot, where F is the 6D wrench (force-torque vector) and x_ddot is the
        6D spatial acceleration.

        The matrix is computed from the joint-space inertia matrix Mq using the Jacobian:
        Mx(q) = (J(q) * Mq⁻¹ * J(q)ᵀ)⁻¹

        For singular configurations where the inverse is ill-conditioned, the pseudoinverse
        is used with a conditioning threshold.

        Parameters
        ----------
        base_frame : Union[str, int], optional
            The reference frame in which the task-space inertia is expressed.
            If None, uses the robot's base body frame.
        site_frame : Union[str, int], optional
            The site frame for which to compute the task-space inertia.
            If None, uses the first site in the robot.

        Returns
        -------
        np.ndarray
            Task-space inertia matrix of shape (6, 6). This symmetric positive semi-definite
            matrix represents the apparent inertia at the site frame when projected through
            the current robot configuration.

        Notes
        -----
        - The matrix is 6x6, combining both linear and rotational inertia components
        - For singular configurations (near kinematic singularities), the pseudoinverse
        provides a numerically stable solution
        - The conditioning threshold (rcond=1e-2) determines when to switch to pseudoinverse
        - The result is cached in self._Mx for potential reuse

        Examples
        --------
        >>> # Compute task-space inertia for end-effector in base frame
        >>> Mx = robot.Mx(base_frame="base", site_frame="end_effector")
        >>> # Use for operational space control: F = Mx * x_ddot_desired
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
        """
        Get the joint positions.

        Returns
        ----------
                Joint positions as a numpy array.
        """
        return np.array(
            [get_joint_q(self.data, self.model, jn) for jn in self.info._joint_ids]
        ).flatten()

    @property
    def dq(self) -> np.ndarray:
        """
        Get the joint velocities.

        Returns
        ----------
                Joint velocities as a numpy array.
        """
        return np.array(
            [get_joint_dq(self.data, self.model, jn) for jn in self.info._joint_ids]
        ).flatten()

    @property
    def ddq(self) -> np.ndarray:
        """
        Get the joint accelerations.

        Returns
        ----------
                Joint accelerations as a numpy array.
        """
        return np.array(
            [get_joint_ddq(self.data, self.model, jn) for jn in self.info._joint_ids]
        ).flatten()
