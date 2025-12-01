from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ompl
import ompl.base as ob
import ompl.geometric as og

from mjsim.base.robot import Robot


def get_compound_pose(state):
    """Extract pose from compound state (position + SO3)"""
    # Position (first subspace)
    pos = state()[0]
    x = pos[0]
    y = pos[1]
    z = pos[2]

    # Orientation (second subspace - SO3)
    rot = state()[1]
    # For SO3StateSpace, we can get the quaternion
    qx = rot.x
    qy = rot.y
    qz = rot.z
    qw = rot.w

    return np.array([x, y, z, qx, qy, qz, qw])


def set_compound_state(state, pose: np.ndarray):
    """Set compound state (position + SO3) from pose array"""
    # Position (first subspace)
    pos = state()[0]
    pos[0] = float(pose[0])  # x
    pos[1] = float(pose[1])  # y
    pos[2] = float(pose[2])  # z

    # Orientation (second subspace - SO3)
    rot = state()[1]
    if len(pose) == 7:
        # pose: [x, y, z, qx, qy, qz, qw]
        rot.x = float(pose[3])
        rot.y = float(pose[4])
        rot.z = float(pose[5])
        rot.w = float(pose[6])
    elif len(pose) == 6:
        # pose: [x, y, z, roll, pitch, yaw]
        # Convert Euler angles to quaternion
        from scipy.spatial.transform import Rotation

        r = Rotation.from_euler("xyz", pose[3:6])
        quat = r.as_quat()  # [x, y, z, w]
        rot.x = float(quat[0])
        rot.y = float(quat[1])
        rot.z = float(quat[2])
        rot.w = float(quat[3])


def qplan(
    robot: Robot,
    start: np.ndarray,
    goal: np.ndarray,
    validity_check_fn: Callable[[ob.State], bool],
    planner_type: ob.Planner = og.RRTConnect,
    max_step_size: float = 0.1,
    simplify: bool = False,
    bounds: Optional[ob.RealVectorBounds] = None,
    timeout: float = 5.0,
    visualize: bool = False,
):
    """
    Joint space planning for n-DOF robot
    """
    info = robot.info

    # Define the joint space (n-DOF robot)
    space = ob.RealVectorStateSpace(info.n_joints)

    # Set bounds for the state space
    if bounds is None:
        bounds = ob.RealVectorBounds(info.n_joints)
        for i in range(info.n_joints):
            bounds.setLow(i, float(robot.info.joint_limits[i][0]))
            bounds.setHigh(i, float(robot.info.joint_limits[i][1]))
    space.setBounds(bounds)

    # Create the SimpleSetup object
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validity_check_fn))

    # Create start state
    start_state = ob.State(space)
    for i in range(info.n_joints):
        start_state[i] = float(start[i])

    # Create goal state
    goal_state = ob.State(space)
    for i in range(info.n_joints):
        goal_state[i] = float(goal[i])

    ss.setStartAndGoalStates(start_state, goal_state)

    # Set up planner
    si = ss.getSpaceInformation()
    planner = planner_type(si)
    planner.setRange(max_step_size)
    ss.setPlanner(planner)

    # Solve the planning problem
    solved = ss.solve(timeout)

    if solved:
        # Simplify if requested
        if simplify:
            ss.simplifySolution()

        # Get solution path
        path = ss.getSolutionPath()

        # Extract path states
        path_states = []
        path_joints = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            path_states.append(state)
            joint_values = [state[j] for j in range(info.n_joints)]
            path_joints.append(joint_values)

        # Visualize if requested
        if visualize:
            visualize_joint_space_path(
                space,
                start_state,
                goal_state,
                path_states,
                bounds,
                robot.info.joint_limits,
            )

        return solved, np.array(path_joints)
    else:
        if visualize:
            visualize_joint_space_path(
                space, start_state, goal_state, None, bounds, robot.info.joint_limits
            )
        return solved, None


def xplan(
    robot: Robot,
    start_pose: np.ndarray,  # [x, y, z, qx, qy, qz, qw] or [x, y, z, roll, pitch, yaw]
    goal_pose: np.ndarray,
    validity_check_fn: Callable[[ob.State], bool],
    planner_type: ob.Planner = og.RRTConnect,
    max_step_size: float = 0.1,
    simplify: bool = False,
    bounds: Optional[ob.RealVectorBounds] = None,
    timeout: float = 5.0,
    visualize: bool = False,
    use_quaternions: bool = True,
):
    """
    Task space (SE3) planning for robot end-effector
    start_pose and goal_pose can be either:
    - [x, y, z, qx, qy, qz, qw] for quaternions
    - [x, y, z, roll, pitch, yaw] for Euler angles
    """

    # Define the state space
    if use_quaternions:
        space = ob.SE3StateSpace()  # Uses (x, y, z, qx, qy, qz, qw)
    else:
        # For SO3 with Euler angles, we need to use compound state space
        space = ob.CompoundStateSpace()
        space.addSubspace(ob.RealVectorStateSpace(3), 1.0)  # Position
        space.addSubspace(ob.SO3StateSpace(), 1.0)  # Orientation

    # Set bounds
    if bounds is None:
        # Default bounds: ±1 meter for position, unlimited rotation
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(-1.0)
        bounds.setHigh(1.0)

        if use_quaternions:
            space.setBounds(bounds)
        else:
            # For compound space, set bounds for position subspace
            space.getSubspace(0).setBounds(bounds)

    # Create SimpleSetup
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validity_check_fn))

    # Create start state
    start_state = ob.State(space)
    if use_quaternions:
        set_se3_state(start_state, start_pose, use_quaternions)
    else:
        set_compound_state(start_state, start_pose)

    # Create goal state
    goal_state = ob.State(space)
    if use_quaternions:
        set_se3_state(goal_state, goal_pose, use_quaternions)
    else:
        set_compound_state(goal_state, goal_pose)

    ss.setStartAndGoalStates(start_state, goal_state)

    # Set up planner
    si = ss.getSpaceInformation()
    planner = planner_type(si)
    planner.setRange(max_step_size)
    ss.setPlanner(planner)

    # Solve the planning problem
    solved = ss.solve(timeout)

    if solved:
        if simplify:
            ss.simplifySolution()

        path = ss.getSolutionPath()
        path_states = extract_path_states(path)

        if visualize:
            visualize_se3_path(
                space, start_state, goal_state, path_states, bounds, use_quaternions
            )

        # Convert path to numpy array of poses
        path_poses = []
        for state in path_states:
            if use_quaternions:
                pose = get_se3_pose(state)
            else:
                pose = get_compound_pose(state)
            path_poses.append(pose)

        return solved, np.array(path_poses)
    else:
        if visualize:
            visualize_se3_path(
                space, start_state, goal_state, None, bounds, use_quaternions
            )
        return solved, None


def set_se3_state(state, pose: np.ndarray, use_quaternions: bool = True):
    """Set SE3 state from pose array"""
    if use_quaternions:
        # pose: [x, y, z, qx, qy, qz, qw]
        state().setX(float(pose[0]))
        state().setY(float(pose[1]))
        state().setZ(float(pose[2]))
        rotation = state().rotation()
        rotation.x = float(pose[3])
        rotation.y = float(pose[4])
        rotation.z = float(pose[5])
        rotation.w = float(pose[6])
    else:
        # pose: [x, y, z, roll, pitch, yaw]
        state().setX(float(pose[0]))
        state().setY(float(pose[1]))
        state().setZ(float(pose[2]))
        state().setRoll(float(pose[3]))
        state().setPitch(float(pose[4]))
        state().setYaw(float(pose[5]))


def get_se3_pose(state):
    """Extract pose from SE3 state"""
    x = state.getX()
    y = state.getY()
    z = state.getZ()
    rot = state.rotation()
    return np.array([x, y, z, rot.x, rot.y, rot.z, rot.w])


def extract_path_states(solution_path):
    """Extract states from a solution path"""
    path_states = []
    for i in range(solution_path.getStateCount()):
        state = solution_path.getState(i)
        path_states.append(state)
    return path_states


def visualize_joint_space_path(
    space, start_state, goal_state, path_states, bounds, joint_limits
):
    """Visualize joint space path"""
    n_joints = space.getDimension()

    fig, axes = plt.subplots(1, n_joints, figsize=(4 * n_joints, 4))
    if n_joints == 1:
        axes = [axes]

    for i in range(n_joints):
        ax = axes[i]

        # Plot joint limits
        ax.axhline(
            y=joint_limits[i][0],
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Joint Limits",
        )
        ax.axhline(y=joint_limits[i][1], color="r", linestyle="--", alpha=0.5)

        # Plot start and goal
        ax.axhline(
            y=start_state[i], color="g", linestyle="-", alpha=0.7, label="Start/Goal"
        )
        ax.axhline(y=goal_state[i], color="b", linestyle="-", alpha=0.7)

        # Plot path if available
        if path_states:
            joint_values = [state[i] for state in path_states]
            ax.plot(
                range(len(joint_values)), joint_values, "k-", linewidth=2, label="Path"
            )
            ax.plot(
                range(len(joint_values)), joint_values, "ko", markersize=3, alpha=0.6
            )

        ax.set_xlabel("Path Step")
        ax.set_ylabel(f"Joint {i + 1} (rad)")
        ax.set_title(f"Joint {i + 1} Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


from ompl.base._base import State
from ompl.util._util import vectorDouble


def visualize_se3_path(
    space,
    start_state: State,
    goal_state: State,
    path_states,
    bounds,
    use_quaternions=True,
):
    """Visualize SE3 path without interrupting execution."""
    fig = plt.figure(figsize=(15, 5))

    # 3D plot
    ax1 = fig.add_subplot(131, projection="3d")

    # Plot start and goal
    if use_quaternions:
        start_pos = [start_state.getX(), start_state.getY(), start_state.getZ()]
        # start_pos = [start_state.getX(), start_state.getY(), start_state.getZ()]
        goal_pos = [goal_state.getX(), goal_state.getY(), goal_state.getZ()]
    else:
        start_pos = [start_state()[0][0], start_state()[0][1], start_state()[0][2]]
        goal_pos = [goal_state()[0][0], goal_state()[0][1], goal_state()[0][2]]

    ax1.plot(
        [start_pos[0]],
        [start_pos[1]],
        [start_pos[2]],
        "go",
        markersize=10,
        label="Start",
    )
    ax1.plot(
        [goal_pos[0]], [goal_pos[1]], [goal_pos[2]], "ro", markersize=10, label="Goal"
    )

    # Plot path
    if path_states:
        path_x, path_y, path_z = [], [], []
        for state in path_states:
            if use_quaternions:
                path_x.append(state.getX())
                path_y.append(state.getY())
                path_z.append(state.getZ())
            else:
                path_x.append(state()[0][0])
                path_y.append(state()[0][1])
                path_z.append(state()[0][2])

        ax1.plot(path_x, path_y, path_z, "b-", linewidth=2, label="Path")
        ax1.plot(path_x, path_y, path_z, "bo", markersize=3, alpha=0.6)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("SE3 Path")
    ax1.legend()

    # Position plots
    ax2 = fig.add_subplot(132)
    if path_states:
        steps = range(len(path_states))
        ax2.plot(steps, path_x, "r-", label="X")
        ax2.plot(steps, path_y, "g-", label="Y")
        ax2.plot(steps, path_z, "b-", label="Z")
        ax2.set_xlabel("Path Step")
        ax2.set_ylabel("Position")
        ax2.set_title("Position vs Step")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Orientation plot (simplified)
    ax3 = fig.add_subplot(133)
    if path_states and use_quaternions:
        # Plot quaternion components
        qx = [state.rotation().x for state in path_states]
        qy = [state.rotation().y for state in path_states]
        qz = [state.rotation().z for state in path_states]
        qw = [state.rotation().w for state in path_states]

        ax3.plot(steps, qx, "r-", label="qx")
        ax3.plot(steps, qy, "g-", label="qy")
        ax3.plot(steps, qz, "b-", label="qz")
        ax3.plot(steps, qw, "k-", label="qw")
        ax3.set_xlabel("Path Step")
        ax3.set_ylabel("Quaternion")
        ax3.set_title("Orientation vs Step")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage functions
def qplan_example():
    """Example of joint space planning"""

    class MockRobot:
        def __init__(self):
            self.info = type("Info", (), {})()
            self.info.n_joints = 2
            self.info.joint_limits = np.array([[-1, 1], [-1, 1]])

    robot = MockRobot()

    def validity_check_fn(state):
        # Simple validity checker - allow all states within joint limits
        for i in range(robot.info.n_joints):
            if (
                state[i] < robot.info.joint_limits[i][0]
                or state[i] > robot.info.joint_limits[i][1]
            ):
                return False
        return True

    start = np.array([-0.5, -0.5])
    goal = np.array([0.5, 0.5])

    solved, path = qplan(
        robot=robot,
        start=start,
        goal=goal,
        validity_check_fn=validity_check_fn,
        visualize=True,
        timeout=5.0,
    )

    return solved, path


def xplan_example():
    """Example of task space planning"""

    class MockRobot:
        def __init__(self):
            pass

    robot = MockRobot()

    def validity_check_fn(state):
        # Simple validity checker - avoid region around origin
        x, y, z = state.getX(), state.getY(), state.getZ()
        # Avoid sphere around origin
        return (x**2 + y**2 + z**2) > 0.25

    # Start and goal poses [x, y, z, qx, qy, qz, qw]
    start_pose = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Identity rotation
    goal_pose = np.array([-0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Identity rotation

    solved, path = xplan(
        robot=robot,
        start_pose=start_pose,
        goal_pose=goal_pose,
        validity_check_fn=validity_check_fn,
        visualize=True,
        timeout=5.0,
    )

    return solved, path


if __name__ == "__main__":
    print("Testing joint space planning...")
    qplan_example()

    print("\nTesting task space planning...")
    xplan_example()
