from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import mink
import numpy as np
import ompl
import ompl.geometric as og
import spatialmath as sm
from ompl import base as ob

# from ompl import geometric as og
# from ompl.base._base import Planner, State
from mjsim.base.robot import Robot
from mjsim.utils.ompl import qplan


def isStateValid(state):
    return state.getX() < 0.6


def visualize_se2_path(
    space, start_state=None, goal_state=None, path_states=None, bounds=None
):
    """Visualize the SE2 path and environment"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Get the bounds
    if bounds is None:
        bounds = space.getBounds()
    x_min, x_max = bounds.low[0], bounds.high[0]
    y_min, y_max = bounds.low[1], bounds.high[1]

    # Plot 1: Environment overview with invalid region
    # Create a grid to visualize valid/invalid regions
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Check validity for each point (simplified - only checking x coordinate)
    valid_region = X < 0.6

    # Plot valid/invalid regions
    ax1.imshow(
        valid_region,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        alpha=0.3,
        cmap="RdYlGn",
    )
    ax1.axvline(
        x=0.6,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Obstacle boundary (x=0.6)",
    )

    # Plot start and goal if available
    if start_state is not None:
        ax1.plot(
            start_state.getX(), start_state.getY(), "go", markersize=10, label="Start"
        )
        # Draw start orientation
        draw_pose(
            ax1, start_state.getX(), start_state.getY(), start_state.getYaw(), "green"
        )

    if goal_state is not None:
        ax1.plot(
            goal_state.getX(), goal_state.getY(), "ro", markersize=10, label="Goal"
        )
        # Draw goal orientation
        draw_pose(ax1, goal_state.getX(), goal_state.getY(), goal_state.getYaw(), "red")

    # Plot path if available
    if path_states:
        path_x = [state.getX() for state in path_states]
        path_y = [state.getY() for state in path_states]
        ax1.plot(path_x, path_y, "b-", linewidth=2, label="Path")
        ax1.plot(path_x, path_y, "bo", markersize=3, alpha=0.6)

        # Draw orientation along the path (every few points)
        for i in range(0, len(path_states), max(1, len(path_states) // 5)):
            state = path_states[i]
            draw_pose(
                ax1, state.getX(), state.getY(), state.getYaw(), "blue", scale=0.1
            )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("SE2 Planning - Environment")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Plot 2: Path details and orientation
    if path_states:
        # Create subplots for position and orientation
        ax2_sub1 = ax2  # For single subplot in second column

        # X and Y position over path length
        path_length = len(path_states)
        steps = range(path_length)

        ax2_sub1.plot(
            steps,
            [state.getX() for state in path_states],
            "r-",
            linewidth=2,
            label="X position",
        )
        ax2_sub1.plot(
            steps,
            [state.getY() for state in path_states],
            "g-",
            linewidth=2,
            label="Y position",
        )
        ax2_sub1.axhline(
            y=0.6, color="black", linestyle="--", alpha=0.5, label="Obstacle boundary"
        )
        ax2_sub1.set_xlabel("Path Step")
        ax2_sub1.set_ylabel("Position")
        ax2_sub1.set_title("Position vs Path Step")
        ax2_sub1.legend()
        ax2_sub1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Also print path information
    if path_states:
        print(f"Path found with {len(path_states)} states")
        print(
            f"Start: ({path_states[0].getX():.3f}, {path_states[0].getY():.3f}, {path_states[0].getYaw():.3f})"
        )
        print(
            f"Goal:  ({path_states[-1].getX():.3f}, {path_states[-1].getY():.3f}, {path_states[-1].getYaw():.3f})"
        )


def draw_pose(ax, x, y, yaw, color, scale=0.2):
    """Draw a pose (x, y, yaw) as an arrow"""
    dx = scale * np.cos(yaw)
    dy = scale * np.sin(yaw)
    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=scale * 0.3,
        head_length=scale * 0.2,
        fc=color,
        ec=color,
        alpha=0.7,
    )


def extract_path_states(solution_path):
    """Extract states from a solution path"""
    path_states = []
    for i in range(solution_path.getStateCount()):
        state = solution_path.getState(i)
        path_states.append(state)
    return path_states


def plan():
    # create an SE2 state space
    space = ob.SE2StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-1)
    bounds.setHigh(1)
    space.setBounds(bounds)

    # create a simple setup object
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    # Create start state
    start = ob.State(space)
    start().setX(0.5)
    start().setY(0.2)
    start().setYaw(0.0)

    # Create goal state
    goal = ob.State(space)
    goal().setX(-0.5)
    goal().setY(-0.3)
    goal().setYaw(np.pi / 4)

    ss.setStartAndGoalStates(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(1.0)

    if solved:
        # get the solution path
        path = ss.getSolutionPath()

        # Extract path states for visualization
        path_states = extract_path_states(path)

        # Print the path
        print("Solution path found!")
        print(f"Path length: {path.length()}")
        print(f"Number of states: {path.getStateCount()}")

        # Visualize the path - pass the individual states
        visualize_se2_path(space, start(), goal(), path_states, bounds)

        return path_states
    else:
        print("No solution found")
        # Visualize just the environment with start and goal
        visualize_se2_path(space, start(), goal(), None, bounds)
        return None


if __name__ == "__main__":
    path = plan()
    print("Planning completed!")

# def isStateValid(state):
#     # "state" is of type SE2StateInternal, so we don't need to use the "()"
#     # operator.
#     #
#     # Some arbitrary condition on the state (note that thanks to
#     # dynamic type checking we can just call getX() and do not need
#     # to convert state to an SE2State.)
#     return state.getX() < 0.6


# def plan():
#     # create an SE2 state space
#     space = ob.SE2StateSpace()

#     # set lower and upper bounds
#     bounds = ob.RealVectorBounds(2)
#     bounds.setLow(-1)
#     bounds.setHigh(1)
#     space.setBounds(bounds)

#     # create a simple setup object
#     ss = og.SimpleSetup(space)
#     ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

#     start = ob.State(space)
#     # we can pick a random start state...
#     start.random()
#     # ... or set specific values
#     start().setX(0.5)

#     goal = ob.State(space)
#     # we can pick a random goal state...
#     goal.random()
#     # ... or set specific values
#     goal().setX(-0.5)

#     ss.setStartAndGoalStates(start, goal)

#     # this will automatically choose a default planner with
#     # default parameters
#     solved = ss.solve(1.0)

#     if solved:
#         # try to shorten the path
#         # ss.simplifySolution()
#         # print the simplified path
#         print(ss.getSolutionPath())


# if __name__ == "__main__":
#     plan()

# ---------------------------------------------------

# def qplan_2d_example():
#     """
#     Simple 2D example with a 2-DOF planar robot arm
#     """

#     # Mock robot info for a 2-DOF robot
#     class RobotInfo:
#         def __init__(self):
#             self.n_joints = 2
#             # Joint limits: [min, max] for each joint
#             self.joint_limits = np.array(
#                 [
#                     [-1, 1],  # Joint 1: full rotation
#                     [-1, 1],  # Joint 2: full rotation
#                 ]
#             )

#     class MockRobot:
#         def __init__(self):
#             self.info = RobotInfo()

#     # Create robot
#     robot = MockRobot()

#     # Define start and goal configurations
#     start = np.array([-0.5, -0.5])  # Both joints at 0 radians
#     goal = np.array([0.5, 0.5])  # Joint1: 90°, Joint2: 45°
#     # goal = np.array([np.pi / 2, np.pi / 4])  # Joint1: 90°, Joint2: 45°

#     # Simple validity checker - no obstacles in this example
#     def validity_check_fn(state):
#         """
#         Simple validity checker that allows all states within joint limits
#         Add obstacle checking logic here for more complex scenarios
#         """
#         q1 = state[0]
#         q2 = state[1]

#         # Check joint limits
#         if (
#             q1 < robot.info.joint_limits[0][0]
#             or q1 > robot.info.joint_limits[0][1]
#             or q2 < robot.info.joint_limits[1][0]
#             or q2 > robot.info.joint_limits[1][1]
#         ):
#             return False

#         # Optional: Add obstacle checking here
#         # For example, check if the end effector collides with obstacles

#         return True

#     # Run the planner
#     solved, path = qplan(
#         robot=robot,
#         start=start,
#         goal=goal,
#         validity_check_fn=validity_check_fn,
#         planner_type=og.RRTConnect,
#         max_step_size=0.01,  # Maximum step size in joint space
#         simplify=True,  # Simplify the path
#         timeout=5.0,  # 5 second timeout
#     )

#     if solved:
#         print("Path found!")
#         print(f"Path length: {len(path)} waypoints")
#         print(f"Start: {path[0]}")
#         print(f"Goal: {path[-1]}")

#         # Visualize the path in joint space
#         visualize_joint_space_path(path, start, goal)

#         return path
#     else:
#         print("No path found within timeout")
#         return None


# def visualize_joint_space_path(path, start, goal):
#     """Visualize the path in joint space"""
#     plt.figure(figsize=(10, 4))

#     # Plot joint 1 trajectory
#     plt.subplot(1, 2, 1)
#     q1 = [point[0] for point in path]
#     steps = range(len(path))
#     plt.plot(steps, q1, "b-", linewidth=2, label="Joint 1")
#     plt.axhline(y=start[0], color="g", linestyle="--", alpha=0.7, label="Start")
#     plt.axhline(y=goal[0], color="r", linestyle="--", alpha=0.7, label="Goal")
#     plt.xlabel("Waypoint")
#     plt.ylabel("Joint 1 (rad)")
#     plt.title("Joint 1 Trajectory")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     # Plot joint 2 trajectory
#     plt.subplot(1, 2, 2)
#     q2 = [point[1] for point in path]
#     plt.plot(steps, q2, "orange", linewidth=2, label="Joint 2")
#     plt.axhline(y=start[1], color="g", linestyle="--", alpha=0.7, label="Start")
#     plt.axhline(y=goal[1], color="r", linestyle="--", alpha=0.7, label="Goal")
#     plt.xlabel("Waypoint")
#     plt.ylabel("Joint 2 (rad)")
#     plt.title("Joint 2 Trajectory")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.show()


# path = qplan_2d_example()

# start = np.array([0.0, 0.0])  # Both joints at 0 radians
# goal = np.array([np.pi / 2, np.pi / 4])  # Joint1: 90°, Joint2: 45°

# visualize_joint_space_path(path, start, goal)


# print("im done")
