from typing import Callable, Optional, Tuple

import mink
import numpy as np
import ompl
import ompl.base as ob
import ompl.geometric as og
import spatialmath as sm
from ompl.base._base import Planner, State

from base.robot import Robot


def xplan(): ...


def qplan(): ...


def mk_plan(
    robot: Robot,
    start: np.ndarray,
    goal: np.ndarray,
    validity_check_fn: Callable[[State], bool],
    planner_type: ob.Planner,
    max_step_size: float = 0.1,
    simplify: bool = False,
) -> Tuple[bool, np.ndarray]:
    """
    Plans a joint trajectory for the robot.

    :param robot_info: RobotInfo object containing details about the robot (e.g., number of joints, joint limits).
    :param start: numpy array representing the start joint configuration.
    :param end: numpy array representing the goal joint configuration.
    :param validity_check_fn: Function to check if a given state is valid.
    :return: numpy array representing the planned path as a sequence of joint configurations, or None if no path is found.
    """

    info = robot.info

    # Define the joint space (n-DOF robot)
    space = ob.RealVectorStateSpace(info.n_joints)

    # Set bounds for the state space
    bounds = ob.RealVectorBounds(info.n_joints)
    for i in range(robot.n_joints):
        bounds.setLow(i, robot.joint_limits.T[i][0])
        bounds.setHigh(i, robot.joint_limits.T[i][1])
    space.setBounds(bounds)

    # Create the SimpleSetup object
    ss = og.SimpleSetup(space)

    # Set the state validity checker
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validity_check_fn))

    # Define the start state
    start_state = ob.State(space)
    for i in range(robot.n_joints):
        start_state[i] = start[i]
    ss.setStartState(start_state)

    # Define the goal state
    goal_state = ob.State(space)
    for i in range(robot.n_joints):
        goal_state[i] = goal[i]
    ss.setGoalState(goal_state)

    # Create the planner (using RRTConnect as default, can be extended to take as a parameter)
    planner: Planner = planner_type(ss.getSpaceInformation())

    planner.setRange(max_step_size)

    ss.setPlanner(planner)

    # Solve the planning problem
    solved = ss.solve(1.0)  # 1-second timeout

    if solved:
        # Simplify and retrieve the solution
        if simplify:
            ss.simplifySolution()
        path = ss.getSolutionPath()

        # Extract joint values from the path
        return solved, np.array(
            [[state[i] for i in range(robot.n_joints)] for state in path.getStates()]
        )
    else:
        # No solution found
        return solved, None


# if __name__ == "__main__":
#     success, plan = mk_plan(
#         robot_info=ur5e.info,
#         start=start,
#         goal=goal,
#         planner_type=og.RRTConnect,
#         validity_check_fn=is_free,
#         max_step_size=0.02,
#         # max_step_size=0.01,
#     )
