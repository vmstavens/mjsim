from typing import List

import numpy as np
from robots.base_robot import BaseRobot
from scipy.linalg import block_diag
from scipy.optimize import OptimizeResult, linprog
from utils.math import skew_symmetric
from utils.mj import JointType, ObjType, get_pose, get_type
from utils.physics import ContactModelType


def compute_Ri_bar(contact_frame: np.ndarray) -> np.ndarray:
    assert contact_frame.shape == (3, 3), (
        f"Contact frame should be (3,3), but got {contact_frame.shape}"
    )

    zero_3x3 = np.zeros(shape=(3, 3))
    Ri_bar = np.block([[contact_frame, zero_3x3], [zero_3x3, contact_frame]])

    return Ri_bar


def compute_Pi(contact_point: np.ndarray, p_world_object: np.ndarray) -> np.ndarray:
    _check_object_position(p_world_object)
    _check_contact_point(contact_point)

    I_3x3 = np.eye(3)
    zero_3x3 = np.zeros(shape=(3, 3))
    Si = skew_symmetric(contact_point - p_world_object)
    Pi = np.block([[I_3x3, zero_3x3], [Si, I_3x3]])
    return Pi


def compute_Gi(
    contact_point: np.ndarray, contact_frame: np.ndarray, p_world_object: np.ndarray
) -> np.ndarray:
    Pi = compute_Pi(contact_point=contact_point, p_world_object=p_world_object)
    Ri_bar = compute_Ri_bar(contact_frame)

    Gi_T = Ri_bar.T @ Pi.T
    Gi = Gi_T.T

    return Gi


def compute_G(
    contact_points: np.ndarray, contact_frames: np.ndarray, p_world_object: np.ndarray
) -> np.ndarray:
    if not isinstance(contact_points, np.ndarray):
        contact_points = np.array(contact_points)
    if not isinstance(p_world_object, np.ndarray):
        p_world_object = np.array(p_world_object)
    if not isinstance(contact_frames, np.ndarray):
        contact_frames = np.array(contact_frames)

    Gis = [
        compute_Gi(
            contact_point=contact_points[i],
            contact_frame=contact_frames[i],
            p_world_object=p_world_object,
        )
        for i in range(len(contact_points))
    ]

    return np.hstack([Gi for Gi in Gis])


def compute_Zi(fingers: List[BaseRobot], contact_point: np.ndarray) -> np.ndarray:
    model = fingers[0].model
    data = fingers[0].data

    joint_ids = [jid for finger in fingers for jid in finger.info.joint_ids]

    joint_frames = [get_pose(model, data, jid, ObjType.JOINT) for jid in joint_ids]
    zero_3x1 = np.zeros(3)

    Zi = []

    for jid, joint_frame in zip(joint_ids, joint_frames):
        dj = np.zeros(3)
        kj = np.zeros(3)
        joint_type = get_type(model, jid, ObjType.JOINT)
        if joint_type is JointType.HINGE:
            S = skew_symmetric(contact_point - joint_frame.t)
            zj = joint_frame.R[:, 2]
            dj = S.T @ zj
            kj = zj
        elif joint_type is JointType.SLIDE:
            zj = joint_frame.R[:, 2]
            dj = zj
            kj = zero_3x1
        dj_kj = np.hstack([dj, kj])
        Zi.append(dj_kj)
    return np.vstack(Zi).T


def compute_J(
    fingers: List[BaseRobot], contact_frames: np.ndarray, contact_points: np.ndarray
) -> np.ndarray:
    J = []
    for contact_frame, contact_point in zip(contact_frames, contact_points):
        Ri_bar = compute_Ri_bar(contact_frame)
        Zi = compute_Zi(fingers, contact_point)
        Ji = Ri_bar.T @ Zi
        J.append(Ji)
    return np.vstack(J)


def rank_condition(G: np.ndarray, nv: int = 6) -> bool:
    return np.linalg.matrix_rank(G) == nv


def frictional_form_closure_condition(G: np.ndarray, F_A: np.ndarray, nc: int) -> bool:
    assert G.shape == (6, 6 * nc), (
        f"Grasping matrix G should be of shape (6, 6*nc), but got {G.shape}"
    )
    # assert F_A.shape == ()
    result: OptimizeResult = solve_lp2(G, F_A, nc)

    return result.success


def control_of_internal_force_condition(
    G: np.ndarray, J: np.ndarray, E: np.ndarray
) -> bool:
    result = solve_lp3(G, J, E)
    return result.success


def in_force_closure(
    fingers: List[BaseRobot],
    contact_points: np.ndarray,
    contact_frames: np.ndarray,
    p_world_object: np.ndarray,
    mu: np.ndarray,
    nc: int,
    nv: int = 6,
    contact_width: float = 0.001,
    contact_length: float = 0.001,
) -> bool:
    G = compute_G(
        contact_points=contact_points,
        contact_frames=contact_frames,
        p_world_object=p_world_object,
    )
    F_A = compute_F(
        mu=mu,
        contact_frames=contact_frames,
        contact_length=contact_length,
        contact_width=contact_width,
    )
    J = compute_J(
        fingers=fingers, contact_frames=contact_frames, contact_points=contact_points
    )
    E = compute_E(nc=nc)
    condition_1 = rank_condition(G, nv)
    condition_2 = frictional_form_closure_condition(G, F_A, nc)
    condition_3 = control_of_internal_force_condition(G, J, E)

    print(f"{G.shape=}")
    print(f"{F_A.shape=}")
    print(f"{J.shape=}")
    print(f"{E.shape=}")

    print(f"is in force closure {condition_1 and condition_2 and condition_3}")
    print(f"\t{condition_1=}")
    print(f"\t{condition_2=}")
    print(f"\t{condition_3=}")

    return condition_1 and condition_2 and condition_3


def compute_Si(
    mu: float, b: float, ng: int, contact_type: ContactModelType = ContactModelType.SOFT
) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, ng, endpoint=False)  # Friction cone angles
    if contact_type is ContactModelType.SOFT:
        S = np.zeros((4, ng + 2))
        S[0, :] = 1  # Normal forces
        S[1, :ng] = mu * np.cos(theta)  # Tangential forces x
        S[2, :ng] = mu * np.sin(theta)  # Tangential forces y
        S[3, ng] = b * mu  # Torsional friction +
        S[3, ng + 1] = -b * mu  # Torsional friction -
        return S
    elif contact_type is ContactModelType.HARD:
        S = np.zeros((3, ng))
        S[0, :] = 1  # Normal forces
        S[1, :ng] = mu * np.cos(theta)  # Tangential forces x
        S[2, :ng] = mu * np.sin(theta)  # Tangential forces y
        return S
    elif contact_type is ContactModelType.PwoF:
        raise NotImplementedError("Point contact without friction not yet supported.")
    else:
        raise ValueError("Not recognized contact type")


def compute_S(
    mu: np.ndarray,
    b: np.ndarray,
    ng: int,
    contact_type: ContactModelType = ContactModelType.SOFT,
) -> np.ndarray:
    mu = np.asarray(mu)
    b = np.asarray(b)
    if mu.shape != b.shape:
        raise ValueError("mu and b must have matching shapes per contact")

    s_blocks = [
        compute_Si(mu[i], b[i], ng, contact_type=contact_type) for i in range(mu.shape[0])
    ]
    return block_diag(*s_blocks)


def compute_Hi(contact_model: ContactModelType = ContactModelType.SOFT) -> np.ndarray:
    HiF = compute_HiF(contact_model)
    HiM = compute_HiM(contact_model)
    zeros_3x3 = np.zeros((3, 3))
    Hi = np.block([[HiF, zeros_3x3], [zeros_3x3, HiM]])
    return Hi


def compute_HiF(contact_model: ContactModelType = ContactModelType.SOFT) -> np.ndarray:
    if contact_model is ContactModelType.SOFT:
        return np.eye(3)
    if contact_model is ContactModelType.HARD:
        return np.zeros((3, 3))
    if contact_model is ContactModelType.PwoF:
        return np.zeros((3, 3))


def compute_HiM(contact_model: ContactModelType = ContactModelType.SOFT) -> np.ndarray:
    if contact_model is ContactModelType.SOFT:
        return np.eye(3)
    if contact_model is ContactModelType.HARD:
        return np.zeros((3, 3))
    if contact_model is ContactModelType.PwoF:
        return np.zeros((3, 3))


def compute_E(
    nc: int, contact_model: ContactModelType = ContactModelType.SOFT
) -> np.ndarray:
    Ei = []
    H = [compute_Hi(contact_model) for i in range(nc)]
    for i in range(nc):
        ei = H[i][0, :]
        Ei.append(ei.T)

    def build_block_matrix(vectors: list) -> np.ndarray:
        n = len(vectors[0])
        # Initialize the output matrix
        result = np.zeros((len(vectors) * n, len(vectors)))

        # Fill the diagonal blocks
        for i, vec in enumerate(vectors):
            result[i * len(vec) : (i + 1) * len(vec), i] = vec
        return result

    return build_block_matrix(Ei)


def compute_e(
    nc: int, contact_model: ContactModelType = ContactModelType.SOFT
) -> np.ndarray:
    H = [compute_Hi(contact_model) for i in range(nc)]
    Ei = []
    for i in range(nc):
        ei = H[i][0, :]
        Ei.append(ei)
    return np.hstack(Ei)


def solve_lp2(G: np.ndarray, F_A: np.ndarray, n_c: int):
    """
    Solves the LP2 linear program for approximate force closure.

    Parameters:
    G (np.ndarray): Grasp matrix (6 x nλ).
    F_A (np.ndarray): Friction cone approximation matrix (n_constraints x nλ).
    n_c (int): Number of contact points.

    Returns:
    dict: Solution dictionary containing `success`, `d_star`, and `lambda`.
    """

    # Dimensions
    n_lambda = G.shape[1]

    # print(f"{G.shape=}")
    # print(f"{F_A.shape=}")
    # print(f"{n_c=}")
    # print(f"{n_lambda=}")

    # Objective: maximize d (equivalent to minimizing -d)
    c = np.zeros(n_lambda + 1)
    c[-1] = -1  # Coefficient for "d" in the objective

    # Equality constraint: Gλ = 0
    A_eq = np.hstack([G, np.zeros((G.shape[0], 1))])
    b_eq = np.zeros(G.shape[0])

    # a_ub_1 = np.hstack([-F_A, np.zeros((F_A.shape[0], 1))])
    # a_ub_2 = np.zeros((1, n_lambda + 1))
    # a_ub_3 = np.hstack([np.ones((1, n_lambda)), [[-1]]])
    # print(f"{a_ub_1.shape=}")
    # print(f"{a_ub_2.shape=}")
    # print(f"{a_ub_3.shape=}")

    # Inequality constraints: F_A λ >= 0, 1d >= 0, and e^T λ <= n_c
    A_ub = np.vstack(
        [
            np.hstack([-F_A, np.zeros((F_A.shape[0], 1))]),  # -F_A λ <= 0 -> F_A λ >= 0
            np.zeros((1, n_lambda + 1)),  # 1d >= 0 -> always satisfied for d >= 0
            np.hstack([np.ones((1, n_lambda)), [[-1]]]),  # e^T λ - d <= n_c
        ]
    )
    b_ub = np.hstack(
        [
            np.zeros(F_A.shape[0]),  # Corresponding to F_A λ >= 0
            [0],  # Corresponding to d >= 0
            [n_c],  # Corresponding to e^T λ - d <= n_c
        ]
    )

    # Solve the linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))
    return result


def solve_lp3(G: np.ndarray, J: np.ndarray, E: np.ndarray) -> OptimizeResult:
    """
    Solves LP3 to verify force closure condition.

    Parameters:
    - G: (m x n) Grasp matrix
    - J: (n x k) Jacobian matrix
    - E: (p x n) Matrix for the inequality constraint

    Returns:
    - result: Optimization result containing the status and the solution.
    """

    # Dimensions
    n = G.shape[1]  # Length of the vector d

    # Objective function: maximize d -> equivalent to minimize -d
    c = np.zeros(n)
    c[-1] = -1  # Maximize the last component, d

    # Equality constraints: G * d = 0 and J.T * d = 0
    A_eq = np.vstack([G, J.T])  # Combine G and J^T
    b_eq = np.zeros(A_eq.shape[0])

    # Inequality constraint: E @ d >= 1 -> equivalent to -E.T @ d <= -1
    A_ineq = -E.T  # Transpose E for proper dimensionality
    b_ineq = -np.ones(A_ineq.shape[0])  # Match the number of rows in A_ineq

    # Bounds: d >= 0
    bounds = [(0, None) for _ in range(n)]

    # print(f"{G.shape=}")
    # print(f"{J.shape=}")
    # print(f"{E.shape=}")
    # print(f"{A_eq.shape=}")
    # print(f"{b_eq.shape=}")
    # print(f"{A_ineq.shape=}")
    # print(f"{b_ineq.shape=}")
    # print(f"{c.shape=}")

    # Solve the linear program
    result = linprog(
        c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    # Check if force closure condition is satisfied
    # if result.success and result.fun < 0:
    #     print("The grasp does NOT have force closure.")
    # else:
    #     print("The grasp has force closure.")

    return result


def compute_Fi(
    mu: float,
    # contact_frame: np.ndarray,
    Ri_bar: np.ndarray,
    contact_length: float = 0.001,
    contact_width: float = 0.001,
):
    """
    Matrix `F` of friction inequalities in world frame.

    This matrix describes the linearized Coulomb friction model (in the
    fixed contact mode) by:

    .. math::

        F w \\leq 0

    where `w` is the contact wrench at the contact point (``self.p``) in
    the world frame. See [Caron15]_ for the derivation of the formula for
    `F`.
    sources:
        https://github.com/stephane-caron/pymanoid/blob/master/pymanoid/contact.py [code]
        https://arxiv.org/pdf/1501.04719 [paper]
        https://hal.science/hal-02108449/document [more paper]
        https://scaron.info/robotics/wrench-friction-cones.html [blog post]
    """
    X, Y = contact_length / 2, contact_width / 2
    mu = mu / np.sqrt(2)  # inner approximation
    local_cone = np.array(
        [
            # fx fy             fz taux tauy tauz
            [-1, 0, -mu, 0, 0, 0],
            [+1, 0, -mu, 0, 0, 0],
            [0, -1, -mu, 0, 0, 0],
            [0, +1, -mu, 0, 0, 0],
            [0, 0, -Y, -1, 0, 0],
            [0, 0, -Y, +1, 0, 0],
            [0, 0, -X, 0, -1, 0],
            [0, 0, -X, 0, +1, 0],
            [-Y, -X, -(X + Y) * mu, +mu, +mu, -1],
            [-Y, +X, -(X + Y) * mu, +mu, -mu, -1],
            [+Y, -X, -(X + Y) * mu, -mu, +mu, -1],
            [+Y, +X, -(X + Y) * mu, -mu, -mu, -1],
            [+Y, +X, -(X + Y) * mu, +mu, +mu, +1],
            [+Y, -X, -(X + Y) * mu, +mu, -mu, +1],
            [-Y, +X, -(X + Y) * mu, -mu, +mu, +1],
            [-Y, -X, -(X + Y) * mu, -mu, -mu, +1],
        ]
    )
    return np.dot(local_cone, Ri_bar.T)
    # return np.dot(local_cone, block_diag(Ri_bar.T, Ri_bar.T))
    # return np.dot(local_cone, block_diag(contact_frame.T, contact_frame.T))


def compute_F(
    mu: np.ndarray,
    contact_frames: np.ndarray,
    contact_length: float = 0.001,
    contact_width: float = 0.001,
) -> np.ndarray:
    """ """
    if len(mu) != len(contact_frames):
        raise ValueError("mu and contact_frames must have the same length.")

    F_blocks = []
    for i, contact_frame in enumerate(contact_frames):
        Ri_bar = compute_Ri_bar(contact_frame)
        Fi = compute_Fi(
            mu[i], Ri_bar, contact_length=contact_length, contact_width=contact_width
        )
        F_blocks.append(Fi)

    return block_diag(*F_blocks)


def _check_contact_point(contact_point: np.ndarray):
    assert contact_point.shape == (3,), (
        f"Contact point should be (3,), but got {contact_point.shape}"
    )


def _check_object_position(p_world_object: np.ndarray):
    assert p_world_object.shape == (3,), (
        f"Object position should be (3,), but got {p_world_object.shape}"
    )
