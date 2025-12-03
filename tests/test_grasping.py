import sys
import types
from enum import Enum

import numpy as np
import pytest

# Provide lightweight stand-ins for optional modules so we can import grasping in isolation.
robots_mod = sys.modules.get("robots", types.ModuleType("robots"))
if "robots" not in sys.modules:
    sys.modules["robots"] = robots_mod
base_robot_mod = sys.modules.get("robots.base_robot", types.ModuleType("robots.base_robot"))
if "robots.base_robot" not in sys.modules:
    sys.modules["robots.base_robot"] = base_robot_mod

if not hasattr(base_robot_mod, "BaseRobot"):
    class _BaseRobot:
        pass
    base_robot_mod.BaseRobot = _BaseRobot
    robots_mod.base_robot = base_robot_mod

utils_pkg = sys.modules.get("utils", types.ModuleType("utils"))
if "utils" not in sys.modules:
    sys.modules["utils"] = utils_pkg

if "utils.math" not in sys.modules:
    math_mod = types.ModuleType("utils.math")

    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ]
        )

    math_mod.skew_symmetric = _skew_symmetric
    utils_pkg.math = math_mod
    sys.modules["utils.math"] = math_mod

if "utils.mj" not in sys.modules:
    mj_mod = types.ModuleType("utils.mj")

    class _JointType(Enum):
        HINGE = 0
        SLIDE = 1

    class _ObjType(Enum):
        JOINT = 0
        BODY = 1
        GEOM = 2

    def _pose_stub(model, data, jid, obj_type):
        return types.SimpleNamespace(R=np.eye(3), t=np.zeros(3))

    def _get_type_stub(model, jid, obj_type):
        return _JointType.HINGE

    mj_mod.JointType = _JointType
    mj_mod.ObjType = _ObjType
    mj_mod.get_pose = _pose_stub
    mj_mod.get_type = _get_type_stub
    utils_pkg.mj = mj_mod
    sys.modules["utils.mj"] = mj_mod

if "utils.physics" not in sys.modules:
    physics_mod = types.ModuleType("utils.physics")

    class _ContactModelType(Enum):
        SOFT = 0
        HARD = 1
        PwoF = 2

    physics_mod.ContactModelType = _ContactModelType
    utils_pkg.physics = physics_mod
    sys.modules["utils.physics"] = physics_mod

from mjsim.utils import grasping


def test_compute_G_returns_stacked_identity():
    """With coincident contacts and identity frames, G should stack identity blocks."""
    contact_points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    contact_frames = np.stack([np.eye(3), np.eye(3)])
    p_world_object = np.zeros(3)

    G = grasping.compute_G(contact_points, contact_frames, p_world_object)

    expected = np.hstack([np.eye(6), np.eye(6)])
    assert G.shape == (6, 12)
    assert np.allclose(G, expected)


def test_compute_F_builds_block_diagonal_cones():
    mu = np.array([0.5, 0.5])
    contact_frames = np.stack([np.eye(3), np.eye(3)])
    contact_length = 0.002
    contact_width = 0.001

    Ri_bar = grasping.compute_Ri_bar(np.eye(3))
    single_F = grasping.compute_Fi(mu[0], Ri_bar, contact_length, contact_width)

    F = grasping.compute_F(mu, contact_frames, contact_length, contact_width)

    assert F.shape == (single_F.shape[0] * 2, single_F.shape[1] * 2)
    # Top-left block matches single contact approximation
    assert np.allclose(F[: single_F.shape[0], : single_F.shape[1]], single_F)
    # Off-diagonal block remains zero from block-diagonal construction
    assert np.allclose(
        F[: single_F.shape[0], single_F.shape[1] :], np.zeros_like(single_F)
    )


def test_compute_S_stack_shapes_and_blocks():
    mu = np.array([0.4, 0.2])
    b = np.array([0.01, 0.02])
    ng = 4

    single_S = grasping.compute_Si(mu[0], b[0], ng)
    S = grasping.compute_S(mu, b, ng)

    assert S.shape == (single_S.shape[0] * 2, single_S.shape[1] * 2)
    assert np.allclose(S[: single_S.shape[0], : single_S.shape[1]], single_S)
    assert np.allclose(
        S[: single_S.shape[0], single_S.shape[1] :], np.zeros_like(single_S)
    )


def test_in_force_closure_returns_true_with_stubbed_lp(monkeypatch: pytest.MonkeyPatch):
    # Stub linear programs to report success regardless of inputs
    monkeypatch.setattr(grasping, "solve_lp2", lambda *_args, **_kwargs: type("R", (), {"success": True})())
    monkeypatch.setattr(grasping, "solve_lp3", lambda *_args, **_kwargs: type("R", (), {"success": True})())
    # Simplify Jacobian computation so dimensions align deterministically
    monkeypatch.setattr(
        grasping,
        "compute_Zi",
        lambda *_args, **_kwargs: np.eye(6, 2),
    )

    contact_points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    contact_frames = np.stack([np.eye(3), np.eye(3)])
    p_world_object = np.zeros(3)
    mu = np.array([0.5, 0.5])

    result = grasping.in_force_closure(
        fingers=[object(), object()],
        contact_points=contact_points,
        contact_frames=contact_frames,
        p_world_object=p_world_object,
        mu=mu,
        nc=2,
    )

    assert result is True
