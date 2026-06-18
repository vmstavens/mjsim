"""Utility helpers for MuJoCo simulations."""

from mjsim.utils.mj import (
    ObjType,
    RobotInfo,
    curve_dlo,
    get_joint_ddq,
    get_joint_dq,
    get_joint_q,
    get_pose,
    name2id,
    set_state,
    update_dlo_ref,
)
from mjsim.utils.mjs import cable, cloth, empty_scene, pipe, replicate

__all__ = [
    "ObjType",
    "RobotInfo",
    "curve_dlo",
    "get_joint_ddq",
    "get_joint_dq",
    "get_joint_q",
    "get_pose",
    "name2id",
    "set_state",
    "update_dlo_ref",
    "cable",
    "cloth",
    "pipe",
    "replicate",
    "empty_scene",
    "qplan",
    "xplan",
]


def __getattr__(name: str):
    if name in {"qplan", "xplan"}:
        from mjsim.utils.ompl import qplan, xplan

        return {"qplan": qplan, "xplan": xplan}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
