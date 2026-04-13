"""Public convenience API for the :mod:`mjsim` MuJoCo toolkit.

The package root intentionally re-exports the most common simulation,
controller, sensor, MJCF-generation, and MuJoCo helper entry points. Heavy
runtime dependencies are imported only when they are available so lightweight
tooling can still inspect the package.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module, util
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mjsim._stubgen import add_existing_stubs_to_path, ensure_stubs

if TYPE_CHECKING:
    # Static-only imports so editors can resolve signatures and docstrings.
    from mjsim.base.robot import Robot  # noqa: F401
    from mjsim.base.sim import BaseSim, SimSync, sleep, thread  # noqa: F401
    from mjsim.ctrl import (  # noqa: F401
        DMPCartesian,
        DMPPosition,
        DMPQuaternion,
        OpSpace,
    )
    from mjsim.sensors import Camera  # noqa: F401
    from mjsim.utils.mj import (  # noqa: F401
        ContactState,
        JointType,
        ObjType,
        RobotInfo,
        add_act_freejoint,
        apply_wrench,
        does_exist,
        get_bodies_in_contact,
        get_contact_states,
        get_geom_distance,
        get_geoms_in_contact,
        get_ids,
        get_joint_ddq,
        get_joint_dim,
        get_joint_dof_indxs,
        get_joint_dq,
        get_joint_pos,
        get_joint_q,
        get_joint_qpos_addr,
        get_joint_qpos_indxs,
        get_names,
        get_number_of,
        get_pose,
        get_type,
        id2name,
        is_robot_entity,
        jnt2act,
        load_keyframe,
        mk_keyframe_file,
        name2id,
        save_keyframe,
        set_joint_ddq,
        set_joint_dq,
        set_joint_q,
        set_pose,
    )
    from mjsim.utils.mjs import (  # noqa: F401
        cable,
        cloth,
        dco,
        deform_3d,
        deform_3d_custom,
        dlo,
        dmo,
        dqo,
        empty_scene,
        mesh,
        pipe,
        pipe_legacy,
        replicate,
        weld,
    )
    from mjsim.utils.ompl import qplan, xplan  # noqa: F401

_RUNNING_STUBGEN_CLI = Path(sys.argv[0]).stem == "mjsim-stubgen"
_LIGHT_IMPORT = os.environ.get("MJSIM_LIGHT_IMPORT") == "1" or _RUNNING_STUBGEN_CLI

_BASE_EXPORTS = ["BaseSim", "SimSync", "Robot", "sleep", "thread"]
_CTRL_EXPORTS = ["DMPCartesian", "DMPPosition", "DMPQuaternion", "OpSpace", "ctrl"]
_SENSOR_EXPORTS = ["Camera"]
_MJ_HELPER_EXPORTS = [
    "JointType",
    "ObjType",
    "RobotInfo",
    "ContactState",
    "get_number_of",
    "get_names",
    "get_type",
    "get_ids",
    "name2id",
    "id2name",
    "does_exist",
    "get_pose",
    "set_pose",
    "get_joint_qpos_addr",
    "set_joint_q",
    "set_joint_dq",
    "set_joint_ddq",
    "get_joint_q",
    "get_joint_dq",
    "get_joint_ddq",
    "get_joint_qpos_indxs",
    "get_joint_dof_indxs",
    "get_joint_dim",
    "load_keyframe",
    "mk_keyframe_file",
    "save_keyframe",
    "apply_wrench",
    "get_geoms_in_contact",
    "get_bodies_in_contact",
    "get_joint_pos",
    "get_geom_distance",
    "get_contact_states",
    "is_robot_entity",
    "jnt2act",
    "add_act_freejoint",
]
_MJCF_EXPORTS = [
    "empty_scene",
    "dlo",
    "dqo",
    "dco",
    "dmo",
    "cable",
    "mesh",
    "cloth",
    "deform_3d",
    "deform_3d_custom",
    "weld",
    "replicate",
    "pipe",
    "pipe_legacy",
]
_PLANNER_EXPORTS = ["qplan", "xplan"]

_EXPORT_MODULES = {
    "ctrl": "mjsim.ctrl",
    "DMPCartesian": "mjsim.ctrl",
    "DMPPosition": "mjsim.ctrl",
    "DMPQuaternion": "mjsim.ctrl",
    "OpSpace": "mjsim.ctrl",
    "Robot": "mjsim.base.robot",
    "BaseSim": "mjsim.base.sim",
    "SimSync": "mjsim.base.sim",
    "sleep": "mjsim.base.sim",
    "thread": "mjsim.base.sim",
    "Camera": "mjsim.sensors",
    **{name: "mjsim.utils.mj" for name in _MJ_HELPER_EXPORTS},
    **{name: "mjsim.utils.mjs" for name in _MJCF_EXPORTS},
    **{name: "mjsim.utils.ompl" for name in _PLANNER_EXPORTS},
}


def _missing_dependency(name: str) -> str | None:
    if name in _BASE_EXPORTS + _SENSOR_EXPORTS + _MJ_HELPER_EXPORTS + _MJCF_EXPORTS:
        return "mujoco" if util.find_spec("mujoco") is None else None
    if name in _PLANNER_EXPORTS:
        return "ompl" if util.find_spec("ompl") is None else None
    return None


def __getattr__(name: str) -> Any:
    """Lazily resolve public exports while keeping ``import mjsim`` lightweight."""

    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if _LIGHT_IMPORT:
        globals()[name] = None
        return None

    missing_dependency = _missing_dependency(name)
    if missing_dependency is not None:
        msg = (
            f"Cannot import mjsim.{name}: optional dependency "
            f"{missing_dependency!r} is not installed."
        )
        raise ImportError(msg)

    module = import_module(_EXPORT_MODULES[name])
    value = module if name == "ctrl" else getattr(module, name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in ``dir(mjsim)``."""

    return sorted(set(globals()) | set(__all__))


if not _LIGHT_IMPORT:
    ensure_stubs()
    add_existing_stubs_to_path()

__all__ = [
    *_BASE_EXPORTS,
    *_CTRL_EXPORTS,
    *_SENSOR_EXPORTS,
    *_MJ_HELPER_EXPORTS,
    *_MJCF_EXPORTS,
    *_PLANNER_EXPORTS,
]
