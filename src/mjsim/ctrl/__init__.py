"""Controllers and motion primitives."""

import os
from importlib import util

_LIGHT_IMPORT = os.environ.get("MJSIM_LIGHT_IMPORT") == "1"

from mjsim.ctrl.dmp_cartesian.dmp_cartesian import DMPCartesian
from mjsim.ctrl.dmp_position import DMPPosition
from mjsim.ctrl.dmp_quaternion import DMPQuaternion

OpSpace = None  # default for environments without MuJoCo
if not _LIGHT_IMPORT and util.find_spec("mujoco") is not None:
    try:
        from mjsim.ctrl.opspace.opspace import OpSpace  # type: ignore[redefined-outer-name]
    except Exception:  # pragma: no cover - MuJoCo or deps missing
        OpSpace = None  # type: ignore[assignment]

__all__ = ["DMPCartesian", "DMPPosition", "DMPQuaternion", "OpSpace"]
