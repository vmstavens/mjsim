"""Convenient entry points for the `mjsim` MuJoCo toolkit."""

from __future__ import annotations

import os
from importlib import util
from typing import TYPE_CHECKING

from mjsim._stubgen import add_existing_stubs_to_path, ensure_stubs
import mjsim.ctrl as ctrl

if TYPE_CHECKING:
    # Static-only imports so editors can resolve signatures and docstrings.
    from mjsim.base.robot import Robot
    from mjsim.base.sim import BaseSim, SimSync, sleep
    from mjsim.ctrl import DMPCartesian, DMPPosition, DMPQuaternion, OpSpace
    from mjsim.utils.mj import get_pose
    from mjsim.utils.mjs import cable, cloth, empty_scene, pipe, replicate
    from mjsim.utils.ompl import qplan, xplan

_LIGHT_IMPORT = os.environ.get("MJSIM_LIGHT_IMPORT") == "1"

if not _LIGHT_IMPORT:
    from mjsim.ctrl import DMPCartesian, DMPPosition, DMPQuaternion, OpSpace

    # Only import modules requiring MuJoCo when available and not in light mode.
    if util.find_spec("mujoco") is not None:
        try:
            from mjsim.base.robot import Robot
            from mjsim.base.sim import BaseSim, SimSync, sleep
        except Exception:  # pragma: no cover - optional import for doc builds or missing libs
            Robot = BaseSim = SimSync = sleep = None  # type: ignore[assignment]

        try:
            from mjsim.utils.mj import get_pose
            from mjsim.utils.mjs import cable, cloth, empty_scene, pipe, replicate
        except Exception:  # pragma: no cover - optional import for doc builds or missing libs
            get_pose = None  # type: ignore[assignment]
            cable = cloth = empty_scene = pipe = replicate = None  # type: ignore[assignment]
    else:  # noqa: WPS513
        Robot = BaseSim = SimSync = sleep = None  # type: ignore[assignment]
        get_pose = None  # type: ignore[assignment]
        cable = cloth = empty_scene = pipe = replicate = None  # type: ignore[assignment]

    if util.find_spec("ompl") is not None:
        try:
            from mjsim.utils.ompl import qplan, xplan
        except Exception:  # pragma: no cover - ompl may be missing
            qplan = xplan = None  # type: ignore[assignment]
    else:
        qplan = xplan = None  # type: ignore[assignment]
else:
    DMPCartesian = DMPPosition = DMPQuaternion = OpSpace = None  # type: ignore[assignment]
    Robot = BaseSim = SimSync = sleep = None  # type: ignore[assignment]
    get_pose = None  # type: ignore[assignment]
    cable = cloth = empty_scene = pipe = replicate = None  # type: ignore[assignment]
    qplan = xplan = None  # type: ignore[assignment]


if not _LIGHT_IMPORT:
    ensure_stubs()
    add_existing_stubs_to_path()

__all__ = [
    "BaseSim",
    "SimSync",
    "Robot",
    "DMPCartesian",
    "DMPPosition",
    "DMPQuaternion",
    "OpSpace",
    "ctrl",
    "empty_scene",
    "cable",
    "cloth",
    "pipe",
    "replicate",
    "get_pose",
    "qplan",
    "xplan",
    "sleep",
]
