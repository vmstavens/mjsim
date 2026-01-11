"""Convenient entry points for the `mjsim` MuJoCo toolkit."""

from __future__ import annotations

import os
from importlib import util

from mjsim._stubgen import add_existing_stubs_to_path, ensure_stubs

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
            from mjsim.utils.mjs import cable, cloth, empty_scene, pipe, replicate
        except Exception:  # pragma: no cover - optional import for doc builds or missing libs
            cable = cloth = empty_scene = pipe = replicate = None  # type: ignore[assignment]
    else:  # noqa: WPS513
        Robot = BaseSim = SimSync = sleep = None  # type: ignore[assignment]
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
    cable = cloth = empty_scene = pipe = replicate = None  # type: ignore[assignment]
    qplan = xplan = None  # type: ignore[assignment]


if not _LIGHT_IMPORT:
    add_existing_stubs_to_path()
    ensure_stubs()

__all__ = [
    "BaseSim",
    "SimSync",
    "Robot",
    "DMPCartesian",
    "DMPPosition",
    "DMPQuaternion",
    "OpSpace",
    "empty_scene",
    "cable",
    "cloth",
    "pipe",
    "replicate",
    "qplan",
    "xplan",
    "sleep",
]
