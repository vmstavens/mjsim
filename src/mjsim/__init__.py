"""Convenient entry points for the `mjsim` MuJoCo toolkit."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import util
from pathlib import Path

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


def _maybe_generate_stubs() -> None:
    """
    Generate binary-extension stubs into the installed package for IDEs.

    We keep this best-effort and skip when:
    - pybind11-stubgen is unavailable
    - the user opts out via MJSIM_SKIP_STUBGEN=1
    - stubs were already generated (presence of .typings/.stamp)
    """

    if os.environ.get("MJSIM_SKIP_STUBGEN") == "1":
        return

    stubgen = shutil.which("pybind11-stubgen")
    if stubgen is None:
        return

    stub_root = Path(__file__).resolve().parent / ".typings"
    stamp = stub_root / ".stamp"
    if stamp.exists():
        return

    stub_root.mkdir(exist_ok=True)

    # Only include packages that are present to avoid needless failures.
    targets = []
    for pkg in ("mujoco", "ompl", "open3d"):
        if util.find_spec(pkg):
            targets.append(pkg)

    # Add any other heavy binary modules we ship alongside.
    if util.find_spec("mink"):
        targets.append("mink")

    if not targets:
        return

    cmd = [stubgen, *targets, "-o", str(stub_root)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stamp.write_text("\n".join(targets))
    except Exception:
        # Suppress failures to avoid impacting runtime; IDE users can rerun manually.
        return


if not _LIGHT_IMPORT:
    _maybe_generate_stubs()

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
