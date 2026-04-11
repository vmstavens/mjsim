"""Base simulation utilities."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from mjsim.base.sim import BaseSim, SimSync, sleep

if TYPE_CHECKING:
    from mjsim.base.robot import Robot


def __getattr__(name: str):
    if name != "Robot":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = import_module("mjsim.base.robot").Robot
    globals()[name] = value
    return value

__all__ = ["BaseSim", "SimSync", "Robot", "sleep"]
