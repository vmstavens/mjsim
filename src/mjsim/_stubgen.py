"""Helpers for generating IDE stubs for bundled binary dependencies."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import util
from pathlib import Path
from typing import Iterable


def _project_root() -> Path | None:
    """Try to locate the project root (where pyproject.toml lives)."""

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def _stub_root() -> Path:
    root = _project_root()
    base = root if root is not None else Path(__file__).resolve().parent.parent
    return base / "typings"


def _add_stub_path(stub_root: Path) -> None:
    if stub_root.is_dir():
        stub_path = str(stub_root)
        if stub_path not in sys.path:
            sys.path.insert(0, stub_path)


def _target_modules() -> Iterable[str]:
    targets: list[str] = []
    for pkg in ("mujoco", "mujoco.mjx", "ompl", "open3d"):
        if util.find_spec(pkg):
            targets.append(pkg)

    if util.find_spec("mink"):
        targets.append("mink")

    return targets


def ensure_stubs() -> None:
    """
    Generate binary-extension stubs into the installed package for IDEs.

    This is intentionally best-effort and silently skips when:
    - pybind11-stubgen is unavailable
    - the user opts out via MJSIM_SKIP_STUBGEN=1
    - stubs already exist (typings/.stamp)
    - none of the target modules are importable
    """

    if os.environ.get("MJSIM_SKIP_STUBGEN") == "1":
        return

    stub_root = _stub_root()
    stamp = stub_root / ".stamp"
    if stamp.exists():
        _add_stub_path(stub_root)
        return

    stubgen = shutil.which("pybind11-stubgen")
    if stubgen is None:
        return

    targets = list(_target_modules())
    if not targets:
        return

    stub_root.mkdir(exist_ok=True)
    output_dir = "./typings"
    output_root = stub_root.parent

    cmd = [
        stubgen,
        *targets,
        "-o",
        output_dir,
        "--ignore-invalid-expressions=.*",
        "--ignore-invalid-identifiers=.*",
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(output_root),
        )
    except Exception:
        return

    stamp.write_text("\n".join(targets))

    _add_stub_path(stub_root)


def add_existing_stubs_to_path() -> None:
    """Expose the generated stubs on sys.path when they already exist."""

    stub_root = _stub_root()
    if (stub_root / ".stamp").exists():
        _add_stub_path(stub_root)


def main() -> None:
    """Run stub generation on demand."""

    ensure_stubs()


if __name__ == "__main__":
    main()
