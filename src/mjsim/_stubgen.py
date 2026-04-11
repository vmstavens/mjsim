"""Helpers for generating IDE stubs for bundled binary dependencies."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
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
    for pkg in ("mujoco", "mujoco.mjx", "ompl", "ompl.base", "ompl.geometric", "open3d"):
        if util.find_spec(pkg):
            targets.append(pkg)

    if util.find_spec("mink"):
        targets.append("mink")

    # targets = ["mujoco", "mujoco.mjx", "ompl", "open3d", "mink"]

    return targets


def _available_modules(modules: Iterable[str]) -> list[str]:
    return [module for module in modules if util.find_spec(module)]


def _run_stubgen(
    stub_root: Path,
    targets: Iterable[str],
    *,
    quiet: bool,
) -> bool:
    stubgen = shutil.which("pybind11-stubgen")
    if stubgen is None:
        if not quiet:
            print("pybind11-stubgen was not found on PATH.", file=sys.stderr)
        return False

    target_list = list(targets)
    if not target_list:
        if not quiet:
            print("No requested modules are importable.", file=sys.stderr)
        return False

    stub_root.mkdir(parents=True, exist_ok=True)
    for target in target_list:
        cmd = [
            stubgen,
            target,
            "-o",
            stub_root.name,
            "--ignore-invalid-expressions=.*",
            "--ignore-invalid-identifiers=.*",
            "--ignore-all-errors",
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL if quiet else None,
                stderr=subprocess.DEVNULL if quiet else None,
                cwd=str(stub_root.parent),
            )
        except Exception as exc:
            if not quiet:
                print(f"pybind11-stubgen failed for {target}: {exc}", file=sys.stderr)
            return False

    return True


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

    targets = list(_target_modules())
    if not _run_stubgen(stub_root, targets, quiet=True):
        return

    stamp.write_text("\n".join(targets))

    _add_stub_path(stub_root)


def add_existing_stubs_to_path() -> None:
    """Expose the generated stubs on sys.path when they already exist."""

    stub_root = _stub_root()
    if (stub_root / ".stamp").exists():
        _add_stub_path(stub_root)


def main(argv: list[str] | None = None) -> None:
    """Run stub generation on demand from the current working directory."""

    parser = ArgumentParser(
        prog="mjsim-stubgen",
        description="Generate pybind11 stubs for mjsim dependencies into ./typings.",
    )
    parser.add_argument(
        "modules",
        nargs="*",
        help="Modules to generate stubs for. Defaults to all importable mjsim binary dependencies.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path("typings"),
        type=Path,
        help="Output directory. Defaults to ./typings.",
    )
    args = parser.parse_args(argv)

    stub_root = args.output if args.output.is_absolute() else Path.cwd() / args.output
    requested = args.modules or list(_target_modules())
    targets = _available_modules(requested)
    missing = sorted(set(requested) - set(targets))
    if args.modules and missing:
        print(
            "Skipping modules that are not importable: " + ", ".join(missing),
            file=sys.stderr,
        )

    if not _run_stubgen(stub_root, targets, quiet=False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
