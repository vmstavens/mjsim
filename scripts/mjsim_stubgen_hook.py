"""Site hook for generating pybind11 stubs during install/import."""

from __future__ import annotations

from pathlib import Path


def _safe_ensure_stubs() -> None:
    try:
        from mjsim._stubgen import ensure_stubs
    except Exception:
        return

    try:
        ensure_stubs()
    except Exception:
        return


def _disable_hook_after_install() -> None:
    """Remove the .pth hook so it only runs once per install."""

    hook_path = Path(__file__).resolve()
    pth_path = hook_path.with_name("mjsim-stubgen.pth")
    if not pth_path.exists():
        return

    # Only delete hooks from site-packages/dist-packages installs.
    parts = {part.lower() for part in pth_path.parts}
    if "site-packages" not in parts and "dist-packages" not in parts:
        return

    try:
        pth_path.unlink()
    except Exception:
        return


_safe_ensure_stubs()
_disable_hook_after_install()
