from __future__ import annotations

from importlib import resources


def test_import_and_assets() -> None:
    import mjsim

    asset = resources.files("mjsim.assets").joinpath("empty_scene.xml")
    assert asset.is_file()
    assert "mjsim" in mjsim.__name__
