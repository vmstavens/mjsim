from __future__ import annotations

from importlib import resources


def test_import_and_assets() -> None:
    import mjsim

    asset = resources.files("mjsim.assets").joinpath("empty_scene.xml")
    assert asset.is_file()
    assert "mjsim" in mjsim.__name__


def test_root_exports_public_convenience_api() -> None:
    import mjsim

    expected_exports = {
        "BaseSim",
        "Robot",
        "Camera",
        "DMPCartesian",
        "DMPPosition",
        "DMPQuaternion",
        "OpSpace",
        "ObjType",
        "RobotInfo",
        "ContactState",
        "empty_scene",
        "cable",
        "cloth",
        "replicate",
        "pipe",
        "get_pose",
        "set_pose",
        "name2id",
        "id2name",
        "get_contact_states",
        "qplan",
        "xplan",
    }

    assert expected_exports.issubset(set(mjsim.__all__))
    assert resources.files("mjsim").joinpath("py.typed").is_file()
