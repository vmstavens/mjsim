from __future__ import annotations

from importlib import resources

import pytest


def test_import_and_assets() -> None:
    import mjsim

    asset = resources.files("mjsim.assets").joinpath("empty_scene.xml")
    assert asset.is_file()
    assert "mjsim" in mjsim.__name__


def test_root_exports_public_convenience_api() -> None:
    import mjsim

    expected_exports = {
        "BaseSim",
        "thread",
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
        "set_state",
        "name2id",
        "id2name",
        "get_contact_states",
        "qplan",
        "xplan",
    }

    assert expected_exports.issubset(set(mjsim.__all__))
    assert resources.files("mjsim").joinpath("py.typed").is_file()


def test_root_basesim_export_resolves_to_class() -> None:
    pytest.importorskip("mujoco")

    import mjsim

    assert isinstance(mjsim.BaseSim, type)


def test_basesim_subclass_uses_model_and_data_properties() -> None:
    pytest.importorskip("mujoco")

    import mjsim

    class MinimalSim(mjsim.BaseSim):
        def __init__(self) -> None:
            super().__init__()
            self._model = object()
            self._data = object()

        @property
        def model(self):
            return self._model

        @property
        def data(self):
            return self._data

    sim = MinimalSim()

    assert sim.model is sim._model
    assert sim.data is sim._data


def test_basesim_discovers_thread_decorated_methods_without_init() -> None:
    pytest.importorskip("mujoco")

    import mjsim

    class DecoratedSim(mjsim.BaseSim):
        def __init__(self) -> None:
            self._model = object()
            self._data = object()

        @property
        def model(self):
            return self._model

        @property
        def data(self):
            return self._data

        @mjsim.thread
        def thread1(self, ss):
            return None

        def helper(self, ss):
            return None

    sim = DecoratedSim()

    assert [callback.__name__ for callback in sim._thread_callbacks()] == ["thread1"]
    with pytest.warns(UserWarning, match="control_loop"):
        sim.control_loop()
