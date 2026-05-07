from __future__ import annotations

import numpy as np
import pytest


def _keyframe_model():
    mj = pytest.importorskip("mujoco")
    xml = """
    <mujoco>
        <worldbody>
            <body name="box">
                <joint name="slide" type="slide" axis="1 0 0"/>
                <geom name="box_geom" type="box" size="0.05 0.05 0.05"/>
            </body>
        </worldbody>
        <actuator>
            <motor name="slide_motor" joint="slide"/>
        </actuator>
        <keyframe>
            <key name="home" qpos="0.25" qvel="0.5" ctrl="0.1"/>
            <key name="away" qpos="1.0" qvel="-0.25" ctrl="-0.1"/>
        </keyframe>
    </mujoco>
    """
    return mj, mj.MjModel.from_xml_string(xml)


def test_set_state_resets_data_from_named_keyframe() -> None:
    mj, model = _keyframe_model()
    data = mj.MjData(model)

    data.qpos[:] = 9.0
    data.qvel[:] = 8.0

    from mjsim.utils.mj import set_state

    set_state(model, data, "home")

    np.testing.assert_allclose(data.qpos, [0.25])
    np.testing.assert_allclose(data.qvel, [0.5])
    np.testing.assert_allclose(data.ctrl, [0.1])


def test_set_state_accepts_keyframe_index() -> None:
    mj, model = _keyframe_model()
    data = mj.MjData(model)

    from mjsim.utils.mj import set_state

    set_state(model, data, 1)

    np.testing.assert_allclose(data.qpos, [1.0])
    np.testing.assert_allclose(data.qvel, [-0.25])
    np.testing.assert_allclose(data.ctrl, [-0.1])


def test_set_state_accepts_key_keyword() -> None:
    mj, model = _keyframe_model()
    data = mj.MjData(model)

    from mjsim.utils.mj import set_state

    set_state(model, data, key="home")

    np.testing.assert_allclose(data.qpos, [0.25])


def test_set_state_sets_explicit_fields_without_keyframe() -> None:
    mj, model = _keyframe_model()
    data = mj.MjData(model)

    from mjsim.utils.mj import set_state

    set_state(model, data, time=2.5, qpos=[0.75], qvel=[-0.5], ctrl=[0.3])

    assert data.time == pytest.approx(2.5)
    np.testing.assert_allclose(data.qpos, [0.75])
    np.testing.assert_allclose(data.qvel, [-0.5])
    np.testing.assert_allclose(data.ctrl, [0.3])


def test_set_state_applies_explicit_fields_after_keyframe() -> None:
    mj, model = _keyframe_model()
    data = mj.MjData(model)

    from mjsim.utils.mj import set_state

    set_state(model, data, "home", qpos=[0.8])

    np.testing.assert_allclose(data.qpos, [0.8])
    np.testing.assert_allclose(data.qvel, [0.5])


def test_set_state_validates_array_size() -> None:
    mj, model = _keyframe_model()
    data = mj.MjData(model)

    from mjsim.utils.mj import set_state

    with pytest.raises(ValueError, match="qpos has size 2, expected 1"):
        set_state(model, data, qpos=[0.1, 0.2])


def test_set_state_rejects_missing_keyframe() -> None:
    mj, model = _keyframe_model()
    data = mj.MjData(model)

    from mjsim.utils.mj import set_state

    with pytest.raises(ValueError, match="Keyframe 'missing' not found"):
        set_state(model, data, "missing")


def test_set_state_is_exported_from_package_root() -> None:
    import mjsim

    assert "set_state" in mjsim.__all__
