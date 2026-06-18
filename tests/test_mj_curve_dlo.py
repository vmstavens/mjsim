import os

import numpy as np
import pytest

if os.environ.get("MJSIM_LIGHT_IMPORT") == "1":
    pytest.skip(
        "Skipping MuJoCo-dependent tests in light mode", allow_module_level=True
    )


def test_curve_dlo_updates_only_prefixed_ball_joints():
    mj = pytest.importorskip("mujoco")

    from mjsim.utils.math import rotvec_to_quat
    from mjsim.utils.mj import curve_dlo

    model = mj.MjModel.from_xml_string(
        """
        <mujoco>
            <worldbody>
            <body name="root">
              <joint name="cable/j0" type="ball"/>
              <geom type="sphere" size="0.01" mass="0.01"/>
              <body name="link">
                <joint name="cable/j1" type="ball"/>
                <geom type="sphere" size="0.01" mass="0.01"/>
              </body>
              <body name="unrelated">
                <joint name="other/j0" type="ball"/>
                <geom type="sphere" size="0.01" mass="0.01"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mj.MjData(model)

    curve_dlo(data, model, theta=0.6, phi=np.pi / 2, forward=False)

    expected = rotvec_to_quat(np.array([0.0, 0.0, 0.3]))
    np.testing.assert_allclose(data.qpos[0:4], expected, atol=1e-15)
    np.testing.assert_allclose(data.qpos[4:8], expected, atol=1e-15)
    np.testing.assert_allclose(data.qpos[8:12], np.array([1.0, 0.0, 0.0, 0.0]))


def test_curve_dlo_rejects_invalid_cable_count_even_without_matches():
    mj = pytest.importorskip("mujoco")

    from mjsim.utils.mj import curve_dlo

    model = mj.MjModel.from_xml_string("<mujoco/>")
    data = mj.MjData(model)

    with pytest.raises(ValueError, match="num_cables must be > 0"):
        curve_dlo(data, model, theta=0.0, phi=0.0, num_cables=0)


def test_update_dlo_ref_updates_mj_model_ball_joints():
    mj = pytest.importorskip("mujoco")

    from mjsim.utils.mj import update_dlo_ref

    model = mj.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <joint name="cable/j0" type="ball"/>
              <geom type="sphere" size="0.01" mass="0.01"/>
              <body name="other">
                <joint name="other/j0" type="ball"/>
                <geom type="sphere" size="0.01" mass="0.01"/>
              </body>
              <body name="link">
                <joint name="cable/j1" type="ball"/>
                <geom type="sphere" size="0.01" mass="0.01"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    qpos = np.array(
        [
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.0,
        ]
    )

    updated_model, updated = update_dlo_ref(model, qpos, "cable/")

    assert updated_model is model
    assert updated == ["cable/j0", "cable/j1"]
    np.testing.assert_allclose(model.qpos_spring[0:4], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(model.qpos_spring[4:8], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(model.qpos_spring[8:12], [0.0, 0.0, 0.0, 1.0])


def test_update_dlo_ref_supports_mjx_model():
    mj = pytest.importorskip("mujoco")
    mjx = pytest.importorskip("mujoco.mjx")

    from mjsim.utils.mj import update_dlo_ref

    model = mj.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <joint name="cable/j0" type="ball"/>
              <geom type="sphere" size="0.01" mass="0.01"/>
              <body name="link">
                <joint name="cable/j1" type="ball"/>
                <geom type="sphere" size="0.01" mass="0.01"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    mjx_model = mjx.put_model(model)

    updated_model, updated = update_dlo_ref(
        mjx_model,
        np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]),
        "cable/",
        joint_name="cable/j1",
    )

    assert updated == ["cable/j1"]
    np.testing.assert_allclose(np.asarray(mjx_model.qpos_spring)[0:4], [1, 0, 0, 0])
    np.testing.assert_allclose(
        np.asarray(updated_model.qpos_spring)[0:4],
        [1.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        np.asarray(updated_model.qpos_spring)[4:8],
        [0.0, 0.0, 0.0, 1.0],
    )
