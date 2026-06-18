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
