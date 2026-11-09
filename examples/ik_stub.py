"""Minimal inverse kinematics example using the current Robot and Mink APIs.

The historical filename is retained, but the example now uses a tiny in-memory
MuJoCo model and the real solver. No model downloads or viewer are required.
"""

import mujoco as mj
import numpy as np
import spatialmath as sm

from mjsim.base.robot import Robot


def build_robot() -> Robot:
    model = mj.MjModel.from_xml_string("""
      <mujoco>
        <worldbody><body name="arm/tool">
          <joint name="arm/x" type="slide" axis="1 0 0" range="-1 1"/>
          <joint name="arm/y" type="slide" axis="0 1 0" range="-1 1"/>
          <joint name="arm/z" type="slide" axis="0 0 1" range="-1 1"/>
          <geom type="sphere" size=".02" mass="1"/>
          <site name="arm/ee"/>
        </body></worldbody>
      </mujoco>
    """)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return Robot(model, data, "arm/")


def main() -> None:
    robot = build_robot()
    target = sm.SE3(0.3, 0.1, 0.2)
    q_sol = robot.ik(T_target=target, q0=np.zeros(3), site_names="arm/ee")
    np.testing.assert_allclose(q_sol, target.t, atol=1e-3)
    print("IK target:", target.t, "solution:", q_sol)


if __name__ == "__main__":
    main()
