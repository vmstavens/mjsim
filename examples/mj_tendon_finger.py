"""A trilinear elastic bunny under a sliding press; historical filename retained.

Use the MuJoCo viewer actuator slider named press to move the indenter.
"""

from pathlib import Path

import mujoco as mj
from physics.palette import apply_palette

import mjsim as ms

_XML = """
<mujoco model="Trilinear">
  <include file="examples/assets/scene.xml" />

  <compiler meshdir=""/>

  <option solver="CG" tolerance="1e-6" integrator="discrete"/>

  <size memory="10M"/>

  <visual>
    <map stiffness="100"/>
  </visual>

  <worldbody>
    <body>
      <joint name="press" type="slide" axis="0 0 1" damping="500"/>
      <geom type="box" size=".02 .2 .2" pos="0 0 .5"/>
    </body>
    <flexcomp type="mesh" file="examples/assets/stanford-bunny.obj" pos="0 0 0" dim="2" euler="90 0 0" cellcount="3 3 3"
              radius=".001" rgba="0 .7 .7 1" mass=".05" name="softbody" dof="trilinear">
      <elasticity young="1e3" poisson="0.1" damping="0.01" elastic2d="none"/>
      <contact selfcollide="none"/>
    </flexcomp>
  </worldbody>

  <actuator>
    <position name="press" joint="press" gear="-1 0 0 0 0 0" ctrlrange="-1 1" kp="1000"/>
  </actuator>
</mujoco>
"""


class Sim(ms.BaseSim):
    def __init__(self):
        self._model, self._data = self._init()

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:
        assets = (Path(__file__).resolve().parent / "assets").as_posix()
        model = mj.MjModel.from_xml_string(
            _XML.replace("examples/assets/", f"{assets}/")
        )
        apply_palette(model, ())
        data = mj.MjData(model)
        return model, data

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    def keyboard_callback(self, key):
        pass


if __name__ == "__main__":
    print("Elastic bunny press demo: move the 'press' actuator slider in the viewer.")
    Sim().run()
