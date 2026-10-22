# mjsim

MuJoCo utilities for quickly stitching together controllers, planners, sensors, and deformable-object assets.

## Installation
- Install MuJoCo and its runtime dependencies per the [official guide](https://mujoco.readthedocs.io/).
- Then install the package (editable mode while developing):
  ```bash
  pip install -e .
  ```

## Quick start

```python
import glfw
import mjsim as ms
import mujoco as mj
from robot_descriptions import ur10e_mj_description, robotiq_2f85_mj_description
import spatialmath as sm

class Sim(ms.BaseSim):
    def __init__(self):
        self._model, self._data = self._init()

        self.ur = ms.Robot(self.model, self.data, "ur/")
        self.gripper = ms.Robot(self.model, self.data, "ur/gripper/")

        T = ms.get_pose(self.model, self.data, self.ur.info.site_names[0], ms.ObjType.SITE)

        self.ctrl = ms.OpSpace(self.ur, gravity_comp=True)

    def _init(self) -> tuple[mj.MjModel, mj.MjData]:

        scene = ms.empty_scene()

        ur = mj.MjSpec.from_file(ur10e_mj_description.MJCF_PATH)

        gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)

        s_attachment = ur.site("attachment_site")

        s_attachment.attach_body(gripper.worldbody.first_body(),prefix="gripper/")

        f_ur = scene.worldbody.add_frame()

        scene.attach(ur, "ur/", frame=f_ur)

        b_ball = scene.worldbody.add_body(name="ball", pos=[0,0,1])
        b_ball.add_geom(name="ball", size=[0.01])
        b_ball.add_freejoint()

        m = scene.compile()
        return m, mj.MjData(m)

    @property
    def model(self):
        return self._model
    @property
    def data(self):
        return self._data

    def keyboard_callback(self, key):
        if key is glfw.KEY_SPACE:
            print("Placing ball at [0, 0, 1]...")
            ms.set_pose(self.model, self.data, "ball", ms.ObjType.BODY, sm.SE3.Tz(1))
        if key is glfw.KEY_PERIOD:
            print("Printing gripper information:")
            print(self.gripper.info)


    @ms.thread
    def see_me_run(self, ss: ms.SimSync):
        while True:
            ss.step()

if __name__ == "__main__":
    sim = Sim()

    sim.run()
```
## Included building blocks
- Controllers: operational-space impedance (`mjsim.ctrl.OpSpace`) and Cartesian/quaternion DMPs (`mjsim.ctrl.DMPCartesian`).
- Planners: OMPL-backed joint- and task-space planning helpers (`mjsim.utils.ompl.qplan`/`xplan`).
- Deformables: shortcuts for cables, cloth, and replicated geometry (`mjsim.utils.mjs.cable`, `cloth`, `pipe`, `replicate`).
- Sensors: MuJoCo camera wrapper with RGB/depth/segmentation and point-cloud extraction (`mjsim.sensors.Camera`).

## Examples

<table>
  <tr>
    <td><a href="examples/physics/models/extended/sleep/dominos.xml"><img src="examples/media/dominos.gif" alt="dominos" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/replicate/newton_cradle.xml"><img src="examples/media/newton_cradle.gif" alt="newton_cradle" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/extended/car/car.xml"><img src="examples/media/car.gif" alt="car" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="examples/physics/models/extended/plugin/sdf/gear.xml"><img src="examples/media/gears.gif" alt="gears" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/extended/plugin/sdf/nutbolt.xml"><img src="examples/media/nut_bolt.gif" alt="nut_bolt" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/extended/cube/cube_3x3x3.xml"><img src="examples/media/twisty_cube.gif" alt="twisty_cube" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="examples/physics/models/extended/flex/flag.xml"><img src="examples/media/flag.gif" alt="flag" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/extended/flex/trampoline.xml"><img src="examples/media/trampoline.gif" alt="trampoline" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/hammock/hammock.xml"><img src="examples/media/hammock.gif" alt="hammock" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="examples/physics/models/extended/flex/basket.xml"><img src="examples/media/basket.gif" alt="basket" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/flex/jelly.xml"><img src="examples/media/jelly.gif" alt="jelly" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/extended/flex/gripper_trilinear.xml"><img src="examples/media/soft_gripper.gif" alt="soft_gripper" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="examples/physics/models/extended/flex/pulley.xml"><img src="examples/media/pulley.gif" alt="pulley" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/extended/plugin/elasticity/coil.xml"><img src="examples/media/coil.gif" alt="coil" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/tendon_arm/arm26.xml"><img src="examples/media/muscle_arm.gif" alt="muscle_arm" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="examples/physics/models/balloons/balloons.xml"><img src="examples/media/balloons.gif" alt="balloons" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/adhesion/active_adhesion.xml"><img src="examples/media/adhesion.gif" alt="adhesion" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/extended/surfacevel/treadmill.xml"><img src="examples/media/treadmill.gif" alt="treadmill" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="examples/conveyor_belt.py"><img src="examples/media/conveyor_belt.gif" alt="conveyor_belt" width="280" height="210"></a></td>
    <td><a href="examples/particle_container.py"><img src="examples/media/particle_container.gif" alt="particle_container" width="280" height="210"></a></td>
    <td><a href="examples/mj_xml.py"><img src="examples/media/mj_xml.gif" alt="mj_xml" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="examples/mocap_gripper_control.py"><img src="examples/media/mocap_gripper_control.gif" alt="mocap_gripper_control" width="280" height="210"></a></td>
    <td><a href="examples/velux_sealing.py"><img src="examples/media/velux_sealing.gif" alt="velux_sealing" width="280" height="210"></a></td>
    <td><a href="examples/physics/models/slider_crank/slider_crank.xml"><img src="examples/media/slider_crank.gif" alt="slider_crank" width="280" height="210"></a></td>
  </tr>
</table>

[Usage](examples/GUIDE.md) · [Sources](examples/physics/models/UPSTREAM.md)

## Testing
Install the optional dev extras and run pytest:
```bash
pip install -e ".[dev]"
pytest
```

### IDE stubs
Generate `typings/` stubs on demand from the root of the project where you use
`mjsim`:
```bash
mjsim-stubgen
```

When using `uv`, run the command through the project environment:
```bash
uv run mjsim-stubgen
```

By default, the command generates stubs for the importable binary dependencies
used by `mjsim`: `mujoco`, `mujoco.mjx`, `ompl`, `open3d`, and `mink`. To target
specific modules or a different output directory:
```bash
mjsim-stubgen mujoco mujoco.mjx -o typings
```
