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
- `examples/controller_dmp.py` – train and roll out a Cartesian DMP.
- `examples/planner_rrt.py` – run a minimal OMPL RRT-Connect in joint and task space.
- `examples/deformable_assets.py` – generate cable and cloth specs and export XML.
- `examples/grasping_force_closure.py` – inspect grasp matrices and friction cones for a toy grasp.
- `examples/cartesian_planning.py` – Cartesian RRT-Connect with a simple obstacle.
- `examples/camera_capture.py` – render RGB/depth from a minimal MuJoCo scene.
- `examples/ik_stub.py` – exercise `Robot.ik` using a stubbed mink backend.

Run any example with `python examples/<file>.py`.

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
