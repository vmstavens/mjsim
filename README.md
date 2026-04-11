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
import mujoco as mj
from importlib import resources
from mjsim import BaseSim

# Load the packaged empty scene as a starting point.
empty_scene = resources.files("mjsim.assets").joinpath("empty_scene.xml")
model = mj.MjModel.from_xml_path(str(empty_scene))
data = mj.MjData(model)

class EchoSim(BaseSim):
    @property
    def model(self): return model
    @property
    def data(self): return data
    def control_loop(self): pass  # fill in your control logic

EchoSim().run(headless=True)
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
