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
During installation (via a `.pth` hook) and on first import the package tries to generate `typings/` stubs in the project root (or the install's `site-packages` root when the project root is unknown) using `pybind11-stubgen` for `mujoco`, `mujoco.mjx`, `ompl`, `open3d` (and `mink` if present). Set `MJSIM_SKIP_STUBGEN=1` to skip or rerun manually:
```bash
pybind11-stubgen mujoco mujoco.mjx ompl open3d mink -o $(python - <<'PY'
from mjsim._stubgen import _stub_root
print(_stub_root())
PY)
```
