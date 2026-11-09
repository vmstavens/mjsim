# mjsim

Build robotics and deformable-object simulations with MuJoCo. mjsim brings
controllers, motion planners, sensors, and reusable scene builders together in Python.

- **Control robots:** operational-space impedance control and Cartesian movement primitives.
- **Plan motion:** joint-space and task-space planning with OMPL.
- **Work with soft materials:** cables, cloth, elastic solids, and mesh-based deformables.
- **See and feel:** RGB, depth, segmentation, point clouds, and GelSight tactile sensing.

[Quick start](docs/quickstart/README.md) · [Examples](examples/GUIDE.md) · [Documentation](docs/README.md)

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
    <td><a href="examples/physics/models/extended/flex/coil.xml"><img src="examples/media/coil.gif" alt="coil" width="280" height="210"></a></td>
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
  <tr>
    <td colspan="3" align="center"><a href="examples/bronchoscope.py"><img src="examples/media/bronchoscope.gif" alt="Bronchoscope insertion with tip-camera view" width="280" height="210"></a></td>
  </tr>
</table>

[Usage](examples/GUIDE.md) · [Sources](examples/physics/models/UPSTREAM.md)

## Installation

Requires **MuJoCo 4.x** and **Python 3.11+**. This repository selects Python 3.12.
Install [Git LFS](https://git-lfs.com/) and [uv](https://docs.astral.sh/uv/),
then run from the repository root:

```bash
git lfs install
git lfs pull
uv sync --locked
uv run python examples/heterogeneous_cube.py
```

`uv run` uses the project's virtual environment and the local editable package.
For a simulation without a viewer, add `--headless`.
See the [MuJoCo guide](https://mujoco.readthedocs.io/) for graphics runtime requirements.

## Start exploring

```bash
uv run python examples/mj_xml.py drape
uv run python examples/physics_gallery.py --list
uv run python examples/rod_twist.py
```

Follow the [Python quick start](docs/quickstart/README.md) to build your own scene,
or browse the [example guide](examples/GUIDE.md) for robot manipulation, tactile
sensing, plastic deformation, and speculative contacts.

Upgrading an existing project? See the [migration guide](docs/migration/README.md).
Testing and IDE setup are covered in the [development guide](docs/development/README.md).
