# Example Simulations

These scripts build and run MuJoCo simulations from the `examples/` directory.
Run them from the repository root so relative asset paths resolve:

```bash
uv run python examples/<file>.py
```

Most examples open the native MuJoCo viewer. Close the viewer window to exit.

| File | Simulation |
| --- | --- |
| `camera_capture.py` | Minimal box scene with a fixed camera. Press `C` to capture and display the camera image with OpenCV. |
| `cartesian_planning.py` | UR10e Cartesian path playback around an obstacle. The planned path is shown in the scene and replayed kinematically. |
| `gelsight_demo.py` | GelSight Mini-style tactile sensor with a spherical indenter. Press `G` to display raw camera, tactile, and depth images. |
| `mk_flex.py` | MuJoCo flex catalogue with a rope, cloth, soft cube, and custom mesh flex. Supports `--headless`. |
| `mocap_rod_wall_force_plot.py` | Mocap body pushes a welded rod into a wall while the weld force is measured and optionally plotted live. Supports `--headless`. |
| `mocap_sensor.py` | Welded free-body load example comparing site force sensor readings with constraint wrench values. |
| `mocap_sensor_api.py` | Box welded through a mocap-mounted load-cell body, using a reusable force/torque sensor helper. Supports `--headless`. |
| `mocap_sensor_dlo.py` | Hanging deformable cable welded to a mocap-mounted load cell with force and torque readings. |
| `modelling.py` | Combined scene with a Robotiq gripper, bunny mesh, cable, cloth, jello block, and free ball. |
| `robot.py` | UR10e with Robotiq gripper using `mjsim.Robot` wrappers and keyboard callbacks. |
| `sim.py` | Minimal UR5e and Robotiq gripper scene stepped in a passive MuJoCo viewer. |
| `test.py` | Local scratch simulation for UR10e, Robotiq gripper, ball, and optional Harting connector mesh if available on disk. |
| `ur5e_control.py` | UR10e tool force/torque sensor simulation with optional gripper and actuated poking finger. |
| `velux_sealing.py` | Flexible Velux sealing strip built from repeated mesh segments. Supports `--headless`. |
| `viser.py` | Unitree G1 humanoid scene served through `mjviser` at `http://localhost:8080`. |

## Preview GIFs

Small GIFs can be added next to this README under `examples/media/`, for example
`examples/media/cartesian_planning.gif` or `examples/media/mk_flex.gif`.
