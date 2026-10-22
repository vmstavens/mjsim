# Running the MuJoCo examples

Start with the **physics gallery** for self-contained physical systems. The gallery
only imports MuJoCo and NumPy, uses bundled XML assets, and works offline. The
remaining examples cover robotics, sensing, planning, and experimental modelling.
Existing script paths are retained so commands and external links keep working.

Run these commands from the repository root with the project environment:

```bash
uv run python examples/physics_gallery.py --list
uv run python examples/physics_gallery.py newton_cradle
uv run python examples/physics_gallery.py jelly --headless --duration 5
```

The gallery itself resolves assets relative to its source file and can also run
from another working directory. Most older examples expect the repository root.
Native viewer examples need a desktop display; headless gallery runs need no OpenGL
context. On macOS, use `uv run mjpython` in place of `uv run python` for the passive viewer.

## Physics gallery

Run `physics_gallery.py <name>` or pass `--list` for the physical systems.

| Name | Required MuJoCo |
| --- | --- |
| `newton_cradle` | 3.3.7+ |
| `balloons` | 3.3.7+ |
| `slider_crank` | 3.3.7+ |
| `muscle_arm` | 3.3.7+ |
| `adhesion` | 3.3.7+ |
| `hammock` | 3.3.7+ |
| `jelly` | 3.3.7+ |
| `car` | 3.11.0+ |
| `dominos` | 3.11.0+ |
| `flag` | 3.11.0+ |
| `trampoline` | 3.11.0+ |
| `soft_gripper` | 3.11.0+ |
| `pulley` | 3.11.0+ |
| `basket` | 3.11.0+ |
| `coil` | 3.11.0+ |
| `gears` | 3.11.0+ |
| `nut_bolt` | 3.11.0+ |
| `treadmill` | 3.11.0+ |
| `twisty_cube` | 3.11.0+ |

The original seven models are pinned to MuJoCo 3.3.7; the expanded selection is
pinned to 3.11.0. The current upstream catalogue was reviewed across vehicle,
contact, deformable, mechanism, and plugin examples. Models requiring newer
features are checked by the loader before compilation. Assets are bundled for
offline use. Sources and Apache 2.0 licensing are in
[UPSTREAM.md](physics/models/UPSTREAM.md).

Use `--manual` for viewer actuator controls, `--original-colors` to compare the
upstream appearance, and `--headless --duration 5` for a simulation-only rollout.
Close the native viewer to exit. Double-click and Ctrl-drag to perturb a body.

## Rendering the GIF grid

```bash
MUJOCO_GL=egl uv run --with pillow python examples/render_physics_gifs.py --fps 10
```

Pass names to render a subset, for example `car dominos gears`. Each model has a
preview duration; `--duration` overrides it. Pillow is only needed for GIF export.
The renderer uses the same model loader, inputs, palette, and cameras as the
interactive gallery. It stops on runtime warnings or non-finite states.
The older gripper example may download its robot description on first use.

## Carbon and flame palette

Every visual example and the XML asset exporter use
[physics/palette.py](physics/palette.py). Most geometry stays
neutral, with orange reserved for selected bodies, tendons, and deformables.

| Color | Hex | Simulation role |
| --- | --- | --- |
| Carbon | `#111214` | Sky and haze |
| Charcoal | `#191b1e` | Ground planes |
| Raised charcoal | `#22252a` | Sky gradient |
| Off-white | `#f2f2ee` | Moving neutral bodies |
| Cool gray | `#adb0b5` | Fixed structures and markers |
| Light orange | `#ff8045` | Tendons, skins, and slider-crank visualizations |
| Flame orange | `#ff6b2b` | Selected bodies and flex objects |
| Graphite | `#36393e` | Available secondary structural tone |

`apply_palette(model, accents=("pusher",))` styles a compiled model before a renderer
or viewer opens. It retains transparency and detaches upstream geometry materials
so textures cannot tint the palette. The palette only changes rendering fields;
physical properties and collision settings are unchanged. Off-white and orange
keep start/goal and actual/target distinct; the admittance target is also box-shaped.
Layered cloth, live plots, browser overlays, and exported XML use the same palette.
The tactile sensor retains its diagnostic RGB response as sensor data.

## Luggage conveyor and carousel


The conveyor follows the [requested reference video](https://www.youtube.com/watch?v=PdSdrqhSiZA):
three feeder belts, an incline, a circular carousel, guardrails, and six luggage shapes.
Bags move through native `surfacevel` contact friction, with no bag actuators.
This example needs **MuJoCo 3.11+** and keeps the carbon-and-flame palette.

```bash
uv run python examples/conveyor_belt.py
uv run python examples/conveyor_belt.py --headless --steps 10000
MUJOCO_GL=egl uv run --with pillow python examples/render_physics_gifs.py conveyor_belt --duration 16 --fps 10
```

Space reverses all moving surfaces, P stops/starts them, and Backspace resets all
bags. The preview is a 16-second rollout. The upstream model is bundled and
[attributed here](physics/models/UPSTREAM.md).


## Existing animated examples

The cloth and sealing previews run for three seconds, stirring for seven, and the
gripper for fourteen. These GIFs use 10 fps. Regenerate just these previews with:

```bash
MUJOCO_GL=egl uv run --with pillow python examples/render_physics_gifs.py mj_xml particle_container velux_sealing mocap_gripper_control --fps 10
```

The gripper uses `robot_descriptions` and may download its model on first use.

## More physical systems and deformables

| Script | Simulation | Notes |
| --- | --- | --- |
| `conveyor_belt.py` | Three feeder belts deliver six bags to a circular luggage carousel using native surface-velocity contact friction | MuJoCo 3.11+; Space reverses, P stops/starts surfaces, Backspace resets; `--headless`. |
| `particle_container.py` | Damped particles stirred by a moving paddle | `--headless`; `--particles` selects approximate count. |
| `mk_flex.py` | Rope, cloth, soft cube, custom mesh flex, and composite cable | `--headless`; requires the newer `make_flex` API (MuJoCo 3.10+). |
| `velux_sealing.py` | Flexible sealing strip built from repeated mesh segments | `--headless`; bundled profile mesh. |
| `earth_digger.py` | Excavator linkage with slider-crank hydraulic transmissions | Native viewer. |
| `mj_tendon_finger.py` | Experimental deformable bunny under an actuated press | Native viewer; newer flex features. |
| `mj_xml.py` | Inline XML experiments for bags and draped deformables | Edit the selected XML in the script; newer flex features. |
| `modelling.py` | Robotiq gripper, bunny, cable, cloth, jello, and free ball | Downloads robot descriptions when uncached. |
| `deformable_assets.py` | Generate cable and cloth specs and export XML | Utility example. |
| `dlo-insert.py` | Cable insertion into a pipe | Viser browser viewer. |

## Robotics and control

Robot examples using `robot_descriptions` may download their model assets on first
run. Some advanced examples need newer MuJoCo APIs than the package minimum.

| Script | Demonstration |
| --- | --- |
| `sim.py` | Minimal UR5e and Robotiq gripper scene in a passive viewer. |
| `robot.py` | UR10e and gripper using the `mjsim.Robot` wrapper and keyboard callbacks. |
| `ur5e_control.py` | Despite its filename, a UR10e tool force/torque sensor scene with optional gripper and poking finger. |
| `mocap_gripper_control.py` | Actuated gripper base picks up, lifts, and rotates a box. Starts with manual controls; `--auto` or `--headless` runs the sequence. |
| `admittance_controller_gui.py` | UR admittance controller with a contact pusher and interactive controls; see [the controller report](report.md). |
| `admittance_controller.py` | Convenience entry point for the admittance demo. |
| `ur10e_dmp_cartesian.py` | Fit a Cartesian DMP to the bundled CSV demonstration and play it on a UR10e. |
| `controller_dmp.py` | Fit and sample a Cartesian/quaternion DMP without running a simulation. |
| `cartesian_planning.py` | UR10e Cartesian path playback around an obstacle. |
| `planner_rrt.py` | Minimal OMPL RRT-Connect in joint and task space. |
| `grasping_force_closure.py` | Toy grasp matrices and friction cones. |
| `ik_stub.py` | IK control flow using a lightweight stand-in backend. |

## Sensors and viewers

| Script | Demonstration |
| --- | --- |
| `camera_capture.py` | Fixed camera in a box scene; `C` displays a capture with OpenCV. |
| `gelsight_demo.py` | GelSight-style tactile sensor with a spherical indenter; `G` displays images. |
| `mocap_sensor.py` | Welded free-body load: compare site force sensor readings with constraint wrenches. |
| `mocap_sensor_api.py` | Box welded to a mocap load cell through a reusable force/torque helper; `--headless`. |
| `mocap_sensor_dlo.py` | Hanging cable welded to a mocap load cell with force and torque readings. |
| `mocap_rod_wall_force_plot.py` | Welded rod pushed into a wall, measuring and optionally plotting force; `--headless`. |
| `viser.py` | Unitree G1 scene served by `mjviser` at `http://localhost:8080`. |

`test.py` is a local scratch scene with UR10e, gripper, ball, and an optional
machine-specific Harting connector mesh. Start with a catalogued example above
for a reproducible demonstration.

## Overlap and palette review

The [complete example review](REVIEW.md) covers all scripts, their overlapping
purposes, and current palette coverage. The main cleanup candidates are the
near-duplicate `test.py` / `robot.py` pair, the convenience admittance wrapper,
and shared load-cell helpers spread across the force demos. The DMP, planning,
and deformable examples also have related themes, but demonstrate distinct APIs
or physical systems.

All displayed scenes and exported deformable assets now use the shared palette.
The four numerical-only examples have no geometry to theme. The review records
marker roles and the intentional distinction between scene styling and diagnostic
sensor responses.
