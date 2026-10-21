# Example overlap and palette review

This review covers all 32 top-level Python scripts, the gallery loader, and the
shared scene defaults. Palette migration is complete for every displayed scene
and the XML asset exporter. The 21 migrated scene builders were compiled and
checked on MuJoCo 3.11.0; this does not certify every example's full interactive
behavior. The conveyor additionally completed a 20-second transport rollout.

## Redundancies and useful overlap

| Examples | Finding | Recommended cleanup |
| --- | --- | --- |
| [`robot.py`](robot.py), [`test.py`](test.py) | Both construct a UR10e, Robotiq gripper, and falling ball; instantiate the same robot wrappers and operational-space controller; and expose the same keyboard actions. The scratch script adds optional Harting mesh loading from machine-specific paths. | Keep `robot.py` as the canonical wrapper demo. Move the optional mesh experiment into a clearly named mesh example or archive the scratch script. |
| [`admittance_controller.py`](admittance_controller.py), [`admittance_controller_gui.py`](admittance_controller_gui.py) | The first imports `Sim` and `use_motor_actuators` from the second. It contains no independent controller implementation, but launches the native viewer instead of the browser GUI entry point. | Keep one implementation and document the two front ends. A wrapper is useful for compatibility; it should not be counted as a separate physical system. |
| [`sim.py`](sim.py), [`robot.py`](robot.py), [`mocap_gripper_control.py`](mocap_gripper_control.py) | All feature a UR/Robotiq or Robotiq scene. `sim.py` shows manual model composition and stepping with a UR5e; `robot.py` uses `BaseSim` with a UR10e; the mocap-named gripper demo uses an actuated base and demonstrates contact grasping. | Keep the differing APIs and grasp task, but identify `robot.py` as the main starting point and `sim.py` as low-level composition. |
| [`mocap_sensor.py`](mocap_sensor.py), [`mocap_sensor_api.py`](mocap_sensor_api.py), [`mocap_rod_wall_force_plot.py`](mocap_rod_wall_force_plot.py), [`mocap_sensor_dlo.py`](mocap_sensor_dlo.py) | Shared weld/load-cell construction and wrench transforms. The cases differ: a free-body site sensor versus equality loads; an instrumented load-cell API; rod/wall contact with live plotting; and a deformable cable with tare and wrench comparison. The DLO script imports `weld_sensor` from the rod example. | Keep the four teaching cases. Extract shared sensor construction and frame transforms into a helper module so examples do not import another runnable demo. `MocapWeldSensors` is the most general reader, while the rod helper also constructs the load cell. |
| [`ur5e_control.py`](ur5e_control.py), [`admittance_controller_gui.py`](admittance_controller_gui.py) | Both instrument a UR wrist/tool and use a contact pusher. The first reads loads with optional gripper/poker; the second closes an admittance-control loop with inverse-dynamics torques. | Keep measurement and closed-loop control as separate examples; share load-cell construction where practical. |
| [`mk_flex.py`](mk_flex.py), [`modelling.py`](modelling.py), [`deformable_assets.py`](deformable_assets.py), gallery `jelly` | Both catalogues include cables, cloth, and soft blocks. `mk_flex.py` compares flex dimensions and a composite cable; `modelling.py` mixes robot and mesh attachment with deformables; `deformable_assets.py` exports XML; `jelly` isolates volumetric elasticity. | Prefer `mk_flex.py` for the visual catalogue and `jelly` for a focused physics demo. Retain XML export as a utility. Narrow `modelling.py` to scene/mesh attachment so it adds a clear lesson. |
| [`mj_xml.py`](mj_xml.py), gallery `hammock` | Both include cloth-like structures and a humanoid. The inline XML also has a bag catching a humanoid and three free cloths draping over a sphere; the hammock pins four corners. These have distinct contact and boundary conditions. | Keep the distinct systems, but split the inline `bags` / `drape` selection into named entries instead of editing source to choose a scene. |
| [`mj_tendon_finger.py`](mj_tendon_finger.py), gallery `muscle_arm` | The filename and startup message promise a tendon finger, but the actual XML contains a trilinear bunny flex and a sliding press. There is no `finger_curl` tendon; Space also refers to an uninitialized `self.animate`. | Rename/rework this as a bunny-press demo and repair its controls. It is not a duplicate of the tendon/muscle arm. |
| [`controller_dmp.py`](controller_dmp.py), [`ur10e_dmp_cartesian.py`](ur10e_dmp_cartesian.py) | The small example fits/samples a Cartesian DMP numerically; the full one loads CSV, solves robot IK, visualizes the path, and records tracking. | Keep both as basic API and integrated robot examples. Shared concept, different learning scope. |
| [`planner_rrt.py`](planner_rrt.py), [`cartesian_planning.py`](cartesian_planning.py) | The former exercises joint/task planners with a mock robot; the latter solves and replays a UR10e path around an obstacle. | Keep both as planner API and integrated scene examples. |
| [`earth_digger.py`](earth_digger.py), gallery `slider_crank` | Both use slider-crank transmissions. The gallery isolates forward/reverse/broken geometry; the excavator applies transmissions to an articulated machine. | Keep both as mechanism and application. |
| [`particle_container.py`](particle_container.py), gallery `adhesion` | Both contain particles, but the former demonstrates viscous/damped stirring, while the latter uses adhesive contacts and a pulley. | Keep both; shared primitive does not imply redundant physics. |

`velux_sealing.py` (articulated profile strip), `dlo-insert.py` (pipe/cable geometry),
`camera_capture.py` (camera capture), `gelsight_demo.py` (tactile imaging), and
`viser.py` (G1 browser viewing) each add a separate modelling or visualization
task. The buoyant balloons and Newton's cradle also have distinct physics.

## Palette coverage

**Shared** means model rendering fields are themed after compilation, directly or
through the gallery loader. **Shared exports** means the generated XML contains
the palette. **N/A** is reserved for the four numerical examples with no visual
scene. There are no remaining unthemed displayed scenes.

| Script | Palette | Current appearance |
| --- | --- | --- |
| `physics_gallery.py` | Shared | All seven models use carbon/flame by default; original colors remain an explicit comparison option. |
| `render_physics_gifs.py` | Shared | Uses the gallery loader, conveyor, and selected existing simulations; matching cameras and per-example sequence durations. |
| `conveyor_belt.py` | Shared | Three dark feeder belts and carousel, neutral rails, orange/off-white luggage; native contact transport. |
| `mk_flex.py` | Shared | Orange deformables and pusher; neutral surroundings. A preview rollout on 3.11.0 reported QACC instability, so this scene is omitted from the GIF selection. |
| `particle_container.py` | Shared | Neutral particles, orange paddle, retained wall transparency. |
| `admittance_controller.py` | Shared | Inherits the themed controller scene; console description matches marker roles. |
| `admittance_controller_gui.py` | Shared | Off-white actual sphere, orange target box and pusher; dark/orange browser controls and updated report. |
| `camera_capture.py` | Shared | Orange target, neutral background; palette applied before creating the camera renderer. |
| `cartesian_planning.py` | Shared | Off-white start, orange goal/path/obstacle; both initial and rebuilt scenes are themed. |
| `controller_dmp.py` | N/A | Numerical trajectory fit/rollout; no rendered scene. |
| `deformable_assets.py` | Shared exports | Cable, cloth, and block XML carry off-white, light-orange, and flame colors respectively. |
| `dlo-insert.py` | Shared | Orange cable, transparent gray pipe, dark floor; matching browser grid, neutral lights, dark/orange controls, and labeled palette-colored axes. |
| `earth_digger.py` | Shared | Orange selected machine links/cab; neutral structure and terrain, orange hydraulic visualizations. |
| `gelsight_demo.py` | Shared | Orange indenter and neutral scene. The sensor algorithm retains its diagnostic RGB response; that is sensor data rather than geometry styling. |
| `grasping_force_closure.py` | N/A | Numerical grasp and friction-cone inspection. |
| `ik_stub.py` | N/A | Stubbed IK control-flow example. |
| `mj_tendon_finger.py` | Shared | Orange bunny flex and neutral press/background. The separate naming/control issues above remain. |
| `mj_xml.py` | Shared | Layered cloth uses flame, off-white, and light orange to remain distinguishable; the bag is also themed when selected. |
| `mocap_gripper_control.py` | Shared | Orange box, off-white moving gripper, charcoal floor. |
| `mocap_rod_wall_force_plot.py` | Shared | Orange rod and target markers; dark force plot uses palette colors plus distinct line styles. |
| `mocap_sensor.py` | Shared | Orange object, off-white mount marker, charcoal floor. |
| `mocap_sensor_api.py` | Shared | Orange object and object site; off-white load-cell site and neutral structure. |
| `mocap_sensor_dlo.py` | Shared | Orange cable and force arrow; neutral mount and off-white sensor site. |
| `modelling.py` | Shared | Attached robot/mesh/deformables themed after compilation. |
| `planner_rrt.py` | N/A | Numerical planner calls with a mock robot. |
| `robot.py` | Shared | Neutral robot/gripper, orange ball, charcoal floor. |
| `sim.py` | Shared | Neutral robot/gripper and charcoal floor. |
| `test.py` | Shared | Same robot palette; optional connector and ball are accented. |
| `ur10e_dmp_cartesian.py` | Shared | Orange DMP path/goal, off-white start; tracking plots use palette colors with dashed target and solid actual traces. |
| `ur5e_control.py` | Shared | Neutral robot/load cell, orange poker/ball, off-white sensor site. The robot is still a UR10e. |
| `velux_sealing.py` | Shared | Neutral articulated strip, light-orange tip marker, charcoal floor. |
| `viser.py` | Shared | Neutral G1 scene and dark/orange browser controls. |

The library's `empty_scene()` defaults and vendored XML retain their original
source colors. Every visual example applies the palette after compilation, so
those source colors are overridden before its viewer or camera renderer opens.
Numerical data, friction, mass, sensors, and constraints are unaffected by this
rendering pass. The helper also themes contact/force visualizations, preserves
transparency, and accepts explicit geometry/site/flex color roles.

## Suggested cleanup order

1. Retire the duplicated scratch robot scene and make the canonical entry points clear.
2. Correct the misleading `ur5e_control.py` and `mj_tendon_finger.py` names/descriptions; repair the bunny-press controls.
3. Extract shared weld/load-cell helpers while retaining the different sensor cases.
4. Keep the completed palette coverage when adding scenes. Use explicit roles for start/goal, actual/target, and overlapping cloth layers.
5. Keep browser overlays and plots on the shared colors; retain diagnostic sensor responses as sensor data. The current migration covers these presentation layers as well as model geometry.

Palette changes retain the existing examples and paths. The conveyor is now the
native surface-velocity luggage carousel from the requested reference; its older
parcel-actuator approximation has been replaced. The redundancy recommendations
above remain separate from this completed visual migration.
