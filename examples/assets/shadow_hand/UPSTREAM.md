# Shadow Hand E3M5 Description (MJCF)

## Local tactile variant

Copied from `~/git/gitlab/mj_sim/assets/shadow_hand/shadow_hand.xml` and its
referenced `new_assets` meshes. This local Menagerie-derived variant has red
`f_distal_mstXL` / `th_distal_mst` caps, 17 oriented sites on each fingertip,
and 24 independent position actuators. The accompanying Apache-2.0 license
is preserved in `LICENSE`.

For this example, `hand.xml` changes the fixed mount pose, uses a 1 ms timestep
and implicitfast integration, sets the 85 taxel markers to 2.5 mm radius (5 mm
diameter, based on the user's estimate), and adds joint position/velocity sensors.
Finger site positions/orientations and the hand meshes are preserved. Thumb
taxel transforms are corrected as described below. The previous
scalar touch sensors are replaced by Python-computed STF-inspired force vectors.

The original local model incorrectly reused the finger taxel transforms on the
larger thumb mesh. The 17 `rh_thdistal_0` through `rh_thdistal_16` sites now use
Shadow Robot's thumb-specific transforms from `th_distal_mst.urdf.xacro`, copied
alongside the model with its BSD license notice intact. Source:
`~/git/urdf2mjcf/urdf2mjcf/sr_common/sr_description/hand/xacro/thumb/th_distal_mst.urdf.xacro`.
Positions use scale 0.001 (millimetres to metres); URDF fixed-axis RPY is converted
to MuJoCo wxyz quaternions using `Rz(yaw) @ Ry(pitch) @ Rx(roll)`. The thumb mesh,
body pose, 2.5 mm marker radii, and separate IK tip site are unchanged. These
taxel centers remain slightly below the mesh surface, as in Shadow's definition.

`scene.xml` adds a fixed solid cylindrical jar and a smaller lid with one
unlimited vertical hinge, damping of 0.02 N m s/rad, and no screw translation.
The lid has angle, angular velocity, and parent-child force/torque sensors;
the latter include constraint and inertial effects, not just hand contact.

Run `python examples/shadow_hand_jar.py` from the repository root. The hand
starts open with the lid centered beneath the palm and clearance for the fingers
to curl down around its sides. Use Up/Down to move the fixed mount by 2 mm,
C/O to curl/open the four fingers, and the viewer actuator sliders for individual
joints including the thumb. T toggles sensing sites; green indicates contact.
P prints all 85 local xyz force vectors in newtons and the lid angle/velocity in radians
and radians/second. R resets the pose and controls. Ctrl + right-drag perturbs
the lid. Moving the fixed mount is an inspection aid and can impose large forces
if moved deeply into the jar. No grasp or lid-turning controller is included.

The Python `readings(model, data, tactile)` function combines native sensors and
the latest tactile sample. The custom taxels are not stored in `data.sensordata`.
Use `--headless` for a one-second smoke check without a display.

### STF-inspired force model

`shadow_hand_tactile.FingertipTactile` provides `forces[finger]`, a (17, 3)
array for each of `ff`, `mf`, `rf`, `lf`, `th`, and named readings such as
`stf_rh_ffdistal_0`. Values are signed forces **on the fingertip**, expressed in
the original site's local xyz axes, in newtons. These axes have not been
calibrated against the hardware; z is not asserted to be its normal axis.

Every active geom contact on a distal body contributes its normal and tangential
force, obtained with `mj_contactForce`. For each contact, weights are proportional
to `exp(-distance_to_taxel_squared / (2 * width**2))`, normalized across the
17 sites on that body. The configurable default width is 5 mm, an assumed
spatial response rather than a hardware specification. Marker radius does not
gate sensing. Contacts between markers are still detected, and transforming the
17 vectors back to world coordinates and summing conserves the net contact
force. Torques and a deformable skin response are not reconstructed. The model
also includes distal-body self-contact and cannot distinguish sensitive skin
from other areas on that same body.

Call `tactile.update()` after stepping physics, at least every 10 ms. It samples
instantaneous force at 100 Hz of simulation time and holds the last sample;
`sample_time` records its timestamp. It recomputes forward dynamics to align
contacts with the current site frames. Missed samples are not reconstructed.
Call `reset()` after manual pose edits; simulation time rewinds reset automatically.

```python
from shadow_hand_tactile import FingertipTactile

tactile = FingertipTactile(model, data)
for _ in range(1000):
    mujoco.mj_step(model, data)
    if tactile.update():
        index_forces = tactile.forces["ff"].copy()  # (17, 3), local xyz in N
```

Shadow's [STF specifications](https://shadowrobot.com/sensors/) describe 17 x 3
Hall-effect responses at 100 Hz. This approximation reports forces, not raw
12-bit Hall readings. No speculative saturation, threshold, noise or quantization
is imposed; hardware calibration data is needed to reproduce those responses.

### MINK fingertip IK

Run `python examples/shadow_hand_jar.py --ik`. This uses the five position-only
FrameTasks plus a posture task approach from `~/git/mink/examples/hand_shadow.py`.
The five colored mocap spheres are targets for thumb, first (index), middle,
ring, and little fingers. Double-click a sphere and Ctrl + right-drag to move it;
Shift changes the translation plane. Press I to toggle IK. Enabling IK, resetting,
or changing mount height rebases targets to the current fingertips. Manual curl
keys are disabled during IK, and IK overwrites the hand actuator sliders.

`shadow_hand_ik.HandIK` exposes `set_target(finger, xyz)` for a future input
signal, in world coordinates and metres. Call it on the simulation thread, then
`ik.step(dt)` once per control interval and step MuJoCo normally. For example:

```python
from shadow_hand_jar import build_model
from shadow_hand_ik import HandIK
import mujoco

model, data = build_model()
ik = HandIK(model, data)
target = data.site("rh_ffdistal_tip").xpos.copy()
target[2] -= 0.015
ik.set_target("first", target)
for _ in range(300):
    ik.step(0.01)
    for _ in range(10):
        mujoco.mj_step(model, data)
```

The solver uses `daqp`, joint limits, a 2 rad/s reference velocity limit, and a
separate kinematic reference configuration. It writes the 24 position actuator
commands, never the simulation's qpos/qvel. The lid DOF is frozen only inside
the IK problem and remains passive in the physical simulation. Targets can be
unreachable; physical tracking can differ from the reference under contact or
actuator force limits. Orientation tracking, collision avoidance, an automatic
grasp/turn sequence, and an external signal transport are not implemented.

## Original upstream documentation

Requires MuJoCo 2.2.2 or later.

## Overview

This package contains assets of the "E3M5" version of the Shadow Hand robot,
including both right-handed and left-handed versions.
The original URDF and assets were provided directly by
[Shadow Robot Company](https://www.shadowrobot.com/) under the
[Apache 2.0 License](LICENSE).

<p float="left">
  <img src="shadow_hand.png" width="400">
</p>

## URDF → MJCF derivation steps

1. Converted the DAE [mesh
   files](https://github.com/shadow-robot/sr_common/tree/noetic-devel/sr_description/meshes/)
   to OBJ format using [Blender](https://www.blender.org/).
2. Processed `.obj` files with [`obj2mjcf`](https://github.com/kevinzakka/obj2mjcf).
3. Added `<mujoco> <compiler discardvisual="false"/> </mujoco>` to the
   [URDF](https://github.com/shadow-robot/sr_common/blob/noetic-devel/sr_description/hand/xacro/forearm/forearm_e.urdf.xacro)'s
   `<robot>` clause in order to preserve visual geometries.
4. Loaded the URDF into MuJoCo and saved a corresponding MJCF.
5. Removed `_E3M5` suffix from mesh names.
6. Added forearm body and its corresponding inertial specs.
7. Removed 2 artifact boxes left from the URDF conversion.
8. Manually edited the MJCF to extract common properties into the `<default>` section.
9. Added `<exclude>` clauses to prevent collisions between the forearm and the
    wrist and thumb bodies.
10. Added position-controlled actuators.
11. Added `impratio=10` for better noslip.
12. Hardened the contacts on the hand geoms.
13. Added `scene_left.xml` and `scene_right.xml` which include the robot, with
    an object, textured groundplane, skybox, and haze.

## License

These models are released under an [Apache-2.0 License](LICENSE).
