# Robotis HX5-D20 tactile jar

The hand MJCF and all 21 STL meshes were copied from
`git@gitlab.sdu.dk:vims/robotis-model.git`, commit
`be7fd4eb89ab77186ae26d731de47fb3e26ab3a9`, directory `assets/mujoco`.
They are included locally so this example requires neither a checkout of that
repository nor Isaac Sim, Isaac Lab, or USD conversion packages.

The upstream asset was extracted from `dextrous_hand_lab_scene` commit
`0e240ea88d5a2c2f13b84df62cb1df57b570b7aa`, file
`simulation_objects/robotis_hx5d20_with_mount.usd`. Neither repository supplies
an asset license; original asset rights remain with their respective owners.
These assets are not relicensed under mjsim's MIT license.

## Local changes

- Removed the upstream floor, light, and five kinematic test probes.
- Mounted the hand palm-down above the existing Shadow Hand example's jar.
- Reduced the physics timestep from 2 ms to 1 ms and scoped the hand's joint
  and contact defaults to the hand so they do not affect the jar or lid.
- Added native joint position/velocity sensors for all 20 joints.
- Copied the jar scene and its five non-colliding mocap IK targets; assigned
  environment collision masks compatible with the Robotis hand.

Meshes, masses, inertias, joint frames and limits, actuator gains and torque
limits, and all 45 tactile site geometries and touch sensors are preserved.
Self-collision remains disabled, as upstream. Mesh collisions use convex hulls.
Actuator gains and the 1 N m torque limit are upstream demo settings, not
hardware-calibrated motor dynamics.

## Running and controls

```bash
uv run python examples/robotis_hand_jar.py
uv run python examples/robotis_hand_jar.py --ik
uv run python examples/robotis_hand_jar.py --headless
uv run python examples/robotis_hand_jar.py --headless --ik
```

C/O incrementally curl/open all fingers, including the reversed thumb flexion
joint. Use the viewer's actuator sliders to adjust individual joints, including
thumb opposition. From reset, ten C presses bring the four fingers into contact
with the lid. Up/Down move the fixed mount by 2 mm; R resets its height, joints,
controls, and sensor values. T shows/hides the sensing volumes (blue unloaded,
red at 10 N); P prints the five force grids and lid angle/velocity. The numeric
overlay always shows all 45 taxels in newtons. Color saturation affects only
visualization, not the readings.

I toggles fingertip IK, also available at startup with `--ik`. Double-click a
colored target sphere, then Ctrl + right-drag to move it; Shift switches the
translation plane. Targets track the central tactile site's position. IK uses
MINK position tasks, a posture task, joint limits, and 2 rad/s reference velocity
limits. It writes only bounded position-actuator commands. Manual curl keys
are disabled during IK and IK overwrites slider commands. Enabling or resetting
IK rebases targets on the current hand pose.

As in the Shadow Hand example, the jar is fixed and its lid has a passive,
unlimited vertical hinge with 0.02 N m s/rad damping. Ctrl + right-drag can
perturb the lid. Hand contact can rotate it; the lid is excluded from IK and
has no actuator. This is an interactive manipulation scene, with no automatic
grasp/turn sequence or screw translation. Moving the fixed mount into the jar
can impose large forces. Lid force/torque sensors include constraint and
inertial effects, not just hand contact.

## Tactile data

`FingertipTactile(model, data).forces` in `robotis_hand_jar.py` returns a copied
`(5, 3, 3)` array of native MuJoCo touch readings in newtons. The order is thumb,
index, middle, ring, little; rows run distal to proximal. Columns increase along
local X for the thumb and local Y for the other fingers. `readings(model, data)`
copies every native sensor, including `taxel_<finger>_<row>_<column>`,
`pos_finger_<finger>_joint_<joint>_right`, corresponding `vel_...` sensors, and
`lid_angle`, `lid_velocity`, `lid_force`, `lid_torque`.

These are summed normal contact forces within the original approximate sensing
zones, not shear vectors, pressure, or optical tactile images. Each grid covers
approximately 18 x 21 mm of the existing inner fingertip surface. The upstream
USD supplies no tactile hardware geometry or calibration. Contacts outside
these volumes can produce zero readings even when the fingertip touches an
object. No sensing geometry, artificial contact fixtures, or force interpolation
is added to the shipped scene.

Sensors update during MuJoCo forward dynamics. The example's `step` advances
10 ms of physics and calls `mj_forward` so the returned sensor data agrees with
the current pose. Read `data.sensor("taxel_1_1_1").data` directly if preferred.
Programmatic IK uses `robotis_hand_ik.HandIK.set_target("first", xyz)` for the
index finger (other names: `thumb`, `middle`, `ring`, `little`), with world xyz
positions in metres, followed by `step(model, data, ik)`.
