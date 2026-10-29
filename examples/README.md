# Examples

[Shadow Hand tactile jar](shadow_hand_jar.py): red fingertip caps, 85 STF-inspired
3D force taxels at 100 Hz, and a damped rotating lid. Run `python examples/shadow_hand_jar.py`.
Add `--ik` for MINK fingertip tracking with draggable targets.

[Closed cable knot recognition](dlo_knot.py): four capsule cables displayed
in two rows in one native MuJoCo scene: **unknot** (blue), **trefoil** (orange),
**figure-eight** (purple), and **overhand** (yellow). Each has ball joints and its own **connect
equality between endpoint sites**. No knot-selection flags are needed:

```bash
python examples/dlo_knot.py
python examples/dlo_knot.py --headless --steps 500
```

The overhand is shown with a gray **physical outer return arc** joining its
tails. With this closure it has trefoil topology, so its readout intentionally
says `detected=trefoil`. It is a different presentation of the same knot type,
not a fourth invariant. `is_target_knot(points, target="overhand")` therefore
aliases the trefoil target for closed curves. This does not add general
open-cable classification. See [MathWorld: Overhand Knot](https://mathworld.wolfram.com/OverhandKnot.html).

Each cable has a scene label and live classification readout. Press **D** to
disturb all cables, **R** to reset all, and **P** to pause/resume. Older MuJoCo
versions without text overlays print classification to the terminal. Green
markers identify the closures. Zero gravity and weak joint springs make the
initialized shapes easy to inspect; these are educational articulated cables,
not calibrated elastic rods. `--segments` controls resolution per cable
(default 48, minimum 24). Physics work is budgeted per frame to keep the viewer
responsive; larger segment counts or slower hardware may advance simulation
time slower than real time.

The reusable API accepts ordered world-space centerline vertices:

```python
from mjsim.utils.knots import is_target_knot, classify_closed_knot

matches = is_target_knot(body_positions, target="trefoil")  # Python bool
result = classify_closed_knot(body_positions)  # label, fingerprint, crossing counts
```

The last-to-first edge is implicit. The example samples cable body origins in
chain order and includes the final tip when necessary; it refuses to classify
when the physical closure gap exceeds 0.1 mm. Do not pass an open cable to this
API expecting automatic virtual closure.

Classification uses magnitudes of the Alexander polynomial at roots of unity
of orders 2, 3, and 4, requiring three projection directions to agree. The
fingerprints are `(1, 1, 1)`, `(3, 2, 1)`, and `(5, 4, 3)` respectively.
Self-intersecting/degenerate geometry and inconclusive calculations raise
`KnotClassificationError`; an unmatched fingerprint yields `result.knot=None`.
The boolean is a **fingerprint match**, not a proof of knot equivalence. Other
knots may share these values, and mirror images are intentionally grouped.
No KMT simplification or open-cable closure is included in this first example.

[Mocap overhand tying](dlo_tie_overhand.py): a native MuJoCo scene starting
with a **straight, stress-free 1.6 m cable**. Its elastic reference stays straight
throughout the trial. The green endpoint gripper remains fixed; the blue gripper
compresses and axially twists the cable to form a loop, threads through
its measured opening, then pulls through and holds the knot. No fixture,
precurved reference, intermediate body targets, or cable-position playback is
used in this default mode.

```bash
python examples/dlo_tie_overhand.py
python examples/dlo_tie_overhand.py --headless
python examples/dlo_tie_overhand.py --preformed  # earlier preformed-loop version
```

Press **P** to pause/resume or **R** to reset. After a one-second straight hold,
loop formation takes 12 simulation seconds. The next two 0.6-second motions
approach and thread the opening; a 1.5-second pull-through precedes the final
hold. Compared with the original timing, approach/insertion peak speeds are
halved and their peak accelerations are reduced by 75%. The default headless trial lasts **22 simulation seconds**. Frame budgets
keep the viewer responsive, so wall-clock time can be longer on slower machines.

The straight-start model uses 40 capsule segments and **endpoint welds** so the
mocap handles transmit orientation as well as position. It starts with zero
spring force. The working end rotates 540 degrees while the endpoint spacing
is reduced; the threading plane is estimated from the actual simulated loop.
The threading motion still uses inertia to keep the opening available briefly.
The model uses a 1 ms timestep, self-contact, and a 1 mm contact margin. This is
a tuned, zero-gravity demonstration for this cable, not a general-purpose
knot-tying planner or calibrated rope/gripper model.

The visible gray return is **virtual**, using the closed overhand example's
cosine/sinusoidal arc convention with a fixed world-space bow (0.5 m for the
larger straight-start scene). It adapts to the measured endpoints and has no
collision geometry or forces. Detection samples the actual cable positions
at **5 Hz in simulation time**, rejects intersections or return clearance below
5 mm, and requires **five consecutive valid trefoil matches**. Tracking error
above 2 mm also invalidates a sample; this can occur during threading.
Invalid samples reset the streak and turn the return red. The readout reports
current success separately from the first success timestamp. In the validated
straight-start run, stable success first appears around **16.0 seconds**.

The optional `--preformed` mode retains the earlier 16-second experiment with
position-only endpoint connections. Its startup withdraws the tip from the
existing overhand core, makes the unknotted preform its spring reference, and
settles it before the visible trial. That preparation is not used in the
default straight-start mode.

Success is conditional on the displayed closure convention. A collision-free
virtual return is not universally topology-neutral; a topology transition can
pass through an invalid closure interval. The headless command fails if the
trial does not finish with stable success. Regression tests verify the straight
initial geometry, zero initial spring force, unchanged spring references,
stationary-gripper behavior, physical segment separation during threading,
and a sustained final trefoil match. No optional knot-analysis package is needed.

<table>
  <tr>
    <td><a href="physics/models/extended/sleep/dominos.xml"><img src="media/dominos.gif" alt="dominos" width="280" height="210"></a></td>
    <td><a href="physics/models/replicate/newton_cradle.xml"><img src="media/newton_cradle.gif" alt="newton_cradle" width="280" height="210"></a></td>
    <td><a href="physics/models/extended/car/car.xml"><img src="media/car.gif" alt="car" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="physics/models/extended/plugin/sdf/gear.xml"><img src="media/gears.gif" alt="gears" width="280" height="210"></a></td>
    <td><a href="physics/models/extended/plugin/sdf/nutbolt.xml"><img src="media/nut_bolt.gif" alt="nut_bolt" width="280" height="210"></a></td>
    <td><a href="physics/models/extended/cube/cube_3x3x3.xml"><img src="media/twisty_cube.gif" alt="twisty_cube" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="physics/models/extended/flex/flag.xml"><img src="media/flag.gif" alt="flag" width="280" height="210"></a></td>
    <td><a href="physics/models/extended/flex/trampoline.xml"><img src="media/trampoline.gif" alt="trampoline" width="280" height="210"></a></td>
    <td><a href="physics/models/hammock/hammock.xml"><img src="media/hammock.gif" alt="hammock" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="physics/models/extended/flex/basket.xml"><img src="media/basket.gif" alt="basket" width="280" height="210"></a></td>
    <td><a href="physics/models/flex/jelly.xml"><img src="media/jelly.gif" alt="jelly" width="280" height="210"></a></td>
    <td><a href="physics/models/extended/flex/gripper_trilinear.xml"><img src="media/soft_gripper.gif" alt="soft_gripper" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="physics/models/extended/flex/pulley.xml"><img src="media/pulley.gif" alt="pulley" width="280" height="210"></a></td>
    <td><a href="physics/models/extended/plugin/elasticity/coil.xml"><img src="media/coil.gif" alt="coil" width="280" height="210"></a></td>
    <td><a href="physics/models/tendon_arm/arm26.xml"><img src="media/muscle_arm.gif" alt="muscle_arm" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="physics/models/balloons/balloons.xml"><img src="media/balloons.gif" alt="balloons" width="280" height="210"></a></td>
    <td><a href="physics/models/adhesion/active_adhesion.xml"><img src="media/adhesion.gif" alt="adhesion" width="280" height="210"></a></td>
    <td><a href="physics/models/extended/surfacevel/treadmill.xml"><img src="media/treadmill.gif" alt="treadmill" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="conveyor_belt.py"><img src="media/conveyor_belt.gif" alt="conveyor_belt" width="280" height="210"></a></td>
    <td><a href="particle_container.py"><img src="media/particle_container.gif" alt="particle_container" width="280" height="210"></a></td>
    <td><a href="mj_xml.py"><img src="media/mj_xml.gif" alt="mj_xml" width="280" height="210"></a></td>
  </tr>
  <tr>
    <td><a href="mocap_gripper_control.py"><img src="media/mocap_gripper_control.gif" alt="mocap_gripper_control" width="280" height="210"></a></td>
    <td><a href="velux_sealing.py"><img src="media/velux_sealing.gif" alt="velux_sealing" width="280" height="210"></a></td>
    <td><a href="physics/models/slider_crank/slider_crank.xml"><img src="media/slider_crank.gif" alt="slider_crank" width="280" height="210"></a></td>
  </tr>
</table>

[Usage](GUIDE.md) · [Sources](physics/models/UPSTREAM.md)
