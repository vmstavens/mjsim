# Upstream MuJoCo models

The seven physics gallery models are vendored, unmodified, from
[google-deepmind/mujoco, release 3.3.7](https://github.com/google-deepmind/mujoco/tree/3.3.7/model),
commit `f1d45bd5422c74beddfb0d1deb590a02583d21de`.
They work offline and require no downloaded meshes or robot descriptions.

The directory structure preserves upstream relative includes and model attachments:

| Gallery entry | Upstream file under `model/` | Supporting files |
| --- | --- | --- |
| `newton_cradle` | `replicate/newton_cradle.xml` | — |
| `balloons` | `balloons/balloons.xml` | — |
| `slider_crank` | `slider_crank/slider_crank.xml` | — |
| `muscle_arm` | `tendon_arm/arm26.xml` | — |
| `adhesion` | `adhesion/active_adhesion.xml` | — |
| `hammock` | `hammock/hammock.xml` | `humanoid/humanoid.xml` |
| `jelly` | `flex/jelly.xml` | `flex/scene.xml` |

These files are licensed under the Apache License 2.0, included in [LICENSE](LICENSE).
Existing upstream copyright and license comments are retained. This license applies
to the vendored files; the rest of mjsim retains its existing license.

The Python gallery adds demonstration actuator inputs and applies a rendering-only
palette at load time. On newer MuJoCo versions exposing `mjINT_DISCRETE`, its loader
switches jelly to that integrator, as required for implicit flex elasticity. On
3.3.7 it uses the original `implicitfast` setting. The XML files stay unchanged.

## Luggage carousel (MuJoCo 3.11+)

`surfacevel/carousel.xml` is vendored unchanged from
[MuJoCo release 3.14.0](https://github.com/google-deepmind/mujoco/blob/3.14.0/model/surfacevel/carousel.xml),
under the same Apache 2.0 license above. It is the luggage-carousel surface-velocity
demonstration associated with the [reference video](https://www.youtube.com/watch?v=PdSdrqhSiZA).
It has no external assets. The bundled model was compiled and simulated with
MuJoCo 3.11.0; the standalone conveyor launcher checks for native `surfacevel`
support and applies only the palette and user-selected surface speeds.

## Expanded gallery (MuJoCo 3.11+)

The `extended/` directory is vendored unchanged from
[MuJoCo release 3.11.0](https://github.com/google-deepmind/mujoco/tree/3.11.0/model),
commit `b85fdca54f0e0038b804af146a0b4e94199e00d0`, under the same Apache 2.0
license above. It preserves the original directory structure and all required
includes, the gripper cap mesh, and cube textures. No runtime downloads are needed.
The current main catalogue was reviewed at commit
`897e3fcb19bd85623c8dffa0d645255c00cd7931` when selecting these systems.

| Gallery name | File beneath `extended/` |
| --- | --- |
| `car` | `car/car.xml` |
| `dominos` | `sleep/dominos.xml` |
| `flag` | `flex/flag.xml` |
| `trampoline` | `flex/trampoline.xml` |
| `soft_gripper` | `flex/gripper_trilinear.xml` |
| `pulley` | `flex/pulley.xml` |
| `basket` | `flex/basket.xml` |
| `coil` | `plugin/elasticity/coil.xml` |
| `gears` | `plugin/sdf/gear.xml` |
| `nut_bolt` | `plugin/sdf/nutbolt.xml` |
| `treadmill` | `surfacevel/treadmill.xml` |
| `twisty_cube` | `cube/cube_3x3x3.xml` |

Python adds wheel/face motor inputs, gripper and coil trajectories, a pulley
keyframe, and an initial nut spin. It also applies the shared palette and camera
framing. In themed coil views, Python disables the cable plugin’s strain heatmap
(`vmax=0`) so it cannot replace the palette during rendering. This changes only
visualization. The source XML and image assets remain unmodified. On versions exposing
`mjINT_DISCRETE`, the soft-gripper loader selects that integrator for implicit
flex elasticity, matching the jelly compatibility handling.
