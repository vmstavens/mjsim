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
