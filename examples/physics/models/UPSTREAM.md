# Upstream MuJoCo models

The bundled gallery XML models and their supporting assets are copied from
[google-deepmind/mujoco release 4.0.0](https://github.com/google-deepmind/mujoco/tree/4.0.0/model).
The repository layout preserves relative includes, attachments, and asset paths.
The models run offline without robot-description downloads.

All vendored files retain their upstream copyright notices and are covered by
the Apache License 2.0 in [LICENSE](LICENSE). The rest of mjsim retains its own
license. The Python loader adds demonstration actuator inputs, initial conditions,
and an optional rendering palette; physics parameters come from the release XML.

| Local directory | Upstream directory under `model/` |
| --- | --- |
| `replicate`, `balloons`, `slider_crank`, `tendon_arm`, `adhesion` | Same path |
| `hammock`, `humanoid`, `flex` | Same path |
| `surfacevel` | `surfacevel` |
| `extended/car`, `extended/sleep`, `extended/flex`, `extended/cube` | Corresponding directory without `extended/` |
| `extended/plugin/sdf` | `plugin/sdf` |
| `plastic` | `plastic` (all six release examples and README) |
| `speculative` | `speculative` (all three release examples and README) |

The coil model moved from `extended/plugin/elasticity/coil.xml` to
`extended/flex/coil.xml`, following the upstream migration from the removed cable
plugin to native 1D flex elasticity. Gallery commands still use the name `coil`.
The unused historical `extended/plugin/elasticity/scene.xml` is retained from
MuJoCo 3.11.0 for old relative links; it is not loaded by the gallery.

The 4.0 refresh also includes updated tendon side sites, flex elastic constants,
integrators, and solver choices. Earlier previews in `examples/media` were
rendered with older versions and may differ slightly from current simulation.
