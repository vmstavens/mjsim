# Migrating from mjsim 0.1 to 0.2

mjsim 0.2 targets MuJoCo 4.x and Python 3.11+. This guide covers environment
upgrades and the API and model changes that affect projects using mjsim 0.1.x.
See the [MuJoCo 4 release notes](https://mujoco.readthedocs.io/en/4.0.0/changelog.html)
for the upstream changes.

## Upgrade your environment

MuJoCo 4 splits its distribution into `mujoco-core` and `mujoco-studio`;
`mujoco` installs both, retaining this package's viewer and camera support.
For an existing **pip** environment, uninstall the old distribution first to
avoid files being removed during the split-package upgrade:

```bash
pip uninstall -y mujoco
pip install --upgrade mjsim
```

With uv, upgrade and verify that the installed MuJoCo version reports 4.x:

```bash
uv sync --upgrade
uv run python -c "import mujoco; print(mujoco.__version__)"
```

To upgrade `mjsim` in a consuming uv project:

```bash
uv sync --upgrade-package mjsim
```

This also refreshes a Git dependency's locked commit. The upgraded package's
minimum dependency versions in `pyproject.toml` apply to the consuming project;
this repository's `uv.lock` does not. To upgrade all dependencies allowed by
the consuming project's constraints, use `uv sync --upgrade`.

For pip installations, use `pip install --upgrade mjsim` (or the same Git URL
used to install it). These commands require the updated release or Git commit
to be available from the project's configured source.

## API and model changes

- `cable()` now creates a native 1D elastic flex. The removed cable plugin and
  `<composite>` element are no longer generated. `bend` maps to Young's modulus
  and `twist` to shear modulus, both in Pa. `n_segments` counts vertices and must
  be at least three. `points=(N, 3)` supplies a curved, stress-free reference;
  `curve` supports straight axis curves such as `"0 s 0"`.
- Vertex bodies are `<name>_<i>` and material-frame hinges are `<name>_twist_0`
  through `<name>_twist`, with `<name>` derived from `prefix.rstrip(":/")`.
  Use names or `flex_vertbodyid`, since added twist joints change state offsets.
  Endpoint site names `<prefix>S_first` and `<prefix>S_last` are preserved, but
  follow vertex translation; control twist hinges for torsional loads.
  `initial="none"` clamps the first edge, `"ball"` pins its first vertex, and
  `"free"` leaves it unpinned. Plugin `vmax` coloring and unsupported legacy
  geom/joint options now raise errors with migration guidance. Nonzero
  `joint_damping` is unsupported on native elastic vertices; use Rayleigh
  elastic `damping` (in seconds), retuning rather than copying old coefficients.
  The legacy `dqo()` cloth helper also rejects unsupported `joint_*` options.
- Parent scenes with elastic flexes should use
  `empty_scene(integrator="discrete", solver="CG", noslip_iterations=0)`.
  Standalone elastic assets select a compatible integrator. Flex edge damping
  defaults to zero because `discrete` rejects edge stiffness and damping;
  use continuum `damping` for elastic materials. Explicit nonzero edge terms
  remain available for compatible integrators.
- Save and restore `mjSTATE_INTEGRATION`, which includes `mjSTATE_PLASTIC`:
  native rods store their transported frames there even without plastic yield.
  `set_state(..., plastic=...)` also accepts explicit plastic/frame state.
- `get_contact_states()` supports native flex contacts (geom names are `None`
  on flex sides) and includes contact `surfacevel`. Geometry/body pair queries
  return only ordinary geom contacts; use the detailed states for flex IDs.
- Flex material IDs are per element; mass-matrix extraction uses
  `mj_fullM(model, data, destination)`. Replication uses the native `<replicate>`
  element. Bundled gallery models are refreshed to release 4.0.0, including
  tendon side-site placements and corrected flex elastic constants. Custom
  material parameters may need retuning under the corrected elastic formulation.

## MuJoCo 4 feature examples

```bash
uv run python examples/rod_twist.py             # Native elastic rod torsion
uv run python examples/rod_twist.py --headless  # Exact full-state replay check
uv run python examples/batch_simulation.py --nsim 64 --threads 4
uv run python examples/physics_gallery.py plastic_wire
uv run python examples/physics_gallery.py plastic_foil --original-colors
uv run python examples/physics_gallery.py plastic_torsion --original-colors
uv run python examples/physics_gallery.py speculative_flipper
```

The gallery includes all six upstream plasticity demos (`plastic_wire`,
`plastic_paper`, `plastic_foil`, `plastic_clay`, `plastic_roller`,
`plastic_torsion`) and all three speculative-contact demos (`speculative_newton`,
`speculative_chain`, `speculative_flipper`). Add `--headless --duration 2` to run
without graphics. `--original-colors` preserves upstream material distinctions
when comparing elastic and plastic objects. The CPU batching example demonstrates
live state views and checks results against a scalar loop; this API was
introduced in MuJoCo 3.17 and is included in 4.0.

Enable predictive rigid-body contacts with `empty_scene(speculative=True)`.
They predict motion from current velocities, so they do not guarantee protection
against tunneling caused by large within-step accelerations.

[Documentation](../README.md) · [Main README](../../README.md)
