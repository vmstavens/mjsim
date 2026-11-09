# Development

Run these commands from the repository root. The project uses Python 3.12 through
`.python-version`; `uv run` uses the local editable package and its environment.

## Tests

Run pytest with the optional development dependencies:
```bash
uv run --extra dev pytest
uv run --extra dev pytest tests/test_heterogeneous_cube.py
```

## IDE stubs
Generate `typings/` stubs on demand from the root of the project where you use
`mjsim`:
```bash
mjsim-stubgen
```

When using `uv`, run the command through the project environment:
```bash
uv run mjsim-stubgen
```

By default, the command generates stubs for the importable binary dependencies
used by `mjsim`: `mujoco`, `ompl` (including its planning submodules),
`open3d`, and `mink` when available. To target specific importable modules or a
different output directory:
```bash
mjsim-stubgen mujoco ompl -o typings
```

[Documentation](../README.md) · [Main README](../../README.md)
