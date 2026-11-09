"""Exercise migrated embedded models and the drape's settling behavior."""

import importlib.util
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _example(name, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(
        f"migration_{name}", EXAMPLES / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frictional_cloths_settle_on_sphere(tmp_path, monkeypatch):
    demo = _example("mj_xml", monkeypatch)
    monkeypatch.chdir(tmp_path)
    sim = demo.Sim("drape")
    model, data = sim.model, sim.data
    assert not np.any(model.flex_passive)
    assert np.all(model.flex_condim == 3)
    model.opt.enableflags |= mj.mjtEnableBit.mjENBL_ENERGY
    mj.mj_step(model, data, nstep=round(3 / model.opt.timestep))
    mj.mj_forward(model, data)
    assert data.time == pytest.approx(3)
    assert not np.any(data.warning.number)
    assert np.isfinite(data.qpos).all()
    assert data.energy[1] < 1e-4  # Settling, not merely finite or warning-free.
    for start, count in zip(model.flex_vertadr, model.flex_vertnum):
        vertices = data.flexvert_xpos[start : start + count]
        assert 0.35 < vertices[:, 2].mean() < 0.6
        assert vertices[:, 2].min() > 0.05  # Each sheet remains on the sphere.


@pytest.mark.parametrize("name,scene", [("mj_xml", "bags"), ("mj_tendon_finger", None)])
def test_embedded_elastic_models_compile_and_step(name, scene, tmp_path, monkeypatch):
    demo = _example(name, monkeypatch)
    monkeypatch.chdir(tmp_path)
    sim = demo.Sim(scene) if scene else demo.Sim()
    model, data = sim.model, sim.data
    assert model.opt.integrator == mj.mjtIntegrator.mjINT_DISCRETE
    mj.mj_step(model, data, nstep=20)
    assert data.time == pytest.approx(20 * model.opt.timestep)
    assert not np.any(data.warning.number)


@pytest.mark.parametrize(
    "name", ["ik_stub", "controller_dmp", "grasping_force_closure"]
)
def test_numerical_examples_use_current_package_apis(name, monkeypatch):
    _example(name, monkeypatch).main()


def test_viser_launcher_resolves_installed_package():
    import os
    import subprocess
    import sys

    code = f"""
import runpy
module = runpy.run_path({str(EXAMPLES / "viser.py")!r})
assert callable(module['viser'].ViserServer)
assert callable(module['Viewer'])
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=EXAMPLES,
        check=True,
        env={**os.environ, "MJSIM_SKIP_STUBGEN": "1"},
        timeout=30,
    )
