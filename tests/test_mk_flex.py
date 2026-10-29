"""Exercise the flex catalogue through settling, impacts, and pusher cycles."""

import importlib.util
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest


@pytest.mark.skipif(
    not hasattr(mj.MjsBody, "make_flex"), reason="requires make_flex API"
)
def test_flex_catalogue_stays_stable(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("example_mk_flex", examples / "mk_flex.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sim = module.Sim()

    # MuJoCo automatically resets unstable simulations, so finite state alone
    # is insufficient: also check warnings and that time advances all the way.
    steps = 12000
    for _ in range(steps):
        sim.step()
    assert sim.d.time == pytest.approx(steps * sim.m.opt.timestep)
    assert not np.any(sim.d.warning.number)
    assert np.isfinite(sim.d.qpos).all()
    assert np.isfinite(sim.d.qvel).all()
