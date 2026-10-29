"""Known knot fingerprints and invariance of polygonal representations."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mjsim.utils.knots import (
    FINGERPRINTS,
    KnotClassificationError,
    classify_closed_knot,
    is_target_knot,
)


def curve(name, count=80):
    t = np.linspace(0, 2 * np.pi, count, endpoint=False)
    if name == "unknot":
        return np.column_stack((np.cos(t), np.sin(t), 0 * t))
    if name == "trefoil":
        return np.column_stack(
            (
                (2 + np.cos(3 * t)) * np.cos(2 * t),
                (2 + np.cos(3 * t)) * np.sin(2 * t),
                np.sin(3 * t),
            )
        )
    return np.column_stack(
        (
            (2 + np.cos(2 * t)) * np.cos(3 * t),
            (2 + np.cos(2 * t)) * np.sin(3 * t),
            np.sin(4 * t),
        )
    )


@pytest.mark.parametrize("name", FINGERPRINTS)
def test_known_knots_and_boolean_targets(name):
    p = curve(name)
    result = classify_closed_knot(p)
    assert result.knot == name
    np.testing.assert_allclose(result.alexander_roots, FINGERPRINTS[name], atol=1e-8)
    for target in FINGERPRINTS:
        assert is_target_knot(p, target) is (name == target)


@pytest.mark.parametrize("name", FINGERPRINTS)
def test_invariance_under_affine_deformation_reversal_and_subdivision(name):
    p = curve(name)
    rng = np.random.default_rng(8)
    for _ in range(8):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        transform = rotation @ np.diag([0.6, 1.3, 2.0])
        q = np.roll(p[::-1], 17, axis=0) @ transform + [7, -2, 4]
        assert is_target_knot(q, name)
        p = p @ rotation
    subdivided = np.empty((2 * len(p), 3))
    subdivided[::2] = p
    subdivided[1::2] = (p + np.roll(p, -1, axis=0)) / 2
    assert is_target_knot(np.vstack((subdivided, subdivided[0])), name)
    assert is_target_knot(p * [-1, 1, 1], name)  # mirrors intentionally grouped
    assert is_target_knot(p * 1e-6, name)


def test_determinant_alone_would_confuse_cinquefoil_and_figure_eight():
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    p = np.column_stack(
        (
            (2 + np.cos(5 * t)) * np.cos(2 * t),
            (2 + np.cos(5 * t)) * np.sin(2 * t),
            np.sin(5 * t),
        )
    )
    result = classify_closed_knot(p)
    assert result.alexander_roots[0] == pytest.approx(5)
    assert result.knot is None
    assert is_target_knot(p, "figure-eight") is False


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]],  # self-crossing
        [[0, 0, 0], [1, 0, 0], [0.5, 0, 0], [0, 1, 0]],  # backtrack
        [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, np.nan, 0]],
        [[0, 0], [1, 0], [0, 1]],
    ],
)
def test_invalid_geometry_does_not_return_false(points):
    with pytest.raises(KnotClassificationError):
        is_target_knot(points)


def test_crossing_budget_and_unknown_target():
    with pytest.raises(KnotClassificationError, match="crossing limit"):
        classify_closed_knot(curve("trefoil"), max_crossings=1)
    with pytest.raises(ValueError, match="Unknown target"):
        is_target_knot(curve("unknot"), "not-a-knot-name")


@pytest.mark.parametrize("name", FINGERPRINTS)
def test_simulated_closed_cable_after_disturbance(name):
    pytest.importorskip("mujoco")
    path = Path(__file__).resolve().parents[1] / "examples" / "dlo_knot.py"
    spec = importlib.util.spec_from_file_location("dlo_knot", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    sim = example.KnotSim(name, count=64)
    initial = sim.centerline()
    assert is_target_knot(initial, name)
    assert sim.model.neq == 1
    sim.disturb()
    for _ in range(300):
        sim.step()
    assert sim.closure_gap() < 1e-4
    assert np.max(np.abs(sim.data.xpos[sim.body_ids] - initial)) > 1e-6
    assert is_target_knot(sim.centerline(), name)
    assert not any(warning.number for warning in sim.data.warning)
    sim.reset()
    np.testing.assert_allclose(sim.centerline(), initial, atol=1e-12)


def test_all_knots_share_one_scene_and_remain_independent():
    mj = pytest.importorskip("mujoco")
    path = Path(__file__).resolve().parents[1] / "examples" / "dlo_knot.py"
    spec = importlib.util.spec_from_file_location("dlo_knot", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    sim = example.KnotSim()
    assert sim.model.neq == 4
    assert tuple(sim.knots) == tuple(example.EXAMPLE_TARGETS)
    initial = {name: sim.centerline(name) for name in sim.knots}
    assert len(set(np.concatenate(list(sim.cable_body_ids.values())))) == 4 * 48
    sim.disturb()
    # Disturb internal joints without accidentally kicking any free root.
    for kind, address in zip(sim.model.jnt_type, sim.model.jnt_dofadr):
        if kind == mj.mjtJoint.mjJNT_FREE:
            np.testing.assert_array_equal(sim.data.qvel[address : address + 6], 0)
    for _ in range(300):
        sim.step()
    for name in sim.knots:
        assert is_target_knot(sim.centerline(name), example.EXAMPLE_TARGETS[name])
        assert sim.closure_gap(name) < 1e-4
    assert not any(warning.number for warning in sim.data.warning)
    scene = mj.MjvScene(sim.model, maxgeom=10)
    sim.add_labels(scene)
    assert scene.ngeom == 4
    assert [scene.geoms[i].label for i in range(4)] == [
        "unknot",
        "trefoil",
        "figure-eight",
        "overhand (closed)",
    ]
    sim.reset()
    for name in sim.knots:
        np.testing.assert_allclose(sim.centerline(name), initial[name], atol=1e-12)


@pytest.mark.parametrize("count", [24, 48, 96])
def test_closed_overhand_is_trefoil_at_supported_resolutions(count):
    pytest.importorskip("mujoco")
    path = Path(__file__).resolve().parents[1] / "examples" / "dlo_knot.py"
    spec = importlib.util.spec_from_file_location("dlo_knot", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    points = example.knot_points("overhand", count)
    assert classify_closed_knot(points).knot == "trefoil"
    assert is_target_knot(points, "overhand")
    assert not is_target_knot(curve("unknot"), "overhand")
