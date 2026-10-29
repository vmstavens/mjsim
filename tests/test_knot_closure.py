import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mjsim.utils.knot_closure import (
    ClosureError,
    KnotProgress,
    close_with_outer_arc,
    outer_return_arc,
)
from mjsim.utils.knots import _segment_distances, classify_closed_knot


def load_example(monkeypatch, filename):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location(filename, examples / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outer_return_joins_endpoints_and_closes_unknot():
    theta = np.linspace(0, np.pi, 20)
    points = np.column_stack((np.cos(theta), np.sin(theta), 0 * theta))
    original = points.copy()
    polygon, arc = close_with_outer_arc(points)
    np.testing.assert_array_equal(points, original)
    np.testing.assert_array_equal(arc[0], points[-1])
    np.testing.assert_array_equal(arc[-1], points[0])
    assert classify_closed_knot(polygon).knot == "unknot"


def test_straight_segmented_cable_is_not_a_degenerate_projection():
    points = np.zeros((41, 3))
    points[:, 0] = np.linspace(0, 1.6, len(points))
    polygon, _ = close_with_outer_arc(points)
    rng = np.random.default_rng(19)
    for _ in range(5):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        assert classify_closed_knot(polygon @ rotation + [2, -3, 7]).knot == "unknot"


@pytest.mark.parametrize("depth", [-0.3, -0.3001])
def test_intersecting_or_nearby_virtual_return_is_rejected(depth):
    points = np.array([[-1, 0, 0], [0, depth, 0], [1, 0, 0]])
    with pytest.raises(ClosureError):
        close_with_outer_arc(points)


def test_bad_closure_parameters():
    points = [[0, 0, 0], [1, 1, 0], [2, 0, 0]]
    with pytest.raises(ValueError, match="bow"):
        outer_return_arc(points, bow=(0, 0, 0))
    with pytest.raises(ValueError, match="clearance"):
        close_with_outer_arc(points, clearance=-1)
    with pytest.raises(ValueError, match="segments"):
        outer_return_arc(points, segments=1)
    with pytest.raises(ClosureError, match="finite"):
        outer_return_arc([[0, 0, 0], [1, np.nan, 0], [2, 0, 0]])


def test_five_valid_matches_and_invalid_samples_reset_streak(monkeypatch):
    example = load_example(monkeypatch, "dlo_knot")
    points = example.knot_points("overhand", 68)[: example.overhand_core_count(68)]
    progress = KnotProgress()
    for i in range(4):
        progress.sample(points, i * 0.2)
        assert not progress.successful
    progress.sample(points, 0.8)
    assert progress.successful
    assert progress.first_success_time == pytest.approx(0.8)
    progress.sample(points, 1.0, tracking_error=0.01)
    assert not progress.successful
    assert progress.error
    assert progress.streak == 0
    assert progress.first_success_time == pytest.approx(0.8)
    progress.sample(points, 1.2)
    assert not progress.successful
    progress.sample([[-1, 0, 0], [0, -0.3, 0], [1, 0, 0]], 1.4)
    assert progress.error
    assert progress.streak == 0


def test_mocap_threading_starts_unknotted_and_finishes_stably_knotted(monkeypatch):
    mj = pytest.importorskip("mujoco")
    example = load_example(monkeypatch, "dlo_tie_overhand")
    sim = example.OverhandTyingSim()
    assert sim.model.nmocap == 2
    assert sim.model.neq == 2
    assert sim.progress.label == "unknot"
    assert sim.progress.error is None
    initial = sim.centerline()
    rest_shape = sim.model.qpos_spring.copy()

    # Holding both handles must not let spring relaxation tie the knot for us.
    for _ in range(round(4 / sim.timestep)):
        sim.step(drive=False)
    assert all(
        label == "unknot" and error is None for _, label, error, _ in sim.history
    )
    assert sim.progress.first_success_time is None
    sim.reset()
    np.testing.assert_allclose(sim.centerline(), initial, atol=1e-12)

    held_target = sim.data.mocap_pos[sim.mocap_ids[0]].copy()
    min_clearance = np.inf
    for _ in range(round(sim.duration / sim.timestep)):
        sim.step()
        if sim.steps % sim.sample_steps == 0:
            # Nonadjacent physical segments remain separated at detector samples.
            p = sim.centerline()
            i, j = np.triu_indices(len(p) - 1, 2)
            min_clearance = min(
                min_clearance, _segment_distances(p[i], p[i + 1], p[j], p[j + 1]).min()
            )
    assert min_clearance > 1.8 * sim.radius
    np.testing.assert_array_equal(sim.data.mocap_pos[sim.mocap_ids[0]], held_target)
    np.testing.assert_array_equal(sim.model.qpos_spring, rest_shape)
    assert sim.progress.successful
    assert sim.progress.label == "trefoil"
    assert sim.progress.first_success_time > 1.0
    assert all(
        label == "trefoil" and error is None and success
        for t, label, error, success in sim.history
        if t >= 12
    )
    assert sim.tracking_error() < 0.002
    assert not any(warning.number for warning in sim.data.warning)

    scene = mj.MjvScene(sim.model, maxgeom=100)
    sim.draw_closure(scene)
    assert scene.ngeom == 24
    sim.reset()
    assert sim.progress.first_success_time is None
    assert sim.progress.streak == 0
    assert sim.progress.label == "unknot"


def test_straight_stress_free_cable_ties_using_only_endpoint_poses(monkeypatch):
    mj = pytest.importorskip("mujoco")
    example = load_example(monkeypatch, "dlo_tie_overhand")
    sim = example.StraightOverhandTyingSim()
    initial = sim.centerline()
    np.testing.assert_allclose(initial[:, 0], np.linspace(0, 1.6, 41), atol=1e-12)
    np.testing.assert_allclose(initial[:, 1:], np.tile([0, 0.6], (41, 1)), atol=1e-12)
    np.testing.assert_array_equal(sim.data.qpos, sim.model.qpos_spring)
    np.testing.assert_allclose(sim.data.qfrc_passive, 0, atol=1e-12)
    assert sim.model.nmocap == 2
    np.testing.assert_array_equal(sim.model.eq_type, mj.mjtEq.mjEQ_WELD)
    assert sim.progress.label == "unknot" and sim.progress.error is None
    assert sim.loop_center is None

    # No hidden rest curvature: a stationary trial remains a straight cable.
    for _ in range(round(2 / sim.timestep)):
        sim.step(drive=False)
    np.testing.assert_allclose(sim.centerline(), initial, atol=1e-9)
    assert all(
        label == "unknot" and error is None for _, label, error, _ in sim.history
    )
    sim.reset()

    reference = sim.model.qpos_spring.copy()
    fixed_position = sim.data.mocap_pos[sim.mocap_ids[0]].copy()
    fixed_rotation = sim.data.mocap_quat[sim.mocap_ids[0]].copy()
    threading_end = 13 + sim.approach_duration + sim.thread_duration + sim.pull_duration
    minimum_distance = np.inf
    i, j = np.triu_indices(40, 2)
    for _ in range(round(sim.duration / sim.timestep)):
        sim.step()
        # Check every integrator step during threading, and regularly
        # elsewhere. Contacts may act before surfaces meet, via geom_margin.
        if 13 <= sim.data.time <= threading_end or sim.steps % 100 == 0:
            p = sim.centerline()
            minimum_distance = min(
                minimum_distance,
                _segment_distances(p[i], p[i + 1], p[j], p[j + 1]).min(),
            )
    assert minimum_distance > 2 * sim.radius
    np.testing.assert_array_equal(sim.model.qpos_spring, reference)
    np.testing.assert_array_equal(sim.data.mocap_pos[sim.mocap_ids[0]], fixed_position)
    np.testing.assert_array_equal(sim.data.mocap_quat[sim.mocap_ids[0]], fixed_rotation)
    assert sim.loop_center is not None
    assert sim.progress.successful
    assert sim.progress.label == "trefoil"
    assert sim.progress.first_success_time > 13 + sim.approach_duration
    assert any(error and "tracking" in error for _, _, error, _ in sim.history)
    assert all(
        label == "trefoil" and error is None and success
        for t, label, error, success in sim.history
        if t >= threading_end + 2
    )
    assert sim.tracking_error() < 0.002
    assert not any(warning.number for warning in sim.data.warning)
    sim.reset()
    np.testing.assert_allclose(sim.centerline(), initial, atol=1e-12)
    assert sim.progress.first_success_time is None
    assert sim.loop_center is None
