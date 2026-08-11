import numpy as np
import spatialmath as sm
from scipy.spatial.transform import Rotation

from mjsim.ctrl import DMPCartesian, DMPQuaternion


def test_dmp_cartesian_rollout_shape():
    ts = np.linspace(0.0, 1.0, 20)
    traj = [sm.SE3(x, 0.0, 0.1) for x in np.linspace(0.0, 0.2, len(ts))]

    dmp = DMPCartesian(n_bfs=20)
    dmp.load(traj, dt=ts[1] - ts[0])

    p, dp, ddp, q, omega, d_omega = dmp.rollout(ts, tau=ts[-1])

    assert p.shape == (len(ts), 3)
    assert dp.shape == (len(ts), 3)
    assert ddp.shape == (len(ts), 3)
    assert len(q) == len(ts)
    assert omega.shape == (len(ts), 3)
    assert d_omega.shape == (len(ts), 3)


def test_dmp_quaternion_rollout_stays_in_same_hemisphere_near_pi():
    ts = np.linspace(0.0, 1.0, 80)
    angles = np.linspace(np.pi - 0.08, np.pi + 0.08, len(ts))
    rotations = Rotation.from_rotvec(
        np.column_stack([np.zeros_like(angles), np.zeros_like(angles), angles])
    )
    quats_xyzw = rotations.as_quat()
    quats = np.column_stack([quats_xyzw[:, 3], quats_xyzw[:, :3]])

    dmp = DMPQuaternion(n_bfs=30)
    dmp.train(quats, ts, tau=ts[-1])

    rollout = []
    x_track = dmp.cs.rollout(ts, ts[-1])
    dt = np.gradient(ts)
    for x, step_dt in zip(x_track, dt):
        quat, _, _ = dmp.step(x, step_dt, ts[-1])
        rollout.append(quat)
    rollout = np.asarray(rollout)

    dots = np.sum(rollout[1:] * rollout[:-1], axis=1)
    assert np.min(dots) > 0.0

    rotations_out = Rotation.from_quat(
        np.column_stack([rollout[:, 1:], rollout[:, 0]])
    )
    jumps = (rotations_out[1:] * rotations_out[:-1].inv()).magnitude()
    assert np.max(jumps) < 0.25
