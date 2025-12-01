import numpy as np
import spatialmath as sm

from mjsim.ctrl import DMPCartesian


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
