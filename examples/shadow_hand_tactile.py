"""STF-inspired 17 x 3 force estimates, not calibrated Hall-effect readings."""

import mujoco as mj
import numpy as np


class FingertipTactile:
    """Sample contact forces at 100 Hz, holding the last sample between updates.

    Each contact's force on a distal body is shared across its 17 taxels using
    normalized Gaussian distance weights. The default 5 mm kernel width is a
    modelling assumption, independent of the 2.5 mm marker radius. All contacts
    on the distal body contribute, with no hard site-volume cutoff. This
    conserves net force in world coordinates, but does not reconstruct torque
    or skin deformation. Local xyz axes are the original model's site axes.
    """

    def __init__(
        self, model, data, *, width=0.005, fingers=("ff", "mf", "rf", "lf", "th")
    ):
        if not np.isfinite(width) or width <= 0:
            raise ValueError("width must be finite and positive")
        self.model, self.data, self.width = model, data, width
        self.site_ids = {
            finger: np.array(
                [model.site(f"rh_{finger}distal_{i}").id for i in range(17)]
            )
            for finger in fingers
        }
        self.body_fingers = {model.body(f"rh_{f}distal").id: f for f in fingers}
        self.forces = {f: np.zeros((17, 3)) for f in fingers}
        self.sample_time = None
        self._next_sample = float(data.time)

    def reset(self):
        for values in self.forces.values():
            values.fill(0)
        self.sample_time = None
        self._next_sample = float(self.data.time)

    def update(self):
        """Call after physics steps. Return True when a new sample is available.

        Uses instantaneous forces at the sample time, not an interval average.
        Recomputes forward dynamics so contacts and site frames match current
        qpos, including when called immediately after mj_step.
        """
        now = float(self.data.time)
        if self.sample_time is not None and now < self.sample_time:
            self.reset()
        if now + 1e-10 < self._next_sample:
            return False
        mj.mj_forward(self.model, self.data)
        world = {f: np.zeros((17, 3)) for f in self.forces}
        wrench = np.zeros(6)
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            if contact.efc_address < 0:
                continue
            mj.mj_contactForce(self.model, self.data, idx, wrench)
            # Contact-frame rows are world axes; positive force acts on geom2.
            force = contact.frame.reshape(3, 3).T @ wrench[:3]
            for geom, sign in ((contact.geom1, -1), (contact.geom2, 1)):
                if geom < 0:
                    continue
                finger = self.body_fingers.get(self.model.geom_bodyid[geom])
                if finger is None:
                    continue
                delta = self.data.site_xpos[self.site_ids[finger]] - contact.pos
                squared = np.einsum("ij,ij->i", delta, delta)
                # Subtract the minimum to avoid underflow for remote contacts.
                weights = np.exp(-(squared - squared.min()) / (2 * self.width**2))
                weights /= weights.sum()
                world[finger] += weights[:, None] * (sign * force)
        for finger, ids in self.site_ids.items():
            rotations = self.data.site_xmat[ids].reshape(-1, 3, 3)
            self.forces[finger][:] = np.einsum("nji,nj->ni", rotations, world[finger])
        self.sample_time = now
        self._next_sample += (
            np.floor((now + 1e-10 - self._next_sample) / 0.01) + 1
        ) * 0.01
        return True

    def readings(self):
        """Copy the latest local xyz force vectors in newtons, one per taxel."""
        return {
            f"stf_rh_{finger}distal_{i}": values[i].copy()
            for finger, values in self.forces.items()
            for i in range(17)
        }
