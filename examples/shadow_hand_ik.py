"""MINK fingertip tracking for the physical Shadow Hand jar example.

Follows the FrameTask/PostureTask approach in mink/examples/hand_shadow.py.
Targets are world-frame positions in metres. Only actuator commands are written;
MuJoCo retains ownership of the physical joint states and the passive lid.
"""

import mink
import mujoco as mj
import numpy as np


FINGERS = dict(thumb="th", first="ff", middle="mf", ring="rf", little="lf")


class HandIK:
    def __init__(self, model, data, solver="daqp"):
        self.model, self.data, self.solver = model, data, solver
        self.configuration = mink.Configuration(model)
        self.posture = mink.PostureTask(model, cost=1e-3)
        self.fingers = {
            finger: mink.FrameTask(
                frame_name=f"rh_{code}distal_tip",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            for finger, code in FINGERS.items()
        }
        self.tasks = [self.posture, *self.fingers.values()]
        self.actuator_ids = np.array(
            [i for i in range(model.nu) if model.actuator(i).name.startswith("rh_A_")]
        )
        joints = model.actuator_trnid[self.actuator_ids, 0]
        self.qpos_ids = model.jnt_qposadr[joints]
        self.limits = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(model, {model.joint(j).name: 2.0 for j in joints}),
        ]
        # The lid is a physical degree of freedom, never an IK decision variable.
        self.constraints = [
            mink.DofFreezingTask(
                model, [int(model.jnt_dofadr[model.joint("lid_hinge").id])]
            )
        ]
        self.reset()

    def reset(self):
        """Rebase posture and targets on the current physical hand pose."""
        mj.mj_forward(self.model, self.data)
        q = self.data.qpos.copy()
        for j in range(self.model.njnt):
            if self.model.jnt_limited[j]:
                idx = self.model.jnt_qposadr[j]
                q[idx] = np.clip(q[idx], *self.model.jnt_range[j])
        self.configuration.update(q)
        self.posture.set_target_from_configuration(self.configuration)
        for finger, code in FINGERS.items():
            mink.move_mocap_to_frame(
                self.model,
                self.data,
                f"{finger}_target",
                f"rh_{code}distal_tip",
                "site",
            )

    def set_target(self, finger, position):
        """Set one fingertip's world position; call from the simulation thread."""
        if finger not in FINGERS:
            raise ValueError(f"Unknown finger {finger!r}; choose from {tuple(FINGERS)}")
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Target must be a finite xyz position in metres")
        mocap = self.model.body(f"{finger}_target").mocapid[0]
        self.data.mocap_pos[mocap] = position

    def step(self, dt):
        """Advance the IK reference and write bounded position actuator targets."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        # Recompute FK, including any edits to the fixed hand mount.
        self.configuration.update(self.configuration.q)
        for finger, task in self.fingers.items():
            task.set_target(
                mink.SE3.from_mocap_name(self.model, self.data, f"{finger}_target")
            )
        velocity = mink.solve_ik(
            self.configuration,
            self.tasks,
            dt,
            self.solver,
            damping=1e-5,
            limits=self.limits,
            constraints=self.constraints,
        )
        self.configuration.integrate_inplace(velocity, dt)
        bounds = self.model.actuator_ctrlrange[self.actuator_ids]
        self.data.ctrl[self.actuator_ids] = np.clip(
            self.configuration.q[self.qpos_ids], bounds[:, 0], bounds[:, 1]
        )
