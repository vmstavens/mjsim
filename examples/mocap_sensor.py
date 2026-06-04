import numpy as np
import mujoco as mj
import mujoco.viewer

from mjsim.utils.mjs import empty_scene


class Sim:
    def __init__(self):
        self.m, self.d = self._build_model()

        self.force_sensor_adr = self.m.sensor("force").adr[0]
        self.ball_free_dofadr = self.m.joint("ball_free_joint").dofadr[0]
        self.ball_body_id = self.m.body("ball").id
        self.ball_site_id = self.m.site("ball_site").id

        self.expected_force_z = (
            -float(self.m.body("ball").mass[0]) * self.m.opt.gravity[2]
        )
        self.expected_torque_y = (
            -self.m.site_pos[self.ball_site_id][0] * self.expected_force_z
        )

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData]:
        """Build a toy model for measuring weld support loads.

        The ball geom is given ``mass=1000``. MuJoCo then computes the body's
        inertial properties from the box shape, so the body has the requested mass
        without us manually specifying the inertia tensor.

        The box geom is centered at the body frame. With this setup, the site is
        0.2 m from the center of mass along local +x, so the nominal static
        support load is:

            Fz = m * g = 1000 kg * 9.82 m/s^2 = 9820 N

        and the nominal moment about the center of mass is:

            tau = r x F = [0.2, 0, 0] x [0, 0, 9820] = [0, -1964, 0] N*m
        """
        scene = empty_scene()

        m_body = scene.worldbody.add_body(name="mocap", mocap=True, pos=[0, 0, 0.3])
        m_geom = m_body.add_geom(
            name="mocap_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.05, 0.05, 0.05],
            contype=0,
            conaffinity=0,
        )
        s_mocap = m_body.add_site(
            name="mocap_site",
            pos=[0, 0, 0],
            size=[0.05],
            rgba=[1, 0, 0, 1],
        )

        b_ball = scene.worldbody.add_body(
            name="ball",
            pos=[0, 0, 0.1],
        )
        g_ball = b_ball.add_geom(
            name="ball_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.2, 0.02, 0.02],
            mass=1000,
        )
        s_ball = b_ball.add_site(
            name="ball_site",
            pos=[0.2, 0, 0],
            # size=[0.01],
            rgba=[0, 1, 0, 1],
        )
        b_ball.add_joint(
            name="ball_free_joint",
            type=mj.mjtJoint.mjJNT_FREE,
        )

        scene.add_equality(
            name="weld",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site",
            name2="ball_site",
            solref=[0.001, 3],
        )

        scene.add_sensor(
            name="force",
            type=mj.mjtSensor.mjSENS_FORCE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="ball_site",
            dim=3,
        )

        m = scene.compile()
        d = mj.MjData(m)

        return m, d

    def cb(self, key: int) -> None:
        pass

    def site_force_sensor(self):
        """Return the raw MuJoCo site force sensor reading.

        A ``mjSENS_FORCE`` sensor does not measure "the force applied at this
        site". It measures the interaction force between the body containing the
        site and that body's parent, expressed in the site frame.

        In this model, ``ball`` is a free body whose parent is ``world``. The weld
        equality connects ``ball_site`` to ``mocap_site`` through a constraint, not
        through a parent-child body joint. Therefore this sensor is expected to be
        near zero even when the equality constraint is carrying the gravity load.
        """
        return self.d.sensordata[self.force_sensor_adr : self.force_sensor_adr + 3]

    def ball_constraint_wrench(self):
        """Return the generalized constraint wrench on the ball free joint.

        The returned vector is:

            [Fx, Fy, Fz, tau_x, tau_y, tau_z]

        from ``d.qfrc_constraint`` at the free joint DOF address. The first three
        values are the translational constraint force on the free body in world
        coordinates. In this isolated example that force is the weld support force.

        At static equilibrium, the weld must balance gravity:

            F_constraint + F_gravity = 0
            F_constraint_z = -m * gravity_z

        With ``m = 1000 kg`` and ``gravity_z = -9.82 m/s^2``, expect:

            F_constraint_z = 9820 N
        """
        return self.d.qfrc_constraint[self.ball_free_dofadr : self.ball_free_dofadr + 6]

    def support_moment_about_com(self, support_force):
        """Compute the support-force moment about the ball center of mass.

        This computes the torque that comes from applying ``support_force`` at
        ``ball_site`` and taking moments about the ball COM:

            r = site_position_world - com_position_world
            tau_com = r x support_force

        ``d.xipos[body_id]`` is used for the COM because it is the world position
        of the body's inertial frame. ``d.xpos[body_id]`` is the world position of
        the body frame. They are equal only when the inertial frame is centered at
        the body origin.

        In this model, the nominal rest value is:

            r = [0.2, 0, 0] m
            F = [0, 0, 9820] N
            tau_com = r x F = [0, -1964, 0] N*m
        """
        moment_arm = (
            self.d.site_xpos[self.ball_site_id] - self.d.xipos[self.ball_body_id]
        )
        return np.cross(moment_arm, support_force)

    def run(self) -> None:
        print(
            "expected at rest: "
            f"support force z ~= {self.expected_force_z:.1f} N | "
            f"support moment y about COM ~= {self.expected_torque_y:.1f} N*m"
        )

        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.cb
        ) as viewer:
            while viewer.is_running():
                mj.mj_step(self.m, self.d)

                f = self.site_force_sensor()
                wrench = self.ball_constraint_wrench()
                constraint_force = wrench[:3]
                support_moment = self.support_moment_about_com(constraint_force)

                print(
                    "site force sensor: "
                    f"{f[0]:.4f}, {f[1]:.4f}, {f[2]:.4f} | "
                    "ball constraint force: "
                    f"{constraint_force[0]:.4f}, "
                    f"{constraint_force[1]:.4f}, "
                    f"{constraint_force[2]:.4f} | "
                    "support moment about COM: "
                    f"{support_moment[0]:.4f}, "
                    f"{support_moment[1]:.4f}, "
                    f"{support_moment[2]:.4f}"
                )

                viewer.sync()


if __name__ == "__main__":
    sim = Sim()
    sim.run()
