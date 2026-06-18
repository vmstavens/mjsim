import numpy as np
import mujoco as mj
import mujoco.viewer

from mjsim.utils.mjs import cable, empty_scene


CABLE_MODEL_NAME = "dlo"
CABLE_ATTACH_PREFIX = "cable/"
CABLE_NAME_PREFIX = f"{CABLE_ATTACH_PREFIX}{CABLE_MODEL_NAME}:"
CABLE_TIP_SITE = f"{CABLE_NAME_PREFIX}S_last"
CABLE_BODY_PREFIX = f"{CABLE_NAME_PREFIX}B"


class Sim:
    def __init__(self):
        self.m, self.d = self._build_model()
        mj.mj_forward(self.m, self.d)

        self.loadcell_site_id = self.m.site("loadcell_site").id
        self.cable_tip_site_id = self.m.site(CABLE_TIP_SITE).id
        self.loadcell_force_adr = self.m.sensor("loadcell_force").adr[0]
        self.loadcell_torque_adr = self.m.sensor("loadcell_torque").adr[0]
        self.cable_body_ids = self._body_ids_with_prefix(CABLE_BODY_PREFIX)

        self.cable_mass = sum(float(self.m.body_mass[i]) for i in self.cable_body_ids)
        self.loadcell_mass = float(self.m.body("loadcell").mass[0])
        self.expected_sensor_force_z = (
            -(self.cable_mass + self.loadcell_mass) * self.m.opt.gravity[2]
        )
        self.expected_support_moment = self.support_moment_about_cable_com(
            np.array([0, 0, self.expected_sensor_force_z])
        )
        self.step_count = 0

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData]:
        """Build a DLO tip-load example with a mocap-mounted load cell.

        ``ms.cable`` creates a composite cable: multiple bodies joined together by
        MuJoCo joints and the cable elasticity plugin. The tip site is welded to a
        small fixed child body called ``loadcell`` under the mocap body:

            mocap -> loadcell --weld equality-- cable tip site

        A MuJoCo ``mjSENS_FORCE`` sensor on ``loadcell_site`` measures the
        parent-child interaction force between ``loadcell`` and ``mocap``. Since
        the cable tip is welded to ``loadcell``, this is the mount/load-cell force.
        The loadcell site is rotated to match the cable endpoint site frame; if the
        welded site frames start with different orientations, the weld will create
        an artificial torque preload.

        The cable is initialized vertically below the mount using ``curve="0 0 s"``
        and an attach frame at z=0.5, so the last site starts at z=1.5 and is welded
        to the mocap site. At static equilibrium, the vertical force should be:

            Fz = (m_cable + m_loadcell) * g

        The loadcell mass is set to 1e-6 kg, so this is effectively the cable
        weight. With the current cable settings, the compiled cable mass is about
        0.18 kg, so expect:

            Fz ~= 0.18 kg * 9.82 m/s^2 = 1.77 N

        The moment caused by this support force about the cable center of mass is:

            r = loadcell_site_position - cable_com
            tau_com = r x F

        In this vertical setup, ``r`` and ``F`` are nearly collinear, so
        ``tau_com`` should be near zero. If you move the cable COM sideways from
        the mount, the expected moment magnitude becomes ``|r_perp| * |F|``.
        """
        scene = empty_scene(memory="100M")

        m_body = scene.worldbody.add_body(name="mocap", mocap=True, pos=[0, 0, 1.5])
        m_body.add_geom(
            name="mocap_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.04, 0.04, 0.04],
            contype=0,
            conaffinity=0,
            rgba=[1, 0, 0, 1],
        )

        loadcell = m_body.add_body(name="loadcell", pos=[0, 0, 0])
        loadcell.add_geom(
            name="loadcell_geom",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.015],
            mass=1e-6,
            contype=0,
            conaffinity=0,
            rgba=[0.2, 0.2, 1, 1],
        )
        loadcell.add_site(
            name="loadcell_site",
            pos=[0, 0, 0],
            quat=[0, 0.70710678118, 0, 0.70710678118],
            size=[0.03],
            rgba=[0, 1, 0, 1],
        )

        dlo = cable(
            model_name=CABLE_MODEL_NAME,
            n_segments=10,
            length=1.0,
            mass=0.2,
            curve="0 0 s",
            segment_size=0.006,
            joint_damping="0.1",
            geom_rgba=[0.2, 0.2, 0.2, 1],
        )
        f_cable = scene.worldbody.add_frame(pos=[0, 0, 0.5])
        scene.attach(dlo, prefix=CABLE_ATTACH_PREFIX, frame=f_cable)

        scene.add_equality(
            name="tip_weld",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="loadcell_site",
            name2=CABLE_TIP_SITE,
            solref=[0.001, 3],
        )

        scene.add_sensor(
            name="loadcell_force",
            type=mj.mjtSensor.mjSENS_FORCE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="loadcell_site",
            dim=3,
        )
        scene.add_sensor(
            name="loadcell_torque",
            type=mj.mjtSensor.mjSENS_TORQUE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="loadcell_site",
            dim=3,
        )

        m = scene.compile()
        d = mj.MjData(m)

        return m, d

    def _body_ids_with_prefix(self, prefix: str) -> list[int]:
        return [
            i
            for i in range(self.m.nbody)
            if (mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, i) or "").startswith(prefix)
        ]

    def cb(self, key: int) -> None:
        pass

    def sensor_vector_to_world(self, vector):
        """Transform a vector from the loadcell site frame to the world frame.

        MuJoCo force and torque site sensors report their values in the sensor site
        frame. The site frame is aligned with the world frame in this example, but
        applying the site rotation keeps the helper correct if the mocap body is
        rotated later:

            v_world = R_site_to_world * v_site
        """
        site_rot = self.d.site_xmat[self.loadcell_site_id].reshape(3, 3)
        return site_rot @ vector

    def loadcell_force_site_frame(self):
        """Return the raw force sensor value in the loadcell site frame.

        The sensor measures the parent-child interaction force between
        ``loadcell`` and ``mocap``. Because the cable tip is welded to the loadcell,
        this is the support force transmitted through the mount.

        For the vertical hanging cable at rest, expect approximately:

            Fz = (m_cable + m_loadcell) * 9.82 ~= 1.77 N
        """
        return self.d.sensordata[self.loadcell_force_adr : self.loadcell_force_adr + 3]

    def loadcell_torque_site_frame(self):
        """Return the raw torque sensor value in the loadcell site frame.

        This is the torque transmitted through the parent-child mount at the
        loadcell site. It is not the same thing as ``r x F`` about the cable COM;
        for that derived moment, use ``support_moment_about_cable_com``.
        """
        return self.d.sensordata[
            self.loadcell_torque_adr : self.loadcell_torque_adr + 3
        ]

    def cable_com(self):
        """Return the cable center of mass in world coordinates.

        The cable is made from multiple generated bodies, so the total COM is the
        mass-weighted average:

            com = sum(m_i * xipos_i) / sum(m_i)

        ``xipos`` is used because it is the world position of each body's inertial
        frame, i.e. its center of mass.
        """
        weighted_positions = sum(
            float(self.m.body_mass[i]) * self.d.xipos[i] for i in self.cable_body_ids
        )
        return weighted_positions / self.cable_mass

    def support_moment_about_cable_com(self, support_force_world):
        """Compute the support-force moment about the cable COM.

        This is the moment generated by applying the measured support force at the
        loadcell site and taking moments about the cable COM:

            r = loadcell_site_position - cable_com
            tau_com = r x support_force_world

        In the default vertical setup, ``r`` is nearly vertical and the support
        force is vertical, so ``tau_com`` should be near zero. If the cable COM is
        offset horizontally by ``d`` from the mount, expect:

            |tau_com| ~= d * |support_force|
        """
        moment_arm = self.d.site_xpos[self.loadcell_site_id] - self.cable_com()
        return np.cross(moment_arm, support_force_world)

    def run(self) -> None:
        print(
            "expected at rest: "
            f"compiled cable mass = {self.cable_mass:.4f} kg | "
            f"loadcell mass = {self.loadcell_mass:.6f} kg | "
            f"loadcell force z ~= {self.expected_sensor_force_z:.4f} N | "
            "support moment about cable COM ~= "
            f"{self.expected_support_moment[0]:.4f}, "
            f"{self.expected_support_moment[1]:.4f}, "
            f"{self.expected_support_moment[2]:.4f} N*m"
        )

        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.cb
        ) as viewer:
            while viewer.is_running():
                mj.mj_step(self.m, self.d)
                self.step_count += 1

                if self.step_count % 50 != 0:
                    viewer.sync()
                    continue

                force_site = self.loadcell_force_site_frame()
                torque_site = self.loadcell_torque_site_frame()
                force_world = self.sensor_vector_to_world(force_site)
                torque_world = self.sensor_vector_to_world(torque_site)
                support_moment = self.support_moment_about_cable_com(force_world)

                print(
                    "loadcell force world: "
                    f"{force_world[0]:.4f}, "
                    f"{force_world[1]:.4f}, "
                    f"{force_world[2]:.4f} | "
                    "loadcell torque world: "
                    f"{torque_world[0]:.4f}, "
                    f"{torque_world[1]:.4f}, "
                    f"{torque_world[2]:.4f} | "
                    "support moment about cable COM: "
                    f"{support_moment[0]:.4f}, "
                    f"{support_moment[1]:.4f}, "
                    f"{support_moment[2]:.4f}"
                )

                viewer.sync()


if __name__ == "__main__":
    sim = Sim()
    sim.run()
