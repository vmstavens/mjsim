import time

import mujoco as mj
import mujoco.viewer
import numpy as np
from mocap_rod_wall_force_plot import weld_sensor

from mjsim.utils.mjs import cable, empty_scene

CABLE_MODEL_NAME = "dlo"
CABLE_ATTACH_PREFIX = "cable/"
CABLE_NAME_PREFIX = f"{CABLE_ATTACH_PREFIX}{CABLE_MODEL_NAME}:"
CABLE_TIP_SITE = f"{CABLE_NAME_PREFIX}S_last"
CABLE_BODY_PREFIX = f"{CABLE_NAME_PREFIX}B"
TIP_WELD_NAME = "tip_weld"
MOCAP_WELD_SITE = "mocap_site"
FORCE_ARROW_SCALE = 0.5
FORCE_ARROW_WIDTH = 0.01
FORCE_ARROW_MIN_LENGTH = 1e-4


class Sim:
    def __init__(self):
        self.m, self.d = self._build_model()
        mj.mj_forward(self.m, self.d)

        self.mocap_weld_site_id = self.m.site(MOCAP_WELD_SITE).id
        self.cable_tip_site_id = self.m.site(CABLE_TIP_SITE).id
        self.cable_body_ids = self._body_ids_with_prefix(CABLE_BODY_PREFIX)

        self.cable_mass = sum(float(self.m.body_mass[i]) for i in self.cable_body_ids)
        self.expected_support_force_z = -self.cable_mass * self.m.opt.gravity[2]
        self.expected_support_moment = self.support_moment_about_cable_com(
            np.array([0, 0, self.expected_support_force_z])
        )
        self.step_count = 0
        self._mocap_site_wrench_zero = np.zeros(6)
        self._weld_constraint_force_zero = np.zeros(3)
        self.zero_step_count: int | None = None

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData]:
        """Build a DLO tip-load example with a mocap-mounted cable.

        ``ms.cable`` creates a composite cable: multiple bodies joined together by
        MuJoCo joints and the cable elasticity plugin. The tip site is welded to a
        site on the mocap body:

            mocap_site --weld equality-- cable tip site

        The site is rotated to match the cable endpoint frame; if the welded site
        frames start with different orientations, the weld will create an
        artificial torque preload. ``weld_sensor`` is called with this explicit
        site name to read the matching force/torque sensors.

        The cable is initialized vertically below the mount using ``curve="0 0 s"``
        and an attach frame at z=0.5, so the last site starts at z=1.5 and is welded
        to the mocap site. At static equilibrium, the vertical force should be:

            Fz = m_cable * g

        With the current cable settings, the compiled cable mass is about 0.18 kg,
        so expect:

            Fz ~= 0.18 kg * 9.82 m/s^2 = 1.77 N

        The moment caused by this support force about the cable center of mass is:

            r = mocap_weld_site_position - cable_com
            tau_com = r x F

        In this vertical setup, ``r`` and ``F`` are nearly collinear, so
        ``tau_com`` should be near zero. If you move the cable COM sideways from
        the mount, the expected moment magnitude becomes ``|r_perp| * |F|``.
        """
        scene = empty_scene(memory="100M")

        m_body = scene.worldbody.add_body(
            name="mocap", mocap=True, pos=[0, 0, 1.5], euler=[0, np.pi / 2, 0]
        )
        m_body.add_geom(
            name="mocap_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.04, 0.04, 0.04],
            contype=0,
            conaffinity=0,
            rgba=[1, 0, 0, 1],
        )
        m_body.add_site(
            name=MOCAP_WELD_SITE,
            pos=[0, 0, 0],
            quat=[0, 0.70710678118, 0, 0.70710678118],
            size=[0.03],
            rgba=[0, 1, 0, 1],
        )

        dlo = cable(
            model_name=CABLE_MODEL_NAME,
            n_segments=10,
            length=1.0,
            twist=5 * 60000,
            bend=5 * 10000000,
            mass=0.2,
            curve="0 0 s",
            segment_size=0.006,
            joint_damping="0.1",
            geom_rgba=[0.2, 0.2, 0.2, 1],
        )
        f_cable = scene.worldbody.add_frame(pos=[0, 0, 0.5])
        scene.attach(dlo, prefix=CABLE_ATTACH_PREFIX, frame=f_cable)

        scene.add_equality(
            name=TIP_WELD_NAME,
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1=MOCAP_WELD_SITE,
            name2=CABLE_TIP_SITE,
            solref=[0.000000000001, 1],
        )
        scene.add_sensor(
            name=f"{TIP_WELD_NAME}_force",
            type=mj.mjtSensor.mjSENS_FORCE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname=MOCAP_WELD_SITE,
            dim=3,
        )
        scene.add_sensor(
            name=f"{TIP_WELD_NAME}_torque",
            type=mj.mjtSensor.mjSENS_TORQUE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname=MOCAP_WELD_SITE,
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
        if key in (ord("z"), ord("Z")):
            self.zero()

    def zero(self) -> None:
        """Use the current force/torque readings as the output zero point."""
        self._mocap_site_wrench_zero = self._mocap_site_wrench_world_raw()
        self._weld_constraint_force_zero = self._weld_constraint_force_world_raw()
        self.zero_step_count = self.step_count
        print(f"zeroed force readings at step {self.step_count}")

    def mocap_site_wrench_world(self):
        """Return ``[Fx, Fy, Fz, Tx, Ty, Tz]`` from the mocap weld sensor API."""
        return self._mocap_site_wrench_world_raw() - self._mocap_site_wrench_zero

    def _mocap_site_wrench_world_raw(self):
        return weld_sensor(self.m, self.d, TIP_WELD_NAME, site_name=MOCAP_WELD_SITE)

    def weld_constraint_force_world(self):
        """Return the site-site weld constraint force for this direct mocap weld."""
        return (
            self._weld_constraint_force_world_raw() - self._weld_constraint_force_zero
        )

    def _weld_constraint_force_world_raw(self):
        """Return the unzeroed site-site weld constraint force."""
        eq_id = self.m.equality(TIP_WELD_NAME).id
        rows = [
            i
            for i in range(self.d.nefc)
            if self.d.efc_type[i] == mj.mjtConstraint.mjCNSTR_EQUALITY
            and self.d.efc_id[i] == eq_id
        ]
        if len(rows) != 6:
            raise RuntimeError(
                f"Expected 6 constraint rows for weld {TIP_WELD_NAME!r}, got {len(rows)}"
            )
        return -self.d.efc_force[rows[:3]].copy()

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

        This is the moment generated by applying the support force at the mocap
        weld site and taking moments about the cable COM:

            r = mocap_weld_site_position - cable_com
            tau_com = r x support_force_world

        In the default vertical setup, ``r`` is nearly vertical and the support
        force is vertical, so ``tau_com`` should be near zero. If the cable COM is
        offset horizontally by ``d`` from the mount, expect:

            |tau_com| ~= d * |support_force|
        """
        moment_arm = self.d.site_xpos[self.mocap_weld_site_id] - self.cable_com()
        return np.cross(moment_arm, support_force_world)

    def draw_sensor_force_arrow(
        self,
        scene: mj.MjvScene,
        sensor_force_world: np.ndarray,
    ) -> None:
        """Draw the zeroed sensor force as a dynamic arrow in the viewer scene."""
        scene.ngeom = 0

        arrow = FORCE_ARROW_SCALE * sensor_force_world
        length = float(np.linalg.norm(arrow))
        if length < FORCE_ARROW_MIN_LENGTH:
            return

        start = self.d.site_xpos[self.mocap_weld_site_id].copy()
        end = start + arrow
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            geom,
            mj.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            np.array([1.0, 0.1, 0.05, 0.85]),
        )
        mj.mjv_connector(
            geom,
            mj.mjtGeom.mjGEOM_ARROW,
            FORCE_ARROW_WIDTH,
            start,
            end,
        )
        scene.ngeom += 1

    def run(self) -> None:
        print(
            "expected at rest: "
            f"compiled cable mass = {self.cable_mass:.4f} kg | "
            f"cable support force z ~= {self.expected_support_force_z:.4f} N | "
            "support moment about cable COM ~= "
            f"{self.expected_support_moment[0]:.4f}, "
            f"{self.expected_support_moment[1]:.4f}, "
            f"{self.expected_support_moment[2]:.4f} N*m"
        )
        print(
            "API call: "
            f'wrench = weld_sensor(sim.m, sim.d, "{TIP_WELD_NAME}", '
            f'site_name="{MOCAP_WELD_SITE}")'
        )

        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.cb
        ) as viewer:
            while viewer.is_running():
                step_start = time.time()
                mj.mj_step(self.m, self.d)
                self.step_count += 1

                mocap_site_wrench = self.mocap_site_wrench_world()
                sensor_force_world = mocap_site_wrench[:3]
                self.draw_sensor_force_arrow(viewer.user_scn, sensor_force_world)

                if self.step_count % 50 == 0:
                    sensor_torque_world = mocap_site_wrench[3:]
                    weld_force_world = self.weld_constraint_force_world()
                    support_moment = self.support_moment_about_cable_com(
                        weld_force_world
                    )

                    print(
                        # "weld constraint force world: "
                        # f"{weld_force_world[0]:.4f}, "
                        # f"{weld_force_world[1]:.4f}, "
                        # f"{weld_force_world[2]:.4f} | "
                        "mocap-site sensor force world: "
                        f"{sensor_force_world[0]:.4f}, "
                        f"{sensor_force_world[1]:.4f}, "
                        f"{sensor_force_world[2]:.4f} | "
                        "mocap-site sensor torque world: "
                        f"{sensor_torque_world[0]:.4f}, "
                        f"{sensor_torque_world[1]:.4f}, "
                        f"{sensor_torque_world[2]:.4f} | "
                        # "support moment about cable COM: "
                        # f"{support_moment[0]:.4f}, "
                        # f"{support_moment[1]:.4f}, "
                        # f"{support_moment[2]:.4f}"
                    )

                viewer.sync()
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    sim = Sim()
    sim.run()
