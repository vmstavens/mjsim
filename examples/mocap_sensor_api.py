from __future__ import annotations

import argparse
from dataclasses import dataclass

import mujoco as mj
import mujoco.viewer
import numpy as np

from mjsim.utils.mjs import empty_scene


@dataclass(frozen=True)
class WeldSensorInfo:
    site_id: int
    force_adr: int
    torque_adr: int


class MocapWeldSensors:
    """Read force/torque sensors associated with site-site welds.

    MuJoCo force and torque sensors do not directly measure an equality weld.
    They measure the parent-child interaction wrench at the body containing the
    sensor site. The reusable pattern is therefore:

        mocap body -> tiny loadcell body -> loadcell_site --weld-- object_site

    Add one force sensor and one torque sensor on ``loadcell_site``. This class
    then maps ``weld_name`` to the instrumented welded site and returns the
    measured 6D wrench.
    """

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        site_for_weld: dict[str, str] | None = None,
    ):
        self.model = model
        self.data = data
        self._site_for_weld = site_for_weld or {}
        self._cache: dict[str, WeldSensorInfo] = {}

    def weld_sensor(self, weld_name: str, frame: str = "world") -> np.ndarray:
        """Return ``[Fx, Fy, Fz, Tx, Ty, Tz]`` measured at a welded loadcell site."""
        if frame not in {"world", "site"}:
            raise ValueError("frame must be 'world' or 'site'")

        info = self._info_for_weld(weld_name)
        force = self._sensor_data(info.force_adr)
        torque = self._sensor_data(info.torque_adr)

        if frame == "world":
            site_rot = self.data.site_xmat[info.site_id].reshape(3, 3)
            force = site_rot @ force
            torque = site_rot @ torque

        return np.concatenate([force, torque])

    def _info_for_weld(self, weld_name: str) -> WeldSensorInfo:
        if weld_name not in self._cache:
            self._cache[weld_name] = self._build_info(weld_name)
        return self._cache[weld_name]

    def _build_info(self, weld_name: str) -> WeldSensorInfo:
        eq_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_EQUALITY, weld_name)
        if eq_id < 0:
            raise ValueError(f"Unknown equality weld {weld_name!r}")
        if self.model.eq_type[eq_id] != mj.mjtEq.mjEQ_WELD:
            raise ValueError(f"Equality {weld_name!r} is not a weld")
        if self.model.eq_objtype[eq_id] != mj.mjtObj.mjOBJ_SITE:
            raise ValueError(f"Weld {weld_name!r} does not connect sites")

        explicit_site = self._site_for_weld.get(weld_name)
        if explicit_site is not None:
            site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, explicit_site)
            if site_id < 0:
                raise ValueError(f"Unknown site {explicit_site!r}")
            return self._sensor_info_for_site(weld_name, site_id)

        site_ids = [
            int(self.model.eq_obj1id[eq_id]),
            int(self.model.eq_obj2id[eq_id]),
        ]
        instrumented = []
        for site_id in site_ids:
            try:
                instrumented.append(self._sensor_info_for_site(weld_name, site_id))
            except ValueError:
                pass

        if len(instrumented) == 1:
            return instrumented[0]
        if len(instrumented) > 1:
            raise ValueError(
                f"Weld {weld_name!r} has sensors on both welded sites; pass "
                "site_for_weld={weld_name: site_name} to disambiguate"
            )
        raise ValueError(
            f"Weld {weld_name!r} has no force+torque sensor pair on either welded "
            "site. Put sensors on a mocap-side loadcell site."
        )

    def _sensor_info_for_site(self, weld_name: str, site_id: int) -> WeldSensorInfo:
        force_adr = self._sensor_address(site_id, mj.mjtSensor.mjSENS_FORCE)
        torque_adr = self._sensor_address(site_id, mj.mjtSensor.mjSENS_TORQUE)
        if force_adr is None or torque_adr is None:
            site_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_SITE, site_id)
            raise ValueError(
                f"Weld {weld_name!r} site {site_name!r} does not have both force "
                "and torque sensors"
            )
        return WeldSensorInfo(site_id=site_id, force_adr=force_adr, torque_adr=torque_adr)

    def _sensor_address(self, site_id: int, sensor_type: mj.mjtSensor) -> int | None:
        for sensor_id in range(self.model.nsensor):
            if (
                self.model.sensor_type[sensor_id] == sensor_type
                and self.model.sensor_objtype[sensor_id] == mj.mjtObj.mjOBJ_SITE
                and self.model.sensor_objid[sensor_id] == site_id
                and self.model.sensor_dim[sensor_id] == 3
            ):
                return int(self.model.sensor_adr[sensor_id])
        return None

    def _sensor_data(self, adr: int) -> np.ndarray:
        return self.data.sensordata[adr : adr + 3].copy()


class Sim:
    def __init__(self):
        self.m, self.d = self._build_model()
        mj.mj_forward(self.m, self.d)

        self.ms = MocapWeldSensors(self.m, self.d)
        self.box_body_id = self.m.body("box").id
        self.box_free_dofadr = self.m.joint("box_freejoint").dofadr[0]
        self.box_mass = float(self.m.body("box").mass[0])
        self.expected_force_z = -self.box_mass * self.m.opt.gravity[2]
        self.step_count = 0

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData]:
        scene = empty_scene()
        scene.option.timestep = 0.002

        mocap = scene.worldbody.add_body(name="mocap", mocap=True, pos=[0, 0, 0.6])
        mocap.add_geom(
            name="mocap_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.04, 0.04, 0.04],
            contype=0,
            conaffinity=0,
            rgba=[1, 0, 0, 1],
        )

        loadcell = mocap.add_body(name="box_loadcell", pos=[0, 0, 0])
        loadcell.add_geom(
            name="box_loadcell_geom",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.018],
            mass=1e-6,
            contype=0,
            conaffinity=0,
            rgba=[0.2, 0.2, 1, 1],
        )
        loadcell.add_site(
            name="box_loadcell_site",
            pos=[0, 0, 0],
            size=[0.03],
            rgba=[0, 1, 0, 1],
        )

        box = scene.worldbody.add_body(name="box", pos=[0.2, 0, 0.6])
        box.add_freejoint(name="box_freejoint")
        box.add_geom(
            name="box_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.2, 0.03, 0.03],
            mass=2.0,
            rgba=[0.8, 0.8, 0.8, 1],
        )
        box.add_site(
            name="box_site",
            pos=[-0.2, 0, 0],
            size=[0.02],
            rgba=[1, 0.8, 0, 1],
        )

        scene.add_equality(
            name="box_weld",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="box_loadcell_site",
            name2="box_site",
            solref=[0.001, 3],
        )
        scene.add_sensor(
            name="box_weld_force",
            type=mj.mjtSensor.mjSENS_FORCE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="box_loadcell_site",
            dim=3,
        )
        scene.add_sensor(
            name="box_weld_torque",
            type=mj.mjtSensor.mjSENS_TORQUE,
            objtype=mj.mjtObj.mjOBJ_SITE,
            objname="box_loadcell_site",
            dim=3,
        )

        m = scene.compile()
        d = mj.MjData(m)
        return m, d

    def apply_demo_force(self) -> None:
        t = self.d.time
        self.d.xfrc_applied[:, :] = 0
        self.d.xfrc_applied[self.box_body_id, :3] = [
            3.0 * np.sin(2.0 * np.pi * 0.5 * t),
            0,
            0,
        ]

    def step(self) -> np.ndarray:
        self.apply_demo_force()
        mj.mj_step(self.m, self.d)
        self.step_count += 1
        return self.ms.weld_sensor("box_weld")

    def real_weld_force(self) -> np.ndarray:
        """Return the actual weld support force on the box in world coordinates.

        This is only a ground-truth shortcut for this demo: the box has one free
        joint and one equality weld, so the translational part of
        ``qfrc_constraint`` is the force applied by that weld to the box.
        """
        return self.d.qfrc_constraint[self.box_free_dofadr : self.box_free_dofadr + 3]

    def run_headless(self, n_steps: int) -> None:
        for _ in range(n_steps):
            wrench = self.step()
        self.print_force_comparison(wrench)

    def run_viewer(self) -> None:
        print(
            "expected at rest: "
            f"box mass = {self.box_mass:.3f} kg | "
            f"weld force z ~= {self.expected_force_z:.3f} N"
        )
        print('API call: wrench = sim.ms.weld_sensor("box_weld")')

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running():
                wrench = self.step()
                if self.step_count % 50 == 0:
                    self.print_force_comparison(wrench)
                viewer.sync()

    def print_force_comparison(self, loadcell_wrench: np.ndarray) -> None:
        real_force = self.real_weld_force()
        estimated_force = loadcell_wrench[:3]
        error = estimated_force - real_force
        print(
            "box_weld force world [N]: "
            f"real=({real_force[0]: .4f}, {real_force[1]: .4f}, "
            f"{real_force[2]: .4f}) | "
            f"loadcell=({estimated_force[0]: .4f}, {estimated_force[1]: .4f}, "
            f"{estimated_force[2]: .4f}) | "
            f"error=({error[0]: .4e}, {error[1]: .4e}, {error[2]: .4e})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    sim = Sim()
    if args.headless:
        sim.run_headless(args.steps)
    else:
        sim.run_viewer()


if __name__ == "__main__":
    main()
