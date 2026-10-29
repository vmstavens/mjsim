"""Tie an initially straight, stress-free cable using two endpoint mocap handles.

Run without flags for the native MuJoCo viewer, or use --headless. The virtual
gray return uses the closed-overhand example's arc convention. It is not a
physical segment and its validity is checked at every classification sample.
Use --preformed to run the earlier preformed-loop demonstration instead.
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from itertools import pairwise
from queue import SimpleQueue

import mujoco as mj
import numpy as np
from dlo_knot import _vector, knot_points, overhand_core_count

from mjsim.utils.knot_closure import KnotProgress, outer_return_arc


class OverhandTyingSim:
    timestep = 0.002
    sample_steps = 100  # 5 Hz in simulation time, independent of rendering.
    duration = 16.0
    radius = 0.0025
    gripper_constraint = "connect"
    closure_bow = (0.0, -0.30, 0.0)
    camera_lookat = (0.0, 0.08, 0.5)
    camera_distance = 1.5
    camera_elevation = -40

    def __init__(self):
        # Use precisely the physical core of the earlier overhand example.
        # Its gray return is replaced by an analysis-only curve in this scene.
        points = knot_points("overhand", 68)[: overhand_core_count(68)]
        self.final_tip = points[-1].copy()
        self.retained_tip = np.array([-0.4, 0.0, 0.2])
        self.initial_tip = np.array([-0.1, 0.15, 0.8])
        self.thread_tip = np.array([-0.1, 0.15, 0.6])
        self.model = self._build_model(points)
        self.data = mj.MjData(self.model)
        self.body_ids = np.array(
            [self.model.body(f"cable_{i}").id for i in range(len(points) - 1)]
        )
        self.mocap_ids = [
            int(self.model.body(name).mocapid[0])
            for name in ("held_gripper", "working_gripper")
        ]
        self.endpoint_ids = [
            self.model.site(name).id for name in ("cable_start", "cable_end")
        ]
        mj.mj_forward(self.model, self.data)

        # Deterministically prepare the loose loop using endpoint motion alone.
        # This happens before the visible trial. Freeze the resulting shape as
        # the spring reference, so holding the tips cannot relax it into a knot.
        self._prepare_move(self.initial_tip, 3.0)
        self._prepare_move(self.initial_tip, 1.0)
        self.model.qpos_spring[:] = self.data.qpos
        self.model.dof_damping[6:] = 0.005
        self.data.qvel[:] = 0
        self._prepare_move(self.initial_tip, 2.0)
        self._initial_qpos = self.data.qpos.copy()
        self._initial_mocap = self.data.mocap_pos.copy()
        self.reset()
        if self.progress.error or self.progress.label != "unknot":
            raise RuntimeError(
                f"Prepared cable must start as a valid unknot: {self.status()}"
            )

    def _build_model(self, points):
        root = ET.Element("mujoco", model="mocap_overhand_tying")
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(
            root,
            "option",
            timestep=str(self.timestep),
            gravity="0 0 0",
            integrator="implicitfast",
            iterations="80",
            tolerance="1e-9",
        )
        ET.SubElement(root, "statistic", center="0 0.1 0.5", extent="1.4")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
        ET.SubElement(
            world, "geom", type="plane", size="1 1 0.01", rgba="0.12 0.14 0.16 1"
        )
        parent = world
        for i in range(len(points) - 1):
            body = ET.SubElement(
                parent,
                "body",
                name=f"cable_{i}",
                pos=_vector(points[i] if i == 0 else points[i] - points[i - 1]),
            )
            if i == 0:
                ET.SubElement(body, "freejoint")
                ET.SubElement(
                    body, "site", name="cable_start", size="0.003", rgba="0 0 0 0"
                )
            else:
                ET.SubElement(
                    body, "joint", type="ball", stiffness="0.02", damping="0.001"
                )
            delta = points[i + 1] - points[i]
            ET.SubElement(
                body,
                "geom",
                type="capsule",
                size=str(self.radius),
                fromto=_vector(np.r_[np.zeros(3), delta]),
                mass="0.004",
                friction="0.3 0.01 0.001",
                solref="0.004 1",
                rgba="0.95 0.72 0.15 1",
            )
            parent = body
        ET.SubElement(
            parent,
            "site",
            name="cable_end",
            pos=_vector(delta),
            size="0.003",
            rgba="0 0 0 0",
        )
        equality = ET.SubElement(root, "equality")
        for name, endpoint, position, color in (
            ("held_gripper", "cable_start", points[0], "0.3 0.9 0.5 1"),
            ("working_gripper", "cable_end", points[-1], "0.3 0.65 1 1"),
        ):
            body = ET.SubElement(
                world, "body", name=name, mocap="true", pos=_vector(position)
            )
            # These are idealized endpoint handles, not collision-enabled jaws.
            ET.SubElement(
                body,
                "geom",
                type="sphere",
                size="0.012",
                rgba=color,
                contype="0",
                conaffinity="0",
            )
            ET.SubElement(
                body, "site", name=f"{name}_site", size="0.003", rgba="0 0 0 0"
            )
            ET.SubElement(
                equality,
                self.gripper_constraint,
                site1=f"{name}_site",
                site2=endpoint,
                solref="0.004 1",
            )
        return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    @staticmethod
    def _blend(start, end, fraction):
        u = np.clip(fraction, 0, 1)
        u = u * u * (3 - 2 * u)
        return (1 - u) * start + u * end

    def _prepare_move(self, destination, seconds):
        mocap = self.mocap_ids[1]
        origin = self.data.mocap_pos[mocap].copy()
        steps = round(seconds / self.timestep)
        for i in range(steps):
            self.data.mocap_pos[mocap] = self._blend(
                origin, destination, (i + 1) / steps
            )
            mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data)

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.data.mocap_pos[:] = self._initial_mocap
        mj.mj_forward(self.model, self.data)
        self.steps = 0
        self.progress = KnotProgress(required_matches=5, closure_bow=self.closure_bow)
        self.history = []
        self.sample()

    def centerline(self):
        return np.vstack(
            (self.data.xpos[self.body_ids], self.data.site_xpos[self.endpoint_ids[1]])
        )

    def tracking_error(self):
        return float(
            np.max(
                np.linalg.norm(
                    self.data.site_xpos[self.endpoint_ids]
                    - self.data.mocap_pos[self.mocap_ids],
                    axis=1,
                )
            )
        )

    def sample(self):
        self.progress.sample(
            self.centerline(), self.data.time, tracking_error=self.tracking_error()
        )
        self.history.append(
            (
                float(self.data.time),
                self.progress.label,
                self.progress.error,
                self.progress.successful,
            )
        )

    def phase(self):
        t = self.data.time
        if t < 1:
            return "Ready: unknotted preformed loop"
        if t < 3:
            return "Thread working end through loop"
        if t < 7:
            return "Draw working end through"
        if t < 11:
            return "Gently pull through and retain knot"
        return "Hold and check"

    def step(self, *, drive=True):
        """Only mocap targets change here; the cable moves through physics."""
        t = (self.steps + 1) * self.timestep
        if drive:
            if t < 1:
                target = self.initial_tip
            elif t < 3:
                target = self._blend(self.initial_tip, self.thread_tip, (t - 1) / 2)
            elif t < 7:
                target = self._blend(self.thread_tip, self.final_tip, (t - 3) / 4)
            else:
                target = self._blend(self.final_tip, self.retained_tip, (t - 7) / 4)
            self.data.mocap_pos[self.mocap_ids[1]] = target
        mj.mj_step(self.model, self.data)
        mj.mj_kinematics(self.model, self.data)
        self.steps += 1
        if self.steps % self.sample_steps == 0:
            self.sample()

    def status(self):
        p = self.progress
        first = (
            "not yet"
            if p.first_success_time is None
            else f"{p.first_success_time:.2f} s"
        )
        label = p.label or "unknown"
        values = (
            "-"
            if p.fingerprint is None
            else ", ".join(f"{x:.4g}" for x in p.fingerprint)
        )
        return (
            f"Time: {self.data.time:.2f} s | {self.phase()}\n"
            f"Closure: {'valid' if p.error is None else 'INVALID: ' + p.error}\n"
            f"Detected: {label} | Alexander: ({values})\n"
            f"Overhand match: {p.label == 'trefoil' and p.error is None}\n"
            f"Consecutive matches: {min(p.streak, p.required_matches)}/{p.required_matches}\n"
            f"Stable success: {p.successful} | first at: {first}\n"
            f"Endpoint error: {self.tracking_error() * 1000:.3f} mm"
        )

    def draw_closure(self, scene):
        scene.ngeom = 0
        arc = outer_return_arc(self.centerline(), bow=self.closure_bow)
        color = [0.95, 0.2, 0.2, 1] if self.progress.error else [0.6, 0.65, 0.7, 1]
        for start, end in pairwise(arc):
            geom = scene.geoms[scene.ngeom]
            mj.mjv_initGeom(
                geom,
                mj.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.asarray(color),
            )
            mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, 0.001, start, end)
            scene.ngeom += 1


class StraightOverhandTyingSim(OverhandTyingSim):
    """A tuned dynamic tying sequence, with no curved rest shape or fixture.

    Endpoint welds transmit both position and orientation. Compression plus
    axial twist forms a loop; its measured plane guides the threading
    motion. Inertia keeps the opening available briefly while the tip passes.
    Only initialization/reset writes qpos. Spring references remain straight.
    """

    timestep = 0.001
    sample_steps = 200
    duration = 22.0
    approach_duration = 0.6
    thread_duration = 0.6
    pull_duration = 1.5
    radius = 0.003
    gripper_constraint = "weld"
    closure_bow = (0.0, -0.50, 0.0)
    camera_lookat = (0.65, 0.10, 0.7)
    camera_distance = 2.5
    camera_elevation = -45

    def __init__(self):
        self.length = 1.6
        self.origin = np.array([0.0, 0.0, 0.6])
        points = np.tile(self.origin, (41, 1))
        points[:, 0] += np.linspace(0, self.length, len(points))
        self.initial_tip = points[-1].copy()
        self.model = self._build_model(points)
        self.model.opt.iterations = 100
        self.model.opt.tolerance = 1e-10
        self.model.dof_damping[6:] = 0.002
        self.model.geom_friction[:, 0] = 1.5
        self.model.geom_margin[:] = 0.001
        self.model.geom_size[0, :2] = (
            2.5  # Render the stage beneath the full straight cable.
        )
        self.data = mj.MjData(self.model)
        self.body_ids = np.array([self.model.body(f"cable_{i}").id for i in range(40)])
        self.mocap_ids = [
            int(self.model.body(name).mocapid[0])
            for name in ("held_gripper", "working_gripper")
        ]
        self.endpoint_ids = [
            self.model.site(name).id for name in ("cable_start", "cable_end")
        ]
        mj.mj_forward(self.model, self.data)
        self._initial_qpos = self.model.qpos0.copy()
        self._initial_mocap = self.data.mocap_pos.copy()
        self.reset()
        if self.progress.error or self.progress.label != "unknot":
            raise RuntimeError(
                f"Straight starting cable must be a valid unknot: {self.status()}"
            )

    def reset(self):
        self.loop_center = None
        self.loop_normal = None
        self._thread_targets = None
        super().reset()

    def _measure_loop(self):
        # This material span encloses the opening for the tuned 40-link model.
        # Measure its frame from simulated positions, never from target points.
        p = self.centerline()
        patch = p[13:28]
        center = patch.mean(axis=0)
        _, singular, axes = np.linalg.svd(patch - center)
        if singular[1] < 0.04:
            raise RuntimeError("Loop opening collapsed before threading")
        normal = axes[-1]
        if np.dot(normal, p[-1] - center) < 0:
            normal = -normal
        self.loop_center, self.loop_normal = center, normal
        self._thread_targets = (
            self.data.mocap_pos[self.mocap_ids[1]].copy(),
            center + 0.15 * normal,
            center - 0.20 * normal,
            center - 0.50 * normal,
        )

    def phase(self):
        t = self.data.time
        if t < 1:
            return "Ready: straight, zero spring strain"
        if t < 13:
            return "Form loop: compress and twist endpoint"
        if t < 13 + self.approach_duration:
            return "Approach measured loop opening"
        if t < 13 + self.approach_duration + self.thread_duration:
            return "Thread through opening"
        if t < 13 + self.approach_duration + self.thread_duration + self.pull_duration:
            return "Pull working end through"
        return "Hold and check"

    def step(self, *, drive=True):
        ready_steps = round(1.0 / self.timestep)
        forming_steps = round(12.0 / self.timestep)
        mocap = self.mocap_ids[1]
        if drive:
            if self.steps < ready_steps:
                self.data.mocap_pos[mocap] = self.initial_tip
            elif self.steps < ready_steps + forming_steps:
                u = (self.steps - ready_steps + 1) / forming_steps
                u = u * u * (3 - 2 * u)
                angle = 3 * np.pi * u
                self.data.mocap_pos[mocap] = self.origin + [
                    self.length * (1 - 0.75 * u),
                    0.1 * np.sin(np.pi * u),
                    0.0,
                ]
                self.data.mocap_quat[mocap] = [
                    np.cos(angle / 2),
                    np.sin(angle / 2),
                    0,
                    0,
                ]
            else:
                if self._thread_targets is None:
                    self._measure_loop()
                a, b, c, d = self._thread_targets
                elapsed = (self.steps - ready_steps - forming_steps + 1) * self.timestep
                approach_end = self.approach_duration
                thread_end = approach_end + self.thread_duration
                if elapsed <= approach_end:
                    target = self._blend(a, b, elapsed / self.approach_duration)
                elif elapsed <= thread_end:
                    target = self._blend(
                        b, c, (elapsed - approach_end) / self.thread_duration
                    )
                else:
                    target = self._blend(
                        c, d, (elapsed - thread_end) / self.pull_duration
                    )
                self.data.mocap_pos[mocap] = target
        mj.mj_step(self.model, self.data)
        mj.mj_kinematics(self.model, self.data)
        self.steps += 1
        if self.steps % self.sample_steps == 0:
            self.sample()


def run_viewer(sim):
    import mujoco.viewer

    keys = SimpleQueue()
    paused = False
    controls = "P: pause/resume | R: reset | gray/red: virtual return valid/invalid"
    print(controls)
    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=keys.put,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = sim.camera_lookat
            viewer.cam.distance = sim.camera_distance
            viewer.cam.azimuth = 90
            viewer.cam.elevation = sim.camera_elevation
        previous_sample = -1
        while viewer.is_running():
            start = time.perf_counter()
            while not keys.empty():
                key = keys.get_nowait()
                if key == ord("P"):
                    paused = not paused
                elif key == ord("R"):
                    sim.reset()
                previous_sample = -1
            if not paused:
                for _ in range(round(1 / (60 * sim.timestep))):
                    sim.step()
                    if time.perf_counter() - start >= 0.010:
                        break
            with viewer.lock():
                sim.draw_closure(viewer.user_scn)
            if previous_sample != len(sim.history):
                status = ("PAUSED\n" if paused else "") + sim.status()
                if hasattr(viewer, "set_texts"):
                    viewer.set_texts(
                        [
                            (
                                mj.mjtFontScale.mjFONTSCALE_100,
                                mj.mjtGridPos.mjGRID_TOPLEFT,
                                status,
                                "",
                            ),
                            (
                                mj.mjtFontScale.mjFONTSCALE_100,
                                mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                                controls
                                + "\nDetection depends on the displayed closure convention.",
                                "",
                            ),
                        ]
                    )
                else:
                    print(status, flush=True)
                previous_sample = len(sim.history)
            viewer.sync()
            time.sleep(max(0, 1 / 60 - (time.perf_counter() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--preformed", action="store_true", help="Run the earlier preformed-loop trial"
    )
    args = parser.parse_args()
    print("Building and validating the starting cable...", flush=True)
    sim = OverhandTyingSim() if args.preformed else StraightOverhandTyingSim()
    if not args.headless:
        run_viewer(sim)
        return
    print(sim.status(), flush=True)
    for _ in range(round(sim.duration / sim.timestep)):
        sim.step()
    print(sim.status(), flush=True)
    if not sim.progress.successful:
        raise RuntimeError("The trial did not finish with a stable overhand match")


if __name__ == "__main__":
    main()
