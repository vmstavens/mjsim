"""Four closed cable examples in one native MuJoCo scene. Run with no flags.

The cable is a chain of rigid capsules with compliant ball joints, initialized
in a stress-free knot shape. A connect equality closes the final tip onto the
first body origin. This educational model is not a calibrated elastic rod.
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from queue import SimpleQueue

import mujoco as mj
import numpy as np

from mjsim.utils.knots import (
    FINGERPRINTS,
    KnotClassificationError,
    classify_closed_knot,
)

# Display names and expected closed-curve topology are deliberately separate.
# An overhand with a simple outer return is another embedding of a trefoil.
EXAMPLE_TARGETS = {**{name: name for name in FINGERPRINTS}, "overhand": "trefoil"}


def overhand_core_count(count):
    return round(0.7 * count)


def knot_points(knot: str, count: int = 48) -> np.ndarray:
    """Sample standard embedded curves; do not duplicate the endpoint."""
    if count < 24:
        raise ValueError("Use at least 24 cable segments")
    t = np.linspace(0, 2 * np.pi, count, endpoint=False)
    if knot == "unknot":
        p = np.column_stack((np.cos(t), np.sin(t), np.zeros(count)))
    elif knot == "trefoil":
        p = np.column_stack(
            (
                (2 + np.cos(3 * t)) * np.cos(2 * t),
                (2 + np.cos(3 * t)) * np.sin(2 * t),
                np.sin(3 * t),
            )
        )
    elif knot == "overhand":
        # Open the outer lobe of a trefoil, keeping its three-crossing core.
        # Join the two tails with a broad, physical outer arc. This explicit
        # construction is not a general virtual-closure algorithm.
        core_count = overhand_core_count(count)
        t = np.linspace(0.6, 2 * np.pi - 0.6, core_count)
        core = np.column_stack(
            (
                (2 + np.cos(3 * t)) * np.cos(2 * t),
                (2 + np.cos(3 * t)) * np.sin(2 * t),
                np.sin(3 * t),
            )
        )
        theta = np.linspace(0, np.pi, count - core_count + 2)[1:-1]
        end = core[-1]
        outer_return = np.column_stack(
            (end[0] + 3 * np.sin(theta), end[1] * np.cos(theta), end[2] * np.cos(theta))
        )
        p = np.vstack((core, outer_return))
        p[:, :2] = p[:, :2] @ np.array([[0, -1], [1, 0]])
        p -= (p.min(axis=0) + p.max(axis=0)) / 2
    elif knot == "figure-eight":
        p = np.column_stack(
            (
                (2 + np.cos(2 * t)) * np.cos(3 * t),
                (2 + np.cos(2 * t)) * np.sin(3 * t),
                np.sin(4 * t),
            )
        )
    else:
        raise ValueError(f"Unknown knot: {knot}")
    p *= 0.6 / np.max(np.ptp(p, axis=0))
    p[:, 2] += 0.4
    return p


def _vector(values):
    return " ".join(format(float(v), ".16g") for v in values)


class KnotSim:
    def __init__(self, knot=None, count=48):
        self.knots = (knot,) if knot is not None else tuple(EXAMPLE_TARGETS)
        root = ET.Element("mujoco", model="closed_knots")
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(
            root,
            "option",
            timestep="0.001",
            gravity="0 0 0",
            integrator="implicitfast",
            iterations="100",
            tolerance="1e-10",
        )
        ET.SubElement(root, "statistic", center="0 0 0.4", extent="3.0")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
        ET.SubElement(
            world, "geom", type="plane", size="2 1 0.01", rgba="0.12 0.14 0.16 1"
        )
        contact = ET.SubElement(root, "contact")
        equality = ET.SubElement(root, "equality")
        colors = (
            (0.3, 0.75, 0.95, 1),
            (0.95, 0.4, 0.12, 1),
            (0.7, 0.5, 0.95, 1),
            (0.95, 0.75, 0.2, 1),
        )
        self.label_positions = {}
        for index, name in enumerate(self.knots):
            x, y = ((index % 2 - 0.5) * 0.95, (0.5 - index // 2) * 0.95)
            if len(self.knots) == 1:
                x, y = 0.0, 0.0
            points = knot_points(name, count) + [x, y, 0]
            self.label_positions[name] = np.array([x, y - 0.38, 0.4])
            self._add_cable(world, contact, equality, points, name, colors[index])
        self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mj.MjData(self.model)
        self.cable_body_ids = {
            name: np.array(
                [self.model.body(f"{name}/cable_{i}").id for i in range(count)]
            )
            for name in self.knots
        }
        # Retain direct access for single-cable numerical experiments.
        self.body_ids = self.cable_body_ids[self.knots[0]]
        self.ball_dofs = np.concatenate(
            [
                np.arange(address, address + 3)
                for kind, address in zip(self.model.jnt_type, self.model.jnt_dofadr)
                if kind == mj.mjtJoint.mjJNT_BALL
            ]
        )
        self.reset()

    @staticmethod
    def _add_cable(world, contact, equality, points, name, color):
        count = len(points)
        parent = world
        radius = min(
            0.003,
            0.15 * np.min(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)),
        )
        for i, point in enumerate(points):
            offset = point if i == 0 else point - points[i - 1]
            body = ET.SubElement(
                parent, "body", name=f"{name}/cable_{i}", pos=_vector(offset)
            )
            if i == 0:
                ET.SubElement(body, "freejoint")
                ET.SubElement(
                    body,
                    "site",
                    name=f"{name}/start",
                    size="0.006",
                    rgba="0.3 0.9 0.5 1",
                )
            else:
                ET.SubElement(
                    body, "joint", type="ball", damping="0.002", stiffness="0.02"
                )
            delta = points[(i + 1) % count] - point
            ET.SubElement(
                body,
                "geom",
                type="capsule",
                size=str(radius),
                fromto=_vector(np.r_[np.zeros(3), delta]),
                mass=str(0.2 / count),
                rgba=_vector(
                    (0.55, 0.6, 0.65, 1)
                    if name == "overhand" and i >= overhand_core_count(count) - 1
                    else color
                ),
                friction="0.5 0.01 0.001",
                solref="0.004 1",
            )
            parent = body
        ET.SubElement(
            parent,
            "site",
            name=f"{name}/end",
            pos=_vector(points[0] - points[-1]),
            size="0.004",
            rgba="1 1 1 1",
        )
        ET.SubElement(
            contact,
            "exclude",
            body1=f"{name}/cable_0",
            body2=f"{name}/cable_{count - 1}",
        )
        ET.SubElement(
            equality,
            "connect",
            site1=f"{name}/start",
            site2=f"{name}/end",
            solref="0.002 1",
        )

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

    def disturb(self):
        """Small joint velocity impulse to demonstrate shape changes."""
        self.data.qvel[self.ball_dofs] += np.random.default_rng(42).normal(
            0, 0.15, len(self.ball_dofs)
        )

    def step(self):
        mj.mj_step(self.model, self.data)
        # Refresh body/site positions without repeating the dynamics and solver.
        mj.mj_kinematics(self.model, self.data)

    def _name(self, knot):
        if knot is None:
            if len(self.knots) != 1:
                raise ValueError(
                    "Specify which cable to sample in the multi-knot scene"
                )
            return self.knots[0]
        if knot not in self.knots:
            raise ValueError(f"No cable named {knot!r}")
        return knot

    def closure_gap(self, knot=None):
        name = self._name(knot)
        start = self.data.site(f"{name}/start").xpos
        end = self.data.site(f"{name}/end").xpos
        return float(np.linalg.norm(start - end))

    def centerline(self, knot=None):
        name = self._name(knot)
        if self.closure_gap(name) > 1e-4:
            raise KnotClassificationError("Physical closure gap exceeds 0.1 mm")
        points = self.data.xpos[self.cable_body_ids[name]].copy()
        tip = self.data.site(f"{name}/end").xpos
        if np.linalg.norm(tip - points[0]) > 1e-8 * np.max(np.ptp(points, axis=0)):
            points = np.vstack((points, tip))
        return points

    def status(self):
        rows = []
        for name in self.knots:
            try:
                result = classify_closed_knot(self.centerline(name))
                values = ", ".join(f"{v:.4g}" for v in result.alexander_roots)
                rows.append(
                    f"{name}: match={result.knot == EXAMPLE_TARGETS[name]} | detected={result.knot or 'unknown'}\n"
                    f"  Alexander: ({values}) | closure: {self.closure_gap(name) * 1000:.4f} mm"
                )
            except KnotClassificationError as error:
                rows.append(f"{name}: classification unavailable: {error}")
        return "\n\n".join(rows)

    def add_labels(self, scene):
        for name, position in self.label_positions.items():
            geom = scene.geoms[scene.ngeom]
            mj.mjv_initGeom(
                geom,
                mj.mjtGeom.mjGEOM_LABEL,
                np.zeros(3),
                position,
                np.eye(3).ravel(),
                np.ones(4),
            )
            geom.label = "overhand (closed)" if name == "overhand" else name
            scene.ngeom += 1


def run_viewer(sim: KnotSim) -> None:
    """Run native MuJoCo rendering and process keyboard input between steps."""
    import mujoco.viewer

    controls = "D: disturb all cables | R: reset all | P: pause"
    print(controls)
    print("Fingerprint matches are not general knot-equivalence proofs.")
    keys = SimpleQueue()
    paused = False
    next_update = 0.0
    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=keys.put,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0, 0.4]
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -70
            sim.add_labels(viewer.user_scn)
        while viewer.is_running():
            start = time.perf_counter()
            # The passive viewer synchronizes model/data in sync(). Keep
            # physics and classification outside its rendering mutex.
            while not keys.empty():
                key = keys.get_nowait()
                if key == ord("D"):
                    sim.disturb()
                elif key == ord("R"):
                    sim.reset()
                elif key == ord("P"):
                    paused = not paused
                next_update = 0.0
            if not paused:
                # Bound physics work per frame. On slower machines simulation
                # time advances more slowly instead of starving viewer updates.
                deadline = start + 0.010
                for _ in range(max(1, round(1 / (60 * sim.model.opt.timestep)))):
                    sim.step()
                    if time.perf_counter() >= deadline:
                        break
            if start >= next_update:
                status = sim.status()
                next_update = start + 0.25
            else:
                status = None
            if status is not None:
                if hasattr(viewer, "set_texts"):
                    viewer.set_texts(
                        [
                            (
                                mj.mjtFontScale.mjFONTSCALE_100,
                                mj.mjtGridPos.mjGRID_TOPLEFT,
                                ("Paused\n" if paused else "Running\n") + status,
                                "",
                            ),
                            (
                                mj.mjtFontScale.mjFONTSCALE_100,
                                mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                                controls
                                + "\nFingerprint match only; not an equivalence proof.",
                                "",
                            ),
                        ]
                    )
                else:
                    # Older MuJoCo releases do not expose custom text overlays.
                    print(status, flush=True)
            viewer.sync()
            time.sleep(max(0, 1 / 60 - (time.perf_counter() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=48)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--steps", type=int, default=500, help="Headless simulation steps"
    )
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    sim = KnotSim(count=args.segments)
    if args.headless:
        sim.disturb()
        for _ in range(args.steps):
            sim.step()
        # Fail visibly in automation if classification is inconclusive.
        for name in sim.knots:
            classify_closed_knot(sim.centerline(name))
        print(sim.status())
        return

    run_viewer(sim)


if __name__ == "__main__":
    main()
