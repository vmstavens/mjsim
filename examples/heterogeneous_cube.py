"""Compress a bonded, equal-density cube with two different elastic moduli.

Run with ``uv run python examples/heterogeneous_cube.py`` for the native MuJoCo
viewer, or add --headless for one load cycle.
The bottom face is clamped. Zero gravity isolates the imposed compression from
self-weight. Shared vertex bodies bond the two tetrahedral flex regions exactly.
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from itertools import permutations, product
from queue import SimpleQueue

import mujoco as mj
import numpy as np


def vector(values):
    return " ".join(str(value) for value in np.asarray(values).ravel())


class HeterogeneousCubeSim:
    side = 0.12
    density = 1000.0
    soft_young = 5000.0
    stiff_young = 25000.0
    poisson = 0.25
    cells = 4
    base_height = 0.025
    plate_half_height = 0.008
    plate_gap = 0.004
    compression = 0.008
    duration = 13.0

    def __init__(self):
        self.points, self.tetrahedra = self._mesh()
        self.rest_volumes = self.element_volumes(self.points)
        self.soft_elements = self._soft_element_mask()
        # Lump each tetrahedron's mass equally to its four vertices, including
        # contributions from BOTH regions at their shared interface.
        self.vertex_masses = np.zeros(len(self.points))
        np.add.at(
            self.vertex_masses,
            self.tetrahedra.ravel(),
            np.repeat(self.density * self.rest_volumes / 4, 4),
        )
        self.model = mj.MjModel.from_xml_string(self._xml())
        self.data = mj.MjData(self.model)
        self.body_ids = np.array(
            [self.model.body(f"node_{i}").id for i in range(len(self.points))]
        )
        self.plate_id = int(self.model.body("plate").mocapid[0])
        self.layers = [
            np.flatnonzero(np.isclose(self.points[:, 2], self.base_height + z))
            for z in (0, self.side / 2, self.side)
        ]
        self.reset()

    def _soft_element_mask(self):
        return self.points[self.tetrahedra, 2].mean(axis=1) > (
            self.base_height + self.side / 2
        )

    def _mesh(self):
        n = self.cells + 1
        points = np.array(list(product(range(n), repeat=3)), dtype=float)
        points *= self.side / self.cells
        points += [-self.side / 2, -self.side / 2, self.base_height]
        tetrahedra = []
        # Six consistently triangulated tetrahedra per grid cell (Freudenthal).
        # Identical face diagonals across neighboring cells avoid mesh cracks.
        for cell in product(range(self.cells), repeat=3):
            for order in permutations(range(3)):
                vertex = np.array(cell)
                corners = [np.ravel_multi_index(vertex, (n, n, n))]
                for axis in order:
                    vertex[axis] += 1
                    corners.append(np.ravel_multi_index(vertex, (n, n, n)))
                p = points[corners]
                if np.linalg.det(p[1:] - p[0]) < 0:
                    corners[1], corners[2] = corners[2], corners[1]
                tetrahedra.append(corners)
        return points, np.asarray(tetrahedra)

    def element_volumes(self, points):
        p = points[self.tetrahedra]
        return np.linalg.det(p[:, 1:] - p[:, :1]) / 6

    def _xml(self):
        root = ET.Element("mujoco", model="Bonded soft and stiff cube")
        # MuJoCo 3.17 requires discrete integration for flex elasticity.
        ET.SubElement(
            root,
            "option",
            timestep="0.0005",
            gravity="0 0 0",
            integrator="discrete",
            solver="Newton",
            iterations="50",
            tolerance="1e-9",
        )
        ET.SubElement(root, "statistic", center="0 0 0.09", extent="0.45")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="0 -0.3 0.8", dir="0 0 -1")
        ET.SubElement(
            world,
            "light",
            pos="-.3 .4 .3",
            dir=".3 -.4 -.2",
            ambient=".35 .35 .35",
            diffuse=".6 .6 .6",
            castshadow="false",
        )
        ET.SubElement(
            world,
            "geom",
            type="plane",
            size="1 1 .01",
            rgba=".16 .18 .21 1",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            world,
            "geom",
            type="box",
            pos="0 0 .0125",
            size=".09 .09 .0125",
            rgba=".35 .38 .42 1",
            contype="0",
            conaffinity="0",
        )
        for i, (point, mass) in enumerate(zip(self.points, self.vertex_masses)):
            body = ET.SubElement(world, "body", name=f"node_{i}", pos=vector(point))
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=str(mass),
                diaginertia=vector(np.full(3, mass * 1e-5)),
            )
            if not np.isclose(point[2], self.base_height):
                for axis in np.eye(3):
                    ET.SubElement(body, "joint", type="slide", axis=vector(axis))
        self.plate_start = (
            self.base_height + self.side + self.plate_gap + self.plate_half_height
        )
        plate = ET.SubElement(
            world,
            "body",
            name="plate",
            mocap="true",
            pos=vector([0, 0, self.plate_start]),
        )
        ET.SubElement(
            plate,
            "geom",
            name="press",
            type="box",
            size=vector([0.085, 0.085, self.plate_half_height]),
            rgba=".75 .78 .82 1",
            contype="1",
            conaffinity="2",
            condim="1",
            solref=".005 1",
            solimp=".95 .99 .001",
        )
        deformable = ET.SubElement(root, "deformable")
        for name, mask, young, color in (
            ("stiff", ~self.soft_elements, self.stiff_young, ".20 .46 .85 1"),
            ("soft", self.soft_elements, self.soft_young, ".95 .46 .16 1"),
        ):
            elements = self.tetrahedra[mask]
            vertices, local_elements = np.unique(elements, return_inverse=True)
            flex = ET.SubElement(
                deformable,
                "flex",
                name=name,
                dim="3",
                radius="0",
                body=" ".join(f"node_{i}" for i in vertices),
                element=vector(local_elements),
                rgba=color,
            )
            ET.SubElement(
                flex,
                "elasticity",
                young=str(young),
                poisson=str(self.poisson),
                damping="0.002",
            )
            # Regions are already bonded, so exclude artificial contact across
            # their common face. Both can still collide with the pressing plate.
            ET.SubElement(
                flex,
                "contact",
                contype="2",
                conaffinity="1",
                condim="1",
                selfcollide="none",
                solref=".005 1",
                solimp=".95 .99 .001",
            )
        return ET.tostring(root, encoding="unicode")

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        self.steps = 0

    @staticmethod
    def smoothstep(u):
        u = np.clip(u, 0, 1)
        return u**3 * (10 + u * (-15 + 6 * u))

    def press_fraction(self, t):
        if t < 5:
            return self.smoothstep((t - 1) / 4)
        if t < 7:
            return 1.0
        return 1 - self.smoothstep((t - 7) / 4)

    def phase(self):
        t = self.data.time
        if t < 1:
            return "Ready"
        if t < 5:
            return "Compress"
        if t < 7:
            return "Hold"
        if t < 11:
            return "Release"
        return "Recovered (R to repeat)"

    def step(self):
        target_time = (self.steps + 1) * self.model.opt.timestep
        travel = (self.plate_gap + self.compression) * self.press_fraction(target_time)
        self.data.mocap_pos[self.plate_id, 2] = self.plate_start - travel
        mj.mj_step(self.model, self.data)
        mj.mj_kinematics(self.model, self.data)
        self.steps += 1

    def positions(self):
        return self.data.xpos[self.body_ids].copy()

    def compressive_strains(self):
        """Mean layer shortening / original layer height, ordered stiff, soft."""
        heights = [
            self.data.xpos[self.body_ids[layer], 2].mean() for layer in self.layers
        ]
        return 1 - np.diff(heights) / (self.side / 2)

    def status(self):
        stiff, soft = self.compressive_strains() * 100
        return (
            f"{self.data.time:.2f} s | {self.phase()}\n"
            f"Orange: soft {self.soft_young / 1000:g} kPa | shortening {soft:.2f}%\n"
            f"Blue: stiff {self.stiff_young / 1000:g} kPa | shortening {stiff:.2f}%\n"
            f"Both: density {self.density:g} kg/m^3 | bonded interface\n"
            "P: pause/resume | R: reset | E: mesh edges"
        )


def run_viewer(sim, *, coat_group=None):
    import mujoco.viewer

    keys = SimpleQueue()
    paused = False
    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=keys.put,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0, 0.09]
            viewer.cam.distance = 0.48
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        last_sample = -1
        while viewer.is_running():
            start = time.perf_counter()
            while not keys.empty():
                key = keys.get_nowait()
                if key == ord("P"):
                    paused = not paused
                    last_sample = -1
                elif key == ord("R"):
                    sim.reset()
                    last_sample = -1
                elif key == ord("E"):
                    with viewer.lock():
                        flag = mj.mjtVisFlag.mjVIS_FLEXEDGE
                        viewer.opt.flags[flag] = not viewer.opt.flags[flag]
                elif key == ord("C") and coat_group is not None:
                    with viewer.lock():
                        viewer.opt.flexgroup[coat_group] = not viewer.opt.flexgroup[
                            coat_group
                        ]
            if not paused and sim.data.time < sim.duration:
                for _ in range(round(1 / (60 * sim.model.opt.timestep))):
                    sim.step()
                    if time.perf_counter() - start > 0.01:
                        break
            sample = int(sim.data.time * 5)
            if sample != last_sample:
                status = ("PAUSED\n" if paused else "") + sim.status()
                if hasattr(viewer, "set_texts"):
                    viewer.set_texts(
                        [
                            (
                                mj.mjtFontScale.mjFONTSCALE_100,
                                mj.mjtGridPos.mjGRID_TOPLEFT,
                                status,
                                "",
                            )
                        ]
                    )
                elif sample % 5 == 0:
                    print(status, flush=True)
                last_sample = sample
            viewer.sync()
            time.sleep(max(0, 1 / 60 - (time.perf_counter() - start)))


def run_headless(sim):
    hold_strains = None
    minimum_volume_ratio = 1.0
    for _ in range(round(sim.duration / sim.model.opt.timestep)):
        sim.step()
        if sim.steps % 200 == 0:
            minimum_volume_ratio = min(
                minimum_volume_ratio,
                float(np.min(sim.element_volumes(sim.positions()) / sim.rest_volumes)),
            )
        if sim.steps == round(7 / sim.model.opt.timestep):
            hold_strains = sim.compressive_strains()
            print(sim.status())
    print(sim.status())
    print(f"Minimum sampled element volume / rest volume: {minimum_volume_ratio:.3f}")
    if (
        hold_strains is None
        or not (hold_strains[1] > 2 * hold_strains[0] > 0)
        or minimum_volume_ratio <= 0
        or np.any(sim.data.warning.number)
        or not np.isfinite(sim.data.qpos).all()
        or not np.isclose(sim.data.time, sim.duration)
        or np.max(np.abs(sim.compressive_strains())) > 0.005
    ):
        raise RuntimeError("Compression/recovery validation failed")
    return hold_strains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="validate one load cycle"
    )
    args = parser.parse_args()
    sim = HeterogeneousCubeSim()
    print(sim.status())
    if args.headless:
        run_headless(sim)
    else:
        run_viewer(sim)


if __name__ == "__main__":
    main()
