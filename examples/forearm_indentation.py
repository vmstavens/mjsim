"""Illustrative passive forearm: two rigid bones, muscle, fat, and skin.

Run: uv run python examples/forearm_indentation.py [--headless] [--csv PATH]
Geometry and elastic parameters are illustrative, not patient-specific.
See examples/README.md for sources, assumptions, and numerical checks.
"""

from __future__ import annotations

import argparse
import csv
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue

import mujoco as mj
import numpy as np
from heterogeneous_cube import HeterogeneousCubeSim, vector
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class Tissue:
    density: float  # kg/m^3
    young: float  # Pa
    poisson: float
    rgba: tuple


# Density: COMSOL Bioheat Materials Library (IT'IS-derived values).
# Skin/fat modulus magnitudes: Iivarinen et al. (2011), PMID 21696992;
# transferred to StVK only as illustrative starting values, not a fitted model.
# Muscle modulus, Poisson ratios, and damping are explicit demo assumptions.
TISSUES = {
    "muscle": Tissue(1090, 30000, 0.40, (0.70, 0.18, 0.22, 0.32)),
    "fat": Tissue(911, 1900, 0.45, (0.96, 0.76, 0.25, 0.24)),
    "skin": Tissue(1109, 210000, 0.40, (0.82, 0.52, 0.36, 0.24)),
}
BONE_DENSITY = 1908.0
BONES = {"radius": (-0.012, 0.003, 0.006), "ulna": (0.012, -0.003, 0.005)}  # x,z,r


def inside_polygon(points, polygon):
    edges = np.roll(polygon, -1, axis=0) - polygon
    delta = np.asarray(points)[:, None, :] - polygon[None, :, :]
    return np.all(
        edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0]
        >= -1e-12,
        axis=1,
    )


def extrude(section, triangles, triangle_materials, longitudinal, center_z):
    """Conforming three-tetrahedron prism split, with a mild distal taper."""
    scales = 1 - 0.15 * (longitudinal - longitudinal[0]) / np.ptp(longitudinal)
    points = np.array(
        [
            [scale * x, y, center_z + scale * z]
            for y, scale in zip(longitudinal, scales)
            for x, z in section
        ]
    )
    n = len(section)
    elements, materials = [], []
    for k in range(len(longitudinal) - 1):
        for tri, material in zip(triangles, triangle_materials):
            a, b, c = np.sort(tri) + k * n
            for tet in (
                [a, b, c, c + n],
                [a, b, b + n, c + n],
                [a, a + n, b + n, c + n],
            ):
                p = points[tet]
                if np.linalg.det(p[1:] - p[0]) < 0:
                    tet[1], tet[2] = tet[2], tet[1]
                elements.append(tet)
                materials.append(material)
    return points, np.asarray(elements), np.asarray(materials)


class ForearmSim:
    timestep = 0.0002
    sectors = 24
    bone_sectors = 8
    center_z = 0.08
    # Refine near the indenter, keeping the full model affordable interactively.
    longitudinal = np.array([-0.10, -0.02, 0, 0.02, 0.10])
    outer_axes = np.array([0.035, 0.026])
    skin_thickness = 0.0015
    fat_thickness = 0.004
    probe_radius = 0.012
    gap = 0.003
    indentation = 0.001
    cycle_duration = 6.0
    transfer_duration = 1.0
    duration = 13.0
    sample_steps = 250
    site_names = ("Above radius", "Between bones")
    smoothstep = staticmethod(HeterogeneousCubeSim.smoothstep)

    def __init__(self):
        self.section, self.triangles, labels, self.section_owners = self._section_mesh()
        self.points, self.tetrahedra, self.material_ids = extrude(
            self.section, self.triangles, labels, self.longitudinal, self.center_z
        )
        self.owners = np.tile(self.section_owners, len(self.longitudinal))
        self.rest_volumes = self.element_volumes(self.points)
        if np.any(self.rest_volumes <= 0):
            raise ValueError("Nonpositive rest element volume")
        densities = np.array([t.density for t in TISSUES.values()])[self.material_ids]
        self.vertex_masses = np.zeros(len(self.points))
        np.add.at(
            self.vertex_masses,
            self.tetrahedra.ravel(),
            np.repeat(self.rest_volumes * densities / 4, 4),
        )
        self.model = mj.MjModel.from_xml_string(self._xml())
        self.data = mj.MjData(self.model)
        self.body_ids = np.array(
            [self.model.body(f"node_{i}").id for i in range(len(self.points))]
        )
        self.probe_geom = self.model.geom("probe").id
        self.probe_mocap = int(self.model.body("probe_body").mocapid[0])
        # Bone and end-face vertices are fixed supports. All others deform.
        self.fixed = (self.owners >= 0) | np.isclose(np.abs(self.points[:, 1]), 0.10)
        self.site_x = np.array([BONES["radius"][0] * 0.925, 0.0])
        self.touch_z = np.array([self._find_touch_height(x) for x in self.site_x])
        self.reset()

    def _section_mesh(self):
        theta = np.arange(self.sectors) * 2 * np.pi / self.sectors
        unit = np.column_stack((np.cos(theta), np.sin(theta)))
        muscle_axes = self.outer_axes - self.skin_thickness - self.fat_thickness
        section = list(unit * muscle_axes)
        owners = [-1] * len(section)
        holes = []
        for owner, (x, z, radius) in enumerate(BONES.values()):
            a = np.arange(self.bone_sectors) * 2 * np.pi / self.bone_sectors
            polygon = np.column_stack((x + radius * np.cos(a), z + radius * np.sin(a)))
            holes.append(polygon)
            section.extend(polygon)
            owners.extend([owner] * len(polygon))
        for x in np.linspace(-0.022, 0.022, 5):
            for z in (-0.012, 0, 0.012):
                if (x / muscle_axes[0]) ** 2 + (z / muscle_axes[1]) ** 2 < 0.8 and all(
                    np.hypot(x - bx, z - bz) > radius + 0.002
                    for bx, bz, radius in BONES.values()
                ):
                    section.append((x, z))
                    owners.append(-1)
        section = np.asarray(section)
        triangles = Delaunay(section).simplices
        centers = section[triangles].mean(axis=1)
        keep = ~np.logical_or.reduce([inside_polygon(centers, hole) for hole in holes])
        triangles = triangles[keep].tolist()
        # Unconstrained Delaunay is safe here only if ALL prescribed boundary
        # segments survived. Reject a changed discretization if they did not.
        edges = {
            tuple(sorted(edge))
            for tri in triangles
            for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
        }
        for start, count in (
            (0, self.sectors),
            (self.sectors, self.bone_sectors),
            (self.sectors + self.bone_sectors, self.bone_sectors),
        ):
            for i in range(count):
                if tuple(sorted((start + i, start + (i + 1) % count))) not in edges:
                    raise ValueError(
                        "Triangulation failed to preserve a material boundary"
                    )
        materials = [0] * len(triangles)
        section = section.tolist()
        inner = 0
        for material, axes in (
            (1, self.outer_axes - self.skin_thickness),
            (2, self.outer_axes),
        ):
            outer = len(section)
            section.extend(unit * axes)
            owners.extend([-1] * self.sectors)
            for i in range(self.sectors):
                j = (i + 1) % self.sectors
                triangles.extend(
                    (
                        (inner + i, outer + i, outer + j),
                        (inner + i, outer + j, inner + j),
                    )
                )
                materials.extend((material, material))
            inner = outer
        return (
            np.asarray(section),
            np.asarray(triangles),
            np.asarray(materials),
            np.asarray(owners),
        )

    def element_volumes(self, points):
        p = points[self.tetrahedra]
        return np.linalg.det(p[:, 1:] - p[:, :1]) / 6

    def _xml(self):
        root = ET.Element("mujoco", model="Illustrative forearm indentation")
        ET.SubElement(
            root,
            "option",
            timestep=str(self.timestep),
            gravity="0 0 0",
            integrator="discrete",
            solver="CG",
            iterations="80",
            tolerance="1e-8",
        )
        ET.SubElement(root, "size", memory="128M")
        ET.SubElement(root, "statistic", center="0 0 .08", extent=".4")
        asset = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(
            world,
            "light",
            pos="-.3 -.3 .5",
            dir=".3 .3 -.4",
            ambient=".3 .3 .3",
            castshadow="false",
        )
        ET.SubElement(world, "light", pos=".3 .3 .5", dir="-.3 -.3 -.4")
        ET.SubElement(
            world,
            "geom",
            type="plane",
            size=".5 .5 .01",
            rgba=".14 .16 .19 1",
            contype="0",
            conaffinity="0",
        )
        bone_bodies = []
        self.bone_volumes = []
        for owner, name in enumerate(BONES):
            indices = np.flatnonzero(self.owners == owner)
            vertices = self.points[indices]
            # Convex tapered polygonal prism: exact match to the tissue hole.
            # MuJoCo supplies the convex hull faces for the visual bone geom.
            ET.SubElement(asset, "mesh", name=name, vertex=vector(vertices))
            body = ET.SubElement(world, "body", name=name)
            polygon = self.section[self.section_owners == owner]
            area = (
                abs(
                    np.sum(
                        polygon[:, 0] * np.roll(polygon[:, 1], -1)
                        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
                    )
                )
                / 2
            )
            volume = area * 0.20 * (1 + 0.85 + 0.85**2) / 3
            self.bone_volumes.append(volume)
            ET.SubElement(
                body,
                "geom",
                type="mesh",
                mesh=name,
                density=str(BONE_DENSITY),
                rgba=".94 .92 .80 1",
                contype="0",
                conaffinity="0",
                group="3",
            )
            bone_bodies.append(body)
        for i, (point, mass, owner) in enumerate(
            zip(self.points, self.vertex_masses, self.owners)
        ):
            parent = bone_bodies[owner] if owner >= 0 else world
            body = ET.SubElement(parent, "body", name=f"node_{i}", pos=vector(point))
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=str(mass),
                diaginertia=vector(np.full(3, mass * 1e-5)),
            )
            if owner < 0 and not np.isclose(abs(point[1]), 0.10):
                for axis in np.eye(3):
                    ET.SubElement(body, "joint", type="slide", axis=vector(axis))
        for y in (-0.106, 0.106):
            ET.SubElement(
                world,
                "geom",
                type="box",
                pos=vector([0, y, self.center_z]),
                size=".039 .006 .030",
                rgba=".28 .31 .35 1",
                contype="0",
                conaffinity="0",
            )
        probe = ET.SubElement(
            world, "body", name="probe_body", mocap="true", pos="0 0 .14"
        )
        ET.SubElement(
            probe,
            "geom",
            name="probe",
            type="sphere",
            size=str(self.probe_radius),
            rgba=".20 .65 .95 1",
            contype="2",
            conaffinity="1",
            condim="1",
            solref=".005 1",
            solimp=".95 .99 .0001",
        )
        ET.SubElement(
            probe,
            "geom",
            type="cylinder",
            pos="0 0 .025",
            size=".003 .02",
            rgba=".5 .55 .6 1",
            contype="0",
            conaffinity="0",
        )
        deformable = ET.SubElement(root, "deformable")
        for material, (name, tissue) in enumerate(TISSUES.items()):
            ids, elements = np.unique(
                self.tetrahedra[self.material_ids == material], return_inverse=True
            )
            flex = ET.SubElement(
                deformable,
                "flex",
                name=name,
                dim="3",
                radius="0",
                body=" ".join(f"node_{i}" for i in ids),
                element=vector(elements),
                rgba=vector(tissue.rgba),
                group=str(material),
            )
            ET.SubElement(
                flex,
                "elasticity",
                young=str(tissue.young),
                poisson=str(tissue.poisson),
                damping=".0001",
            )
            ET.SubElement(
                flex,
                "contact",
                contype="1" if name == "skin" else "0",
                conaffinity="2" if name == "skin" else "0",
                condim="1",
                selfcollide="none",
                solref=".005 1",
                solimp=".95 .99 .0001",
            )
        return ET.tostring(root, encoding="unicode")

    def _find_touch_height(self, x):
        """Calibrate zero depth against actual faceted skin/probe collision."""
        low, high = self.center_z + 0.025, self.center_z + 0.06
        for _ in range(26):
            z = (low + high) / 2
            self.data.mocap_pos[self.probe_mocap] = [x, 0, z]
            mj.mj_forward(self.model, self.data)
            touching = any(
                c.geom1 == self.probe_geom or c.geom2 == self.probe_geom
                for c in self.data.contact
            )
            if touching:
                low = z
            else:
                high = z
        return high

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        self.data.mocap_pos[self.probe_mocap] = [
            self.site_x[0],
            0,
            self.touch_z[0] + self.gap,
        ]
        mj.mj_forward(self.model, self.data)
        self.steps = 0
        self.history = []
        self.minimum_volume_ratio = 1.0
        self.sample()

    def trial(self, t=None):
        t = self.data.time if t is None else t
        site = int(t >= self.cycle_duration + self.transfer_duration)
        local = t - site * (self.cycle_duration + self.transfer_duration)
        if self.cycle_duration <= t < self.cycle_duration + self.transfer_duration:
            return -1, "Move to next site", 0.0
        if local < 0.5:
            phase, fraction = "Ready", 0.0
        elif local < 2:
            phase, fraction = "Indent", self.smoothstep((local - 0.5) / 1.5)
        elif local < 3:
            phase, fraction = "Hold", 1.0
        elif local < 4.5:
            phase, fraction = "Release", 1 - self.smoothstep((local - 3) / 1.5)
        else:
            phase, fraction = "Recover", 0.0
        return site, phase, float(fraction)

    def step(self):
        t = (self.steps + 1) * self.timestep
        site, _, fraction = self.trial(t)
        if site < 0:
            u = self.smoothstep((t - self.cycle_duration) / self.transfer_duration)
            x = (1 - u) * self.site_x[0] + u * self.site_x[1]
            z = (1 - u) * self.touch_z[0] + u * self.touch_z[1] + self.gap
        else:
            x = self.site_x[site]
            z = self.touch_z[site] + self.gap - fraction * (self.gap + self.indentation)
        self.data.mocap_pos[self.probe_mocap] = [x, 0, z]
        mj.mj_step(self.model, self.data)
        mj.mj_kinematics(self.model, self.data)
        self.steps += 1
        if self.steps % self.sample_steps == 0:
            self.sample()

    def positions(self):
        return self.data.xpos[self.body_ids].copy()

    def probe_force(self):
        """Signed world-z reaction on the probe from actual solved contacts."""
        force = np.zeros(3)
        local = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            if contact.geom1 == self.probe_geom or contact.geom2 == self.probe_geom:
                mj.mj_contactForce(self.model, self.data, i, local)
                sign = 1 if contact.geom2 == self.probe_geom else -1
                force += sign * (contact.frame.reshape(3, 3).T @ local[:3])
        return float(force[2])

    def sample(self):
        # Refresh the contact solution at the same state as positions/depth.
        mj.mj_forward(self.model, self.data)
        if np.any(self.data.warning.number) or not np.isclose(
            self.data.time, self.steps * self.timestep, atol=1e-8, rtol=0
        ):
            raise RuntimeError("MuJoCo solver warning or unexpected simulation reset")
        site, phase, _ = self.trial()
        depth = (
            0
            if site < 0
            else max(0, self.touch_z[site] - self.data.mocap_pos[self.probe_mocap, 2])
        )
        force = self.probe_force()
        ratios = self.element_volumes(self.positions()) / self.rest_volumes
        if not np.isfinite(ratios).all() or np.min(ratios) <= 0:
            raise RuntimeError("Forearm element inversion or nonfinite volume")
        self.minimum_volume_ratio = min(
            self.minimum_volume_ratio, float(np.min(ratios))
        )
        self.history.append((float(self.data.time), site, phase, float(depth), force))

    def status(self):
        t, site, phase, depth, force = self.history[-1]
        location = "Transfer" if site < 0 else self.site_names[site]
        return (
            f"{t:.2f} s | {location} | {phase}\n"
            f"Indentation: {depth * 1000:.2f} mm | Upward probe force: {force:.3f} N\n"
            "Rigid radius/ulna; bonded muscle, fat, skin | illustrative passive model\n"
            "P: pause | R: restart | E: mesh | 1: skin | 2: fat | 3: muscle"
        )

    def write_csv(self, path):
        with Path(path).open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                ("time_s", "site", "phase", "indentation_m", "probe_force_z_N")
            )
            for t, site, phase, depth, force in self.history:
                writer.writerow(
                    (
                        t,
                        "transfer" if site < 0 else self.site_names[site],
                        phase,
                        depth,
                        force,
                    )
                )


def run_headless(sim):
    for _ in range(round(sim.duration / sim.timestep)):
        sim.step()
        if sim.steps % round(1 / sim.timestep) == 0:
            print(sim.status(), flush=True)
    forces = []
    for site in range(2):
        rows = [row for row in sim.history if row[1] == site and row[2] == "Hold"]
        forces.append(float(np.mean([row[4] for row in rows[-5:]])))
    recovery = float(np.max(np.linalg.norm(sim.positions() - sim.points, axis=1)))
    if (
        not np.isfinite(sim.data.qpos).all()
        or not np.isfinite(sim.data.qvel).all()
        or np.any(sim.data.warning.number)
        or not np.isclose(sim.data.time, sim.duration)
        or not all(np.isfinite(f) and f > 0.001 for f in forces)
        or recovery > 0.0002
        or abs(sim.probe_force()) > 0.001
    ):
        raise RuntimeError(
            f"Forearm validation failed: forces={forces}, recovery={recovery}"
        )
    print(
        f"Hold forces: above radius {forces[0]:.3f} N; between bones {forces[1]:.3f} N"
    )
    print(
        f"Minimum sampled volume ratio: {sim.minimum_volume_ratio:.3f}; recovery: {recovery * 1000:.3f} mm"
    )
    return np.asarray(forces)


def force_figure(sim):
    figure = mj.MjvFigure()
    mj.mjv_defaultFigure(figure)
    figure.title = "Force (N) vs depth"
    figure.xlabel = "Indentation (mm)"
    figure.flg_legend = True
    figure.linename[0] = "Above radius (N)"
    figure.linename[1] = "Between bones (N)"
    figure.linergb[0] = [0.3, 0.7, 1]
    figure.linergb[1] = [1, 0.65, 0.2]
    for site in range(2):
        rows = [
            row
            for row in sim.history
            if row[1] == site and row[2] in ("Indent", "Hold", "Release")
        ]
        limit = figure.linedata.shape[1] // 2
        rows = rows[-limit:]
        figure.linepnt[site] = len(rows)
        figure.linedata[site, : 2 * len(rows)] = np.array(
            [[r[3] * 1000, r[4]] for r in rows]
        ).ravel()
    return figure


def run_viewer(sim):
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
            viewer.cam.lookat[:] = [0, 0, 0.085]
            viewer.cam.distance = 0.48
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
        last_sample = -1
        while viewer.is_running():
            start = time.perf_counter()
            while not keys.empty():
                key = keys.get_nowait()
                if key == ord("P"):
                    paused = not paused
                elif key == ord("R"):
                    sim.reset()
                elif key == ord("E"):
                    with viewer.lock():
                        flag = mj.mjtVisFlag.mjVIS_FLEXEDGE
                        viewer.opt.flags[flag] = not viewer.opt.flags[flag]
                elif key in (ord("1"), ord("2"), ord("3")):
                    group = {ord("1"): 2, ord("2"): 1, ord("3"): 0}[key]
                    with viewer.lock():
                        viewer.opt.flexgroup[group] = not viewer.opt.flexgroup[group]
                last_sample = -1
            if not paused and sim.data.time < sim.duration:
                for _ in range(round(1 / (60 * sim.timestep))):
                    sim.step()
                    if time.perf_counter() - start > 0.010:
                        break
            if last_sample != len(sim.history):
                viewer.set_texts(
                    [
                        (
                            mj.mjtFontScale.mjFONTSCALE_100,
                            mj.mjtGridPos.mjGRID_TOPLEFT,
                            ("PAUSED\n" if paused else "") + sim.status(),
                            "",
                        )
                    ]
                )
                viewport = viewer.viewport
                width = min(400, viewport.width // 2)
                height = min(230, viewport.height // 3)
                viewer.set_figures(
                    [
                        (
                            mj.MjrRect(viewport.width - width, 0, width, height),
                            force_figure(sim),
                        )
                    ]
                )
                last_sample = len(sim.history)
            viewer.sync()
            time.sleep(max(0, 1 / 60 - (time.perf_counter() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="validate both indentation sites"
    )
    parser.add_argument(
        "--csv", type=Path, help="save sampled force-depth history on exit"
    )
    args = parser.parse_args()
    sim = ForearmSim()
    print(sim.status(), flush=True)
    if args.headless:
        run_headless(sim)
    else:
        run_viewer(sim)
    if args.csv:
        sim.write_csv(args.csv)
        print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
