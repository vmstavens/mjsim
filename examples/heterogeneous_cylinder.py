"""Compress a solid cylinder with a bonded stiff core and soft outer coat.

Run: uv run python examples/heterogeneous_cylinder.py [--headless]
The translucent coat can be hidden with C; this changes only visualization.
Both regions have equal density and extend the full height of the cylinder.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

import numpy as np
from heterogeneous_cube import HeterogeneousCubeSim, run_viewer, vector


class HeterogeneousCylinderSim(HeterogeneousCubeSim):
    outer_radius = 0.06
    core_radius = 0.03
    sectors = 16
    axial_cells = 4

    def _mesh(self):
        # One shared cross-section: a center fan for the solid core, then an
        # annular coat. The core boundary is an actual mesh boundary, not a
        # centroid approximation that could leave elements straddling regions.
        theta = np.arange(self.sectors) * 2 * np.pi / self.sectors
        ring = np.column_stack((np.cos(theta), np.sin(theta)))
        section = np.vstack(
            (np.zeros((1, 2)), self.core_radius * ring, self.outer_radius * ring)
        )
        triangles, soft = [], []
        for i in range(self.sectors):
            j = (i + 1) % self.sectors
            inner_i, inner_j = 1 + i, 1 + j
            outer_i, outer_j = 1 + self.sectors + i, 1 + self.sectors + j
            triangles.extend(
                (
                    (0, inner_i, inner_j),
                    (inner_i, outer_i, outer_j),
                    (inner_i, outer_j, inner_j),
                )
            )
            soft.extend((False, True, True))
        n = len(section)
        points = np.array(
            [
                [x, y, self.base_height + self.side * k / self.axial_cells]
                for k in range(self.axial_cells + 1)
                for x, y in section
            ]
        )
        tetrahedra, materials = [], []
        for k in range(self.axial_cells):
            for triangle, is_soft in zip(triangles, soft):
                # Sort cross-section ids before splitting every prism. This
                # gives matching diagonals on every shared rectangular face.
                a, b, c = np.sort(triangle) + k * n
                for tet in (
                    [a, b, c, c + n],
                    [a, b, b + n, c + n],
                    [a, a + n, b + n, c + n],
                ):
                    p = points[tet]
                    if np.linalg.det(p[1:] - p[0]) < 0:
                        tet[1], tet[2] = tet[2], tet[1]
                    tetrahedra.append(tet)
                    materials.append(is_soft)
        self._material_mask = np.asarray(materials)
        return points, np.asarray(tetrahedra)

    def _soft_element_mask(self):
        return self._material_mask.copy()

    def _xml(self):
        # Reuse the cube's shared-node bodies, lumped masses, elastic material
        # definitions, collision filtering, and load trajectory.
        root = ET.fromstring(super()._xml())
        root.set("model", "Stiff cylindrical core with soft coat")
        base = root.find("worldbody/geom[@type='box']")
        base.set("type", "cylinder")
        base.set("size", vector([0.075, self.base_height / 2]))
        plate = root.find("worldbody/body[@name='plate']/geom")
        plate.set("type", "cylinder")
        plate.set("size", vector([0.075, self.plate_half_height]))
        coat = root.find("deformable/flex[@name='soft']")
        coat.set("rgba", ".95 .46 .16 .28")
        coat.set("group", "1")
        return ET.tostring(root, encoding="unicode")

    def __init__(self):
        super().__init__()
        rest = self.points[self.tetrahedra]
        self._rest_inverse = np.linalg.inv(
            (rest[:, 1:] - rest[:, :1]).transpose(0, 2, 1)
        )

    def deformation_gradients(self):
        p = self.positions()[self.tetrahedra]
        return (p[:, 1:] - p[:, :1]).transpose(0, 2, 1) @ self._rest_inverse

    def _region_means(self, values, weights):
        return np.array(
            [
                np.average(values[mask], weights=weights[mask])
                for mask in (~self.soft_elements, self.soft_elements)
            ]
        )

    def compressive_strains(self):
        """Reference-volume mean axial shortening, ordered core then coat."""
        return self._region_means(
            1 - self.deformation_gradients()[:, 2, 2], self.rest_volumes
        )

    def axial_stresses(self):
        """Current-volume mean compressive Cauchy stress, core then coat (Pa).

        Reconstruct the Saint Venant–Kirchhoff elastic stress from actual vertex
        positions. This excludes damping and contact impulses; interpret it at
        the settled hold, not as a measurement of instantaneous plate force.
        """
        f = self.deformation_gradients()
        strain = (f.transpose(0, 2, 1) @ f - np.eye(3)) / 2
        young = np.where(self.soft_elements, self.soft_young, self.stiff_young)
        mu = young / (2 * (1 + self.poisson))
        lam = young * self.poisson / ((1 + self.poisson) * (1 - 2 * self.poisson))
        second_piola = 2 * mu[:, None, None] * strain
        second_piola += (lam * np.trace(strain, axis1=1, axis2=2))[
            :, None, None
        ] * np.eye(3)
        jacobian = np.linalg.det(f)
        sigma = (f @ second_piola @ f.transpose(0, 2, 1)) / jacobian[:, None, None]
        return self._region_means(-sigma[:, 2, 2], self.rest_volumes * jacobian)

    def status(self):
        core, coat = self.compressive_strains() * 100
        core_stress, coat_stress = self.axial_stresses() / 1000
        return (
            f"{self.data.time:.2f} s | {self.phase()}\n"
            f"Blue core: {self.stiff_young / 1000:g} kPa | shortening {core:.2f}%\n"
            f"Orange coat: {self.soft_young / 1000:g} kPa | shortening {coat:.2f}%\n"
            f"Mean axial elastic stress: core {core_stress:.2f}, coat {coat_stress:.2f} kPa\n"
            f"Both: density {self.density:g} kg/m^3 | bonded radial interface\n"
            "P: pause/resume | R: reset | E: mesh edges | C: show/hide coat"
        )


def run_headless(sim):
    minimum_volume_ratio = 1.0
    hold = None
    for _ in range(round(sim.duration / sim.model.opt.timestep)):
        sim.step()
        if sim.steps % 200 == 0:
            ratio = sim.element_volumes(sim.positions()) / sim.rest_volumes
            if not np.isfinite(ratio).all():
                raise RuntimeError("Nonfinite cylinder element volumes")
            minimum_volume_ratio = min(minimum_volume_ratio, float(ratio.min()))
        if sim.steps == round(7 / sim.model.opt.timestep):
            hold = (sim.compressive_strains(), sim.axial_stresses())
            print(sim.status(), flush=True)
    recovery = np.max(np.abs(sim.positions() - sim.points))
    print(sim.status())
    print(f"Minimum sampled element volume / rest volume: {minimum_volume_ratio:.3f}")
    print(f"Maximum residual vertex displacement: {recovery * 1000:.3f} mm")
    if (
        hold is None
        or not np.all((hold[0] > 0.04) & (hold[0] < 0.09))
        or not (hold[1][0] > 3 * hold[1][1] > 0)
        or minimum_volume_ratio <= 0
        or np.any(sim.data.warning.number)
        or not np.isfinite(sim.data.qpos).all()
        or not np.isfinite(sim.data.qvel).all()
        or not np.isclose(sim.data.time, sim.duration)
        or recovery > 1e-4
    ):
        raise RuntimeError("Cylinder compression/recovery validation failed")
    return hold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="validate one load cycle"
    )
    args = parser.parse_args()
    sim = HeterogeneousCylinderSim()
    print(sim.status(), flush=True)
    if args.headless:
        run_headless(sim)
    else:
        run_viewer(sim, coat_group=1)


if __name__ == "__main__":
    main()
