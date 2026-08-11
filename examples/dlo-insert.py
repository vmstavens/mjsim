"""DLO pipe insertion demo rendered with Viser."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco as mj
import numpy as np

from mjsim.utils.mjs import cable, empty_scene

EXAMPLES_DIR = Path(__file__).resolve().parent
removed_paths = [
    path for path in sys.path if Path(path or ".").resolve() == EXAMPLES_DIR
]
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != EXAMPLES_DIR]
import viser  # noqa: E402
from mjviser import Viewer  # noqa: E402

sys.path[:0] = removed_paths


CABLE_MODEL_NAME = "dlo"
CABLE_ATTACH_PREFIX = "cable/"
CABLE_SITE_FIRST = f"{CABLE_ATTACH_PREFIX}{CABLE_MODEL_NAME}:S_first"
CABLE_SITE_LAST = f"{CABLE_ATTACH_PREFIX}{CABLE_MODEL_NAME}:S_last"
MOCAP_BODY = "mocap"
MOCAP_SITE = "mocap_site"
PIPE_CENTER = np.array([0.0, 0.08, 0.13])
FIGURE_CENTER = np.array([0.0, -0.045, 0.14])
RZ_MINUS_90 = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

PALETTE = {
    "floor": [0.88, 0.88, 0.84, 1.0],
    "pipe": [0.18, 0.52, 0.85, 0.34],
    "cable": [0.95, 0.48, 0.10, 1.0],
}


@dataclass(frozen=True)
class PipeSpec:
    inner_radius: float = 0.012
    wall_thickness: float = 0.006
    length: float = 0.16
    segments: int = 48

    @property
    def center_radius(self) -> float:
        return self.inner_radius + 0.5 * self.wall_thickness

    @property
    def outer_radius(self) -> float:
        return self.inner_radius + self.wall_thickness


class DloInsertSim:
    """Small MuJoCo scene for visualizing a stationary mocap-held DLO at a pipe."""

    def __init__(self) -> None:
        self.step_count = 0
        self.initial_mocap_pos = np.array([-0.0025, -0.30, PIPE_CENTER[2]])
        self.model, self.data = self._build_model()
        self.mocap_id = int(self.model.body_mocapid[self.model.body(MOCAP_BODY).id])
        self.reset()

    def _build_model(self) -> tuple[mj.MjModel, mj.MjData]:
        scene = empty_scene(
            sim_name="dlo_insert_viser",
            memory="100M",
            timestep=0.001,
            tolerance=1e-8,
            iterations=80,
            solver="Newton",
            nativeccd=True,
            gravity=(0.0, 0.0, 0.0),
            statistic_center=FIGURE_CENTER,
            statistic_extent=0.28,
            statistic_meansize=0.025,
            add_assets=False,
            add_light=False,
            add_floor=False,
            visual_overrides={
                "headlight": {
                    "diffuse": [0.45, 0.45, 0.45],
                    "ambient": [0.30, 0.30, 0.30],
                    "specular": [0.18, 0.18, 0.18],
                },
                "global_": {
                    "azimuth": 140,
                    "elevation": -18,
                    "offwidth": 3200,
                    "offheight": 2400,
                },
                "rgba": {"haze": [0.86, 0.89, 0.92, 1.0]},
            },
        )

        self._add_stage(scene)
        self._add_pipe(scene, PipeSpec(), center=PIPE_CENTER)
        self._add_mocap_gripper(scene)
        self._add_cable(scene)

        scene.add_equality(
            name="mocap_to_dlo",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1=MOCAP_SITE,
            name2=CABLE_SITE_FIRST,
            solref=[0.0005, 1.0],
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        )

        model = scene.compile()
        data = mj.MjData(model)
        return model, data

    def _add_stage(self, scene: mj.MjSpec) -> None:
        scene.worldbody.add_geom(
            name="paper_floor",
            type=mj.mjtGeom.mjGEOM_PLANE,
            size=[0.70, 0.70, 0.02],
            rgba=PALETTE["floor"],
            solimp=[0.0, 0.0, 0.0, 0.0, 1.0],
        )
        scene.worldbody.add_light(
            name="key_light",
            pos=[-0.25, -0.45, 0.80],
            dir=[0.35, 0.55, -1.0],
            diffuse=[0.85, 0.83, 0.78],
            specular=[0.24, 0.24, 0.24],
            type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        scene.worldbody.add_light(
            name="rim_light",
            pos=[0.35, 0.18, 0.55],
            dir=[-0.30, -0.15, -1.0],
            diffuse=[0.28, 0.38, 0.48],
            specular=[0.08, 0.08, 0.08],
            type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
        )

    def _add_pipe(
        self,
        scene: mj.MjSpec,
        pipe: PipeSpec,
        *,
        center: Sequence[float],
    ) -> None:
        center = np.asarray(center, dtype=float)
        arc_half_width = math.pi * pipe.center_radius / pipe.segments
        half_length = 0.5 * pipe.length
        half_thickness = 0.5 * pipe.wall_thickness

        scene.worldbody.add_site(
            name="target",
            pos=[center[0], center[1] + half_length, center[2]],
            size=[pipe.inner_radius],
            rgba=[0.10, 0.80, 0.25, 0.0],
            group=3,
        )

        for index in range(pipe.segments):
            theta = 2.0 * math.pi * index / pipe.segments
            radial = np.array([math.cos(theta), 0.0, math.sin(theta)])
            pos = center + radial * pipe.center_radius

            # Rotate the box so local x is radial, local y is the pipe axis, and
            # local z is tangential around the pipe.
            quat = [
                math.cos(-theta / 2.0),
                0.0,
                math.sin(-theta / 2.0),
                0.0,
            ]
            scene.worldbody.add_geom(
                name=f"pipe_wall_{index:02d}",
                type=mj.mjtGeom.mjGEOM_BOX,
                pos=pos.tolist(),
                quat=quat,
                size=[half_thickness, half_length, arc_half_width],
                rgba=PALETTE["pipe"],
                friction=[0.6, 0.02, 0.001],
                condim=4,
                solref=[0.001, 1.0],
                solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
            )

        for y, name in (
            (center[1] - half_length, "pipe_entry"),
            (center[1] + half_length, "pipe_exit"),
        ):
            scene.worldbody.add_site(
                name=name,
                pos=[center[0], y, center[2]],
                size=[pipe.outer_radius],
                rgba=[1.0, 0.7, 0.1, 0.0],
                group=3,
            )

    def _add_mocap_gripper(self, scene: mj.MjSpec) -> None:
        mocap = scene.worldbody.add_body(
            name=MOCAP_BODY,
            mocap=True,
            pos=self.initial_mocap_pos.tolist(),
        )
        mocap.add_site(
            name=MOCAP_SITE,
            size=[0.01],
            rgba=[1.0, 0.2, 0.1, 0.0],
            group=3,
        )

    def _add_cable(self, scene: mj.MjSpec) -> None:
        dlo = cable(
            model_name=CABLE_MODEL_NAME,
            n_segments=12,
            length=0.40,
            curve="0 s 0",
            twist=180_000.0,
            bend=3_000_000.0,
            mass=0.045,
            segment_size=0.0035,
            geom_rgba=PALETTE["cable"],
            geom_friction="0.9 0.02 0.001",
            geom_condim=4,
            geom_solref="0.001 1",
            joint_damping="0.03",
        )
        cable_frame = scene.worldbody.add_frame(pos=self.initial_mocap_pos.tolist())
        scene.attach(dlo, prefix=CABLE_ATTACH_PREFIX, frame=cable_frame)

    def reset(self) -> None:
        mj.mj_resetData(self.model, self.data)
        self.data.mocap_pos[self.mocap_id] = self.initial_mocap_pos
        self.data.mocap_quat[self.mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        self.step_count = 0
        mj.mj_forward(self.model, self.data)

    def step(self) -> None:
        mj.mj_step(self.model, self.data)
        self.step_count += 1

    def status(self) -> str:
        tip = self.data.site_xpos[self.model.site(CABLE_SITE_LAST).id]
        target = self.data.site_xpos[self.model.site("target").id]
        error = np.linalg.norm(tip - target)
        return (
            f"time: {self.data.time: .3f} s\n"
            f"tip: [{tip[0]: .3f}, {tip[1]: .3f}, {tip[2]: .3f}] m\n"
            f"tip-target distance: {error: .3f} m\n"
            f"contacts: {self.data.ncon}"
        )


def _mat_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mj.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quat


class PaperVisuals:
    """Viser-only composition aids for screenshots and paper figures."""

    def __init__(self, server: viser.ViserServer, sim: DloInsertSim) -> None:
        self.server = server
        self.sim = sim
        self.visible = True
        self.frames_visible = True
        self.static_handles = []
        self.scene_offset = np.zeros(3)
        self.frame_site_ids = {
            "cable_base": self.sim.model.site(CABLE_SITE_FIRST).id,
            "tip": self.sim.model.site(CABLE_SITE_LAST).id,
            "pipe_input": self.sim.model.site("pipe_entry").id,
            "pipe_output": self.sim.model.site("pipe_exit").id,
            "tcp": self.sim.model.site(MOCAP_SITE).id,
        }
        self.frame_orientation_corrections = {
            "cable_base": RZ_MINUS_90,
            "tip": RZ_MINUS_90,
        }
        self.frame_handles = {}

        self._add_lighting()
        self._add_static_overlays()
        self._add_site_frames()

    def _add_lighting(self) -> None:
        self.server.scene.add_light_ambient(
            "/paper/lights/ambient",
            color=(245, 246, 247),
            intensity=0.65,
        )
        self.server.scene.add_light_hemisphere(
            "/paper/lights/hemi",
            sky_color=(212, 225, 240),
            ground_color=(214, 211, 202),
            intensity=0.55,
        )
        self.server.scene.add_light_directional(
            "/paper/lights/key",
            color=(255, 246, 230),
            intensity=1.35,
            cast_shadow=True,
            position=(-0.30, -0.35, 0.75),
        )

    def _add_static_overlays(self) -> None:
        self.static_handles.append(
            self.server.scene.add_grid(
                "/paper/floor_grid",
                width=0.54,
                height=0.54,
                plane="xy",
                cell_size=0.04,
                cell_color=(205, 207, 204),
                cell_thickness=0.55,
                section_size=0.16,
                section_color=(178, 181, 180),
                section_thickness=0.85,
                plane_opacity=0.0,
                shadow_opacity=0.18,
                position=(0.0, 0.0, 0.001),
            )
        )

    def _add_site_frames(self) -> None:
        for name, site_id in self.frame_site_ids.items():
            self.frame_handles[name] = self.server.scene.add_frame(
                f"/paper/frames/{name}",
                axes_length=0.035,
                axes_radius=0.0012,
                origin_radius=0.003,
                position=self.sim.data.site_xpos[site_id] + self.scene_offset,
                wxyz=self._frame_wxyz(name, site_id),
                visible=self.visible and self.frames_visible,
            )

    def _frame_wxyz(self, name: str, site_id: int) -> np.ndarray:
        matrix = self.sim.data.site_xmat[site_id].reshape(3, 3)
        correction = self.frame_orientation_corrections.get(name)
        if correction is not None:
            matrix = matrix @ correction
        return _mat_to_wxyz(matrix)

    def _update_site_frames(self, scene_offset: np.ndarray | None = None) -> None:
        if scene_offset is not None:
            self.scene_offset = scene_offset
        for name, site_id in self.frame_site_ids.items():
            handle = self.frame_handles[name]
            handle.position = self.sim.data.site_xpos[site_id] + self.scene_offset
            handle.wxyz = self._frame_wxyz(name, site_id)
            handle.visible = self.visible and self.frames_visible

    def set_visible(self, visible: bool) -> None:
        self.visible = visible
        for handle in self.static_handles:
            handle.visible = visible
        self._update_site_frames()

    def set_frames_visible(self, visible: bool) -> None:
        self.frames_visible = visible
        self._update_site_frames()

    def update(self, scene_offset: np.ndarray | None = None) -> None:
        self._update_site_frames(scene_offset)


def add_gui(
    server: viser.ViserServer,
    sim: DloInsertSim,
    visuals: PaperVisuals | None = None,
):
    server.gui.set_panel_label("DLO Insert")
    readout = server.gui.add_text("State", sim.status(), multiline=True, disabled=True)
    reset_button = server.gui.add_button(
        "Reset simulation",
        color="gray",
        icon=viser.Icon.ROTATE_CLOCKWISE,
    )

    if visuals is not None:
        with server.gui.add_folder("Paper Figure", expand_by_default=True):
            overlays = server.gui.add_checkbox(
                "Overlays",
                initial_value=visuals.visible,
            )
            frames = server.gui.add_checkbox(
                "Site frames",
                initial_value=visuals.frames_visible,
            )

        @overlays.on_update
        def _(_event) -> None:
            visuals.set_visible(bool(overlays.value))

        @frames.on_update
        def _(_event) -> None:
            visuals.set_frames_visible(bool(frames.value))

    @reset_button.on_click
    def _(_event) -> None:
        sim.reset()
        if visuals is not None:
            visuals.update()
        readout.value = sim.status()

    return readout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    sim = DloInsertSim()
    server = viser.ViserServer(port=args.port)
    visuals = PaperVisuals(server, sim)
    readout = add_gui(server, sim, visuals)

    def step(_model: mj.MjModel, _data: mj.MjData) -> None:
        sim.step()
        if sim.step_count % 50 == 0:
            readout.value = sim.status()

    def reset(_model: mj.MjModel, _data: mj.MjData) -> None:
        sim.reset()
        visuals.update()
        readout.value = sim.status()

    def render(scene) -> None:
        scene.update_from_mjdata(sim.data)
        visuals.update(getattr(scene, "_scene_offset", np.zeros(3)))

    print(f"Viser DLO insertion demo: http://localhost:{args.port}")
    viewer = Viewer(
        sim.model,
        sim.data,
        step_fn=step,
        render_fn=render,
        reset_fn=reset,
        server=server,
    )
    viewer.scene.camera_tracking_enabled = False
    viewer.run()


if __name__ == "__main__":
    main()
