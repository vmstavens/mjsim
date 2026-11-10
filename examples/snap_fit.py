"""Friction-held V-clip and a lid hinged on the left; built entirely with MjSpec.

Double-click the orange mocap handle, then Ctrl + right-drag to move the finger.
A/D: left/right; W/S: up/down; Q/E: depth (0.5 mm steps). R clears force history.
P plays/replays the opening plan; X stops it. Manual nudges also stop the plan.
"""

import time
from collections import deque
from itertools import pairwise
from queue import SimpleQueue

import mujoco as mj
import mujoco.viewer
import numpy as np
from physics.palette import (
    CARBON,
    CHARCOAL,
    COOL_GRAY,
    FLAME,
    GRAPHITE,
    LIGHT_ORANGE,
    OFF_WHITE,
    RAISED,
    apply_palette,
)

# Bending rigidity (N m²); hinge stiffness is rigidity / segment length.
SUPPORT_RIGIDITY = 0.005
CLIP_RIGIDITY = 0.001


def build_spec() -> mj.MjSpec:
    spec = mj.MjSpec()
    spec.modelname = "Friction V-clip"
    spec.compiler.degree = False
    spec.option.timestep = 0.0005
    spec.option.noslip_iterations = 10
    spec.option.impratio = 10
    spec.option.gravity = [0, 0, -9.81]
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    spec.stat.center = [0, 0, 0.1]
    spec.stat.extent = 0.35
    spec.visual.map.stiffness = 1
    spec.default.geom.friction = [1.5, 0.005, 0.0001]
    spec.default.geom.solref = [0.02, 1]
    spec.default.joint.damping = [0.001, 0, 0]
    spec.default.joint.armature = 0.00001
    spec.add_texture(
        name="sky",
        type=mj.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mj.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=RAISED[:3],
        width=32,
        height=32,
    )
    world = spec.worldbody
    world.add_light(pos=[0, -0.3, 0.8], dir=[0, 0, -1])
    world.add_geom(
        name="ground",
        type=mj.mjtGeom.mjGEOM_PLANE,
        size=[0.5, 0.5, 0.01],
        rgba=CHARCOAL,
    )

    def box(parent, name, pos, size, color=COOL_GRAY, mass=0.01, moving=False):
        return parent.add_geom(
            name=name,
            type=mj.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=color,
            mass=mass,
            contype=2 if moving else 1,
            conaffinity=1 if moving else 2,
        )

    # Box cross-section with an open pocket for the V spring. No undercuts.
    box(world, "box_bottom", [0, 0, 0.025], [0.115, 0.015, 0.005])
    box(world, "box_left", [-0.110, 0, 0.0875], [0.005, 0.015, 0.0575])
    box(world, "box_rim", [-0.0275, 0, 0.146], [0.0825, 0.015, 0.004])
    box(world, "pocket_left", [0.0525, 0, 0.120], [0.0025, 0.015, 0.030])
    box(world, "pocket_bottom", [0.0725, 0, 0.087], [0.0175, 0.015, 0.003])
    box(world, "box_right", [0.101, 0, 0.0875], [0.014, 0.015, 0.0625])

    lid = world.add_body(name="lid", pos=[-0.105, 0, 0.156])
    lid.add_joint(
        name="lid_hinge",
        type=mj.mjtJoint.mjJNT_HINGE,
        axis=[0, -1, 0],
        limited=True,
        range=[0, 1.4],
        damping=[0.01, 0, 0],
    )
    box(
        lid, "lid_plate", [0.0815, 0, 0], [0.0815, 0.015, 0.006], OFF_WHITE, 0.025, True
    )
    lid.add_geom(
        name="lid_joint_marker",
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[0.004, 0, 0],
        rgba=COOL_GRAY,
        pos=[0, -0.017, 0],
        contype=0,
        conaffinity=0,
        mass=0,
    )

    # Repeated boxes must be NESTED bodies, not independent replicated siblings:
    # every new hinge connects the next strip segment to the preceding one.
    # Tight rounded V: a vertical attached leg and an outward-angled free leg.
    radius = 0.003
    center = np.array([0.163 + radius, -0.055])
    tip = np.array([0.198, 0.0])
    offset = tip - center
    tilt = np.arctan2(offset[0], offset[1]) - np.arcsin(radius / np.linalg.norm(offset))
    angles = np.linspace(np.pi, 2 * np.pi - tilt, 25)
    arc = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    left = np.linspace([0.163, 0], arc[0], 11)
    right = np.linspace(arc[-1], tip, 11)
    points = np.concatenate([left, arc[1:], right[1:]])
    parent, previous = lid, np.zeros(2)
    for i, (start, end) in enumerate(pairwise(points)):
        delta, offset = end - start, start - previous
        length = np.linalg.norm(delta)
        angle = np.arctan2(delta[0], delta[1])
        segment = parent.add_body(name=f"clip_{i}", pos=[offset[0], 0, offset[1]])
        # The attached leg is stiffer; bending is concentrated in the return
        # and free leg, so squeezing does not fold the support into the wall.
        segment.add_joint(
            name=f"bend_{i}",
            type=mj.mjtJoint.mjJNT_HINGE,
            axis=[0, 1, 0],
            stiffness=[(SUPPORT_RIGIDITY if i < 10 else CLIP_RIGIDITY) / length, 0, 0],
            limited=True,
            range=[-0.25, 0.25],
        )
        segment.add_geom(
            name=f"strip_{i}",
            type=mj.mjtGeom.mjGEOM_BOX,
            pos=[delta[0] / 2, 0, delta[1] / 2],
            quat=[np.cos(angle / 2), 0, np.sin(angle / 2), 0],
            size=[0.0012, 0.010, length / 2],
            rgba=FLAME,
            mass=0.0001,
            contype=2,
            conaffinity=1,
        )
        segment.add_geom(
            name=f"clip_joint_marker_{i}",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            pos=[0, -0.011, 0],
            size=[min(0.0008, length * 0.22), 0, 0],
            rgba=COOL_GRAY,
            contype=0,
            conaffinity=0,
            mass=0,
        )
        parent, previous = segment, start
    end = points[-1] - previous
    tab = parent.add_body(name="press_tab", pos=[end[0], 0, end[1]])
    box(tab, "tab", [0.006, 0, 0.002], [0.009, 0.012, 0.004], FLAME, 0.002, True)
    # A welded dynamic mount follows the orange mocap handle; the rigid child
    # finger carries a load cell and can contact both the clip and fixed box.
    # A force sensor belongs on this parent-child interface, not on world.
    handle = world.add_body(name="finger_mocap", mocap=True, pos=[0.142, 0, 0.160])
    handle.add_geom(
        name="finger_handle",
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[0.006, 0, 0],
        rgba=LIGHT_ORANGE,
        contype=0,
        conaffinity=0,
    )
    handle.add_site(name="finger_target", size=[0.001, 0.001, 0.001], rgba=[0, 0, 0, 0])
    mount = world.add_body(name="finger_mount", pos=[0.142, 0, 0.160])
    mount.add_freejoint(name="finger_mount_free")
    mount.add_geom(
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[0.001, 0, 0],
        mass=0.005,
        rgba=[0, 0, 0, 0],
        contype=0,
        conaffinity=0,
    )
    mount.add_site(
        name="finger_mount_site", size=[0.001, 0.001, 0.001], rgba=[0, 0, 0, 0]
    )
    spec.add_equality(
        name="finger_follow",
        type=mj.mjtEq.mjEQ_WELD,
        objtype=mj.mjtObj.mjOBJ_SITE,
        name1="finger_target",
        name2="finger_mount_site",
        solref=[0.0015, 1],
        solimp=[0.99, 0.999, 0.0001, 0.5, 2],
    )
    finger = mount.add_body(name="finger")
    finger.add_geom(
        name="fingertip",
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        fromto=[-0.022, 0, 0, 0, 0, 0],
        size=[0.004, 0, 0],
        rgba=OFF_WHITE,
        mass=0.001,
        contype=4,
        conaffinity=3,
    )
    finger.add_site(
        name="finger_loadcell", size=[0.001, 0.001, 0.001], rgba=[0, 0, 0, 0]
    )
    spec.add_sensor(
        name="finger_force",
        type=mj.mjtSensor.mjSENS_FORCE,
        objtype=mj.mjtObj.mjOBJ_SITE,
        objname="finger_loadcell",
    )
    return spec


def apply_theme(model):
    apply_palette(
        model,
        ("clip", "press_tab"),
        geom_colors={
            "joint_marker": COOL_GRAY,
            "finger_handle": LIGHT_ORANGE,
            "fingertip": OFF_WHITE,
        },
    )


def finger_force(model, data):
    """World-frame support force, including dynamic loads, minus static weight."""
    site = model.site("finger_loadcell").id
    adr = model.sensor("finger_force").adr[0]
    rotation = data.site_xmat[site].reshape(3, 3)
    weight = model.body("finger").mass[0] * model.opt.gravity
    return rotation @ data.sensordata[adr : adr + 3] + weight


class OpeningPlan:
    """Press using the measured finger, then assist lifting with a bounded torque."""

    def __init__(self, model):
        self.model = model
        self.mocap = model.body("finger_mocap").mocapid[0]
        hinge = model.joint("lid_hinge")
        self.qpos = hinge.qposadr[0]
        self.dof = hinge.dofadr[0]
        self.pivot = model.body("lid").pos.copy()
        self.press_position = np.array([0.128, 0, 0.156])
        self.torque_limit = 0.045  # N m; separate from the plotted finger force.
        self.running = False
        self.phase = "Ready"

    def start(self, data):
        # Replay always starts from the same closed, unperturbed model.
        mj.mj_resetData(self.model, data)
        mj.mj_forward(self.model, data)
        self.start_position = data.mocap_pos[self.mocap].copy()
        self.running = True
        self.phase = "Settling"

    def stop(self, data):
        self.running = False
        self.phase = "Manual"
        data.qfrc_applied[self.dof] = 0

    @staticmethod
    def smooth(value):
        value = np.clip(value, 0, 1)
        return value * value * (3 - 2 * value)

    def update(self, data):
        if not self.running:
            return
        t = data.time
        angle = 0.0
        data.qfrc_applied[self.dof] = 0
        if t < 1:
            self.phase = "Settling"
            position = self.start_position
        elif t < 4:
            self.phase = "Pressing" if t < 3 else "Holding press"
            blend = self.smooth((t - 1) / 2)
            position = self.start_position + blend * (
                self.press_position - self.start_position
            )
        else:
            self.phase = "Opening" if t < 9 else "Open"
            target = 1.05 * self.smooth((t - 4) / 5)
            angle = float(data.qpos[self.qpos])
            torque = 0.4 * (target - angle) - 0.02 * data.qvel[self.dof]
            data.qfrc_applied[self.dof] = np.clip(torque, 0, self.torque_limit)
            # Follow the actual lid angle while retaining the inward squeeze.
            dx, dy, dz = self.press_position - self.pivot
            c, s = np.cos(angle), np.sin(angle)
            position = self.pivot + [c * dx - s * dz, dy, s * dx + c * dz]
        data.mocap_pos[self.mocap] = position
        data.mocap_quat[self.mocap] = [np.cos(angle / 2), 0, -np.sin(angle / 2), 0]


class ForcePlot:
    """Recent load-cell force; peak includes every physics step."""

    def __init__(self, window=5):
        self.window = window
        self.figure = mj.MjvFigure()
        mj.mjv_defaultFigure(self.figure)
        self.figure.xlabel = "Time (s)"
        self.figure.flg_legend = True
        self.figure.flg_extend = False
        self.figure.linewidth = 3
        self.figure.yformat = "%.2f"
        self.figure.linename[0] = "|F| (N)"
        self.figure.linename[1] = "Left press (N)"
        self.figure.linergb[0] = FLAME[:3]
        self.figure.linergb[1] = OFF_WHITE[:3]
        self.figure.figurergba = CARBON
        self.figure.panergba = CHARCOAL
        self.figure.legendrgba = RAISED
        self.figure.textrgb = OFF_WHITE[:3]
        self.figure.gridrgb = GRAPHITE[:3]
        self.history = deque(maxlen=1000)
        self.clear()

    def clear(self):
        self.history.clear()
        self.force = np.zeros(3)
        self.peak = 0.0
        self.figure.linepnt[:] = 0

    def observe(self, force):
        self.force = force
        self.peak = max(self.peak, float(np.linalg.norm(force)))

    def update(self, now, phase="Finger"):
        magnitude = float(np.linalg.norm(self.force))
        self.history.append((now, magnitude, max(0.0, -self.force[0])))
        while self.history and self.history[0][0] < now - self.window:
            self.history.popleft()
        rows = np.asarray(self.history)
        for line in range(2):
            self.figure.linepnt[line] = len(rows)
            self.figure.linedata[line, : 2 * len(rows)] = np.column_stack(
                [rows[:, 0] - now, rows[:, line + 1]]
            ).ravel()
        self.figure.range[0] = [-self.window, 0]
        self.figure.range[1] = [0, max(0.1, float(rows[:, 1:].max()) * 1.15)]
        self.figure.title = f"{phase}: {magnitude:.3f} N | peak {self.peak:.3f} N"


def main():
    model = build_spec().compile()
    apply_theme(model)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    mocap = model.body("finger_mocap").mocapid[0]
    plot = ForcePlot()
    plan = OpeningPlan(model)
    keys = SimpleQueue()
    moves = {
        ord("A"): [-1, 0, 0],
        ord("D"): [1, 0, 0],
        ord("W"): [0, 0, 1],
        ord("S"): [0, 0, -1],
        ord("Q"): [0, -1, 0],
        ord("E"): [0, 1, 0],
    }
    print("P: play/replay opening plan | X: stop | R: clear force peak.")
    print("Move orange finger: A/D left/right, W/S up/down, Q/E depth (0.5 mm).")
    print("Or double-click orange handle, then Ctrl + right-drag. R clears force peak.")
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
        viewer.cam.lookat[:] = [0.015, 0, 0.1]
        viewer.cam.distance = 0.50
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -12
        while viewer.is_running():
            start = time.monotonic()
            with viewer.lock():
                while not keys.empty():
                    key = keys.get()
                    if key == ord("P"):
                        plan.start(data)
                        plot.clear()
                    elif key == ord("X"):
                        plan.stop(data)
                    elif key in moves:
                        plan.stop(data)
                        data.mocap_pos[mocap] += 0.0005 * np.asarray(moves[key])
                    elif key == ord("R"):
                        plot.clear()
                for _ in range(32):
                    plan.update(data)
                    mj.mj_step(model, data)
                    plot.observe(finger_force(model, data))
            plot.update(data.time, plan.phase)
            viewport = viewer.viewport
            width, height = (
                min(650, viewport.width * 3 // 5),
                min(400, viewport.height // 2),
            )
            # Overlay methods and sync take their own locks: keep them outside
            # viewer.lock() so the first frame cannot deadlock.
            viewer.set_figures(
                (
                    mj.MjrRect(
                        viewport.left + viewport.width - width - 10,
                        viewport.bottom + 10,
                        width,
                        height,
                    ),
                    plot.figure,
                )
            )
            viewer.sync()
            time.sleep(max(0, 0.016 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
