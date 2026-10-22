"""Pinned upstream models with small, explicit demonstration inputs."""

from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np

from .palette import apply_palette


@dataclass(frozen=True)
class Example:
    path: str
    description: str
    observe: str
    accents: tuple[str, ...] = ()
    lookat: tuple[float, float, float] | None = None
    distance: float | None = None
    azimuth: float | None = None
    elevation: float | None = None
    duration: float = 4
    min_version: tuple[int, int, int] = (3, 3, 7)
    keyframe: int | None = None
    track_body: str | None = None


EXAMPLES = {
    "newton_cradle": Example(
        "replicate/newton_cradle.xml",
        "Momentum transfer through seven suspended balls.",
        "Two displaced balls start the collision chain. The 0.1 ms timestep resolves the stiff impacts.",
        ("perturbed",),
        lookat=(0, 0, 0.13),
        distance=0.65,
    ),
    "balloons": Example(
        "balloons/balloons.xml",
        "Helium buoyancy, air drag, and limited-length strings.",
        "Four balloons pull a shared weight; buoyancy is approximated with gravity compensation.",
        ("orange", "pink"),
    ),
    "slider_crank": Example(
        "slider_crank/slider_crank.xml",
        "Forward, reverse, and broken slider-crank transmissions.",
        "The third crank starts outside its valid geometry. Use --manual to explore actuator sliders.",
    ),
    "muscle_arm": Example(
        "tendon_arm/arm26.xml",
        "A two-link arm pulled by six wrapping muscle tendons.",
        "Alternating muscle excitations bend the arm; orange tendons wrap around the joint cylinders.",
    ),
    "adhesion": Example(
        "adhesion/active_adhesion.xml",
        "Switchable adhesion on an arm and a pulley-mounted sphere.",
        "A timed winch input lowers and raises the adhesive sphere over particles. Use --manual to explore all five actuators.",
        ("4boxes", "sphere"),
    ),
    "hammock": Example(
        "hammock/hammock.xml",
        "A passive humanoid drops onto a hammock pinned at four corners.",
        "The orange flex grid sags and transfers the impact to its pinned corners.",
    ),
    "jelly": Example(
        "flex/jelly.xml",
        "A volumetric elastic block falls onto an inclined capsule.",
        "The orange 8 x 8 x 8 flex grid squashes on impact. Ctrl-drag the mocap capsule to deform it.",
    ),
    "car": Example(
        "extended/car/car.xml",
        "Differential-drive car.",
        "Two wheel motors combine forward and steering inputs.",
        ("chasis",),
        lookat=(0, 0, 0.05),
        distance=0.65,
        azimuth=125,
        elevation=-30,
        min_version=(3, 11, 0),
        track_body="car",
    ),
    "dominos": Example(
        "extended/sleep/dominos.xml",
        "A winding domino chain reaction.",
        "The tilted first tile wakes and topples the chain through contact.",
        lookat=(0.7, 0, 0.1),
        distance=4.2,
        azimuth=100,
        elevation=-60,
        duration=8,
        min_version=(3, 11, 0),
    ),
    "flag": Example(
        "extended/flex/flag.xml",
        "Wind-driven flexible flag.",
        "A corner-pinned sheet unfurls under aerodynamic forces.",
        lookat=(0.2, 0, 1.2),
        distance=2.6,
        azimuth=120,
        elevation=-20,
        min_version=(3, 11, 0),
    ),
    "trampoline": Example(
        "extended/flex/trampoline.xml",
        "A ball bouncing on an elastic trampoline.",
        "Four pinned corners support the stretched membrane.",
        lookat=(0, 0, 0.9),
        distance=3.5,
        azimuth=120,
        elevation=-25,
        min_version=(3, 11, 0),
    ),
    "soft_gripper": Example(
        "extended/flex/gripper_trilinear.xml",
        "Deformable fingertip pads grasp a block.",
        "The tendon closes the padded fingers before the hand lifts.",
        lookat=(0, 0, 0.35),
        distance=1.5,
        azimuth=120,
        elevation=-20,
        duration=5,
        min_version=(3, 11, 0),
    ),
    "pulley": Example(
        "extended/flex/pulley.xml",
        "A flexible loop driven around two pulleys.",
        "The rollers separate while the left roller drives the constrained cable.",
        lookat=(0, 0, 1),
        distance=4.2,
        azimuth=90,
        elevation=-70,
        min_version=(3, 11, 0),
        keyframe=0,
    ),
    "basket": Example(
        "extended/flex/basket.xml",
        "A flexible basket catches falling balls.",
        "Forty-five free spheres deform the thin-walled container.",
        lookat=(0, 0, 1.3),
        distance=4.5,
        azimuth=125,
        elevation=-25,
        duration=3,
        min_version=(3, 11, 0),
    ),
    "coil": Example(
        "extended/plugin/elasticity/coil.xml",
        "Free and actuated elastic coils.",
        "One coil drops while a slider stretches the other through a cable plugin.",
        ("actuated",),
        lookat=(0, 0, 0.25),
        distance=1.5,
        azimuth=105,
        elevation=-20,
        min_version=(3, 11, 0),
    ),
    "gears": Example(
        "extended/plugin/sdf/gear.xml",
        "Meshing gears with signed-distance contacts.",
        "A motor drives one gear; tooth contacts turn the other.",
        ("gear2",),
        lookat=(1.4, 0, 0.2),
        distance=10,
        azimuth=90,
        elevation=-65,
        min_version=(3, 11, 0),
    ),
    "nut_bolt": Example(
        "extended/plugin/sdf/nutbolt.xml",
        "Threaded nut and bolt contacts.",
        "A small initial spin lets the free nut move along the fixed threaded bolt.",
        ("nut",),
        lookat=(0, 0, 0.55),
        distance=3.4,
        azimuth=125,
        elevation=-20,
        min_version=(3, 11, 0),
    ),
    "treadmill": Example(
        "extended/surfacevel/treadmill.xml",
        "A treadmill transports a passive humanoid.",
        "Native moving-surface friction sweeps the collapsing rider off the deck.",
        ("deck", "screen"),
        lookat=(0.35, 0, 0.6),
        distance=4.8,
        azimuth=130,
        elevation=-25,
        min_version=(3, 11, 0),
    ),
    "twisty_cube": Example(
        "extended/cube/cube_3x3x3.xml",
        "A mechanically coupled 3 x 3 x 3 twisty cube.",
        "Face motors turn interacting cubelets with ball joints and contact constraints.",
        ("pX", "pZ"),
        lookat=(0, 0, 0),
        distance=0.20,
        azimuth=135,
        elevation=-25,
        min_version=(3, 11, 0),
    ),
}


def configure_camera(name: str, model: mj.MjModel, camera: mj.MjvCamera) -> None:
    example = EXAMPLES[name]
    camera.lookat[:] = model.stat.center if example.lookat is None else example.lookat
    camera.distance = (
        2.5 * model.stat.extent if example.distance is None else example.distance
    )
    camera.azimuth = (
        model.vis.global_.azimuth if example.azimuth is None else example.azimuth
    )
    camera.elevation = (
        model.vis.global_.elevation if example.elevation is None else example.elevation
    )


def load_example(
    name: str, *, original_colors: bool = False
) -> tuple[mj.MjModel, mj.MjData]:
    example = EXAMPLES[name]
    if tuple(map(int, mj.__version__.split(".")[:3])) < example.min_version:
        version = ".".join(map(str, example.min_version))
        raise RuntimeError(
            f"{name} requires MuJoCo >= {version}; found {mj.__version__}"
        )
    path = Path(__file__).resolve().parent / "models" / example.path
    spec = mj.MjSpec.from_file(str(path))
    if name == "coil" and not original_colors:
        # The cable plugin otherwise replaces geom colors with its strain heatmap
        # during scene updates. vmax affects visualization only.
        for plugin in spec.plugins:
            plugin.config = {**plugin.config, "vmax": "0"}
    # Newer MuJoCo releases require the discrete integrator for implicit flex
    # elasticity; 3.3.7 uses the integrator authored in the upstream model.
    if name in ("jelly", "soft_gripper") and hasattr(
        mj.mjtIntegrator, "mjINT_DISCRETE"
    ):
        spec.option.integrator = mj.mjtIntegrator.mjINT_DISCRETE
    model = spec.compile()
    if not original_colors:
        apply_palette(model, example.accents)
        if name == "dominos":
            from .palette import FLAME

            dynamic_geoms = np.flatnonzero(model.geom_bodyid > 0)
            model.geom_rgba[dynamic_geoms[::4], :3] = FLAME[:3]
    data = mj.MjData(model)
    if example.keyframe is not None:
        mj.mj_resetDataKeyframe(model, data, example.keyframe)
    if name == "nut_bolt":
        data.qvel[5] = 3
    mj.mj_forward(model, data)
    return model, data


def track_camera(
    name: str, model: mj.MjModel, data: mj.MjData, camera: mj.MjvCamera
) -> None:
    body = EXAMPLES[name].track_body
    if body is not None:
        camera.lookat[:] = data.xpos[model.body(body).id]


def drive(name: str, model: mj.MjModel, data: mj.MjData) -> None:
    """Update demonstration inputs; manual mode leaves these to viewer sliders."""
    t = data.time
    if name == "slider_crank":
        data.ctrl[:] = 0.07 * np.sin(1.5 * t + np.arange(model.nu))
    elif name == "muscle_arm":
        phase = np.sin(
            1.5 * t + np.array([0, np.pi, 0.8, 0.8 + np.pi, 1.6, 1.6 + np.pi])
        )
        data.ctrl[:] = 0.25 + 0.2 * phase
    elif name == "adhesion":
        phase = t % 8
        data.ctrl[model.actuator("winch").id] = -0.6 if phase < 3 else 0.4
        data.ctrl[model.actuator("adhere_winch").id] = float(1 < phase < 6)
        data.ctrl[model.actuator("adhere_arm").id] = 1
    elif name == "car":
        data.ctrl[:] = [0.3, 0.12 * np.sin(0.8 * t)]
    elif name == "soft_gripper":
        data.ctrl[model.actuator("grasp").id] = 0.17 * np.clip(t - 0.5, 0, 1)
        data.ctrl[model.actuator("lift").id] = 0.3 * np.clip((t - 2) / 1.5, 0, 1)
    elif name == "coil":
        data.ctrl[:] = 0.07 * (1 - np.cos(2 * t))
    elif name == "gears":
        data.ctrl[:] = 0.25
    elif name == "twisty_cube":
        data.ctrl[:] = 0
        phase = t % 4
        if phase < 1.5:
            data.ctrl[model.actuator("red").id] = 0.015
        elif 2 < phase < 3.5:
            data.ctrl[model.actuator("white").id] = 0.015
