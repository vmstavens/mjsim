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
}


def configure_camera(name: str, model: mj.MjModel, camera: mj.MjvCamera) -> None:
    example = EXAMPLES[name]
    camera.lookat[:] = model.stat.center if example.lookat is None else example.lookat
    camera.distance = (
        2.5 * model.stat.extent if example.distance is None else example.distance
    )
    camera.azimuth = model.vis.global_.azimuth
    camera.elevation = model.vis.global_.elevation


def load_example(
    name: str, *, original_colors: bool = False
) -> tuple[mj.MjModel, mj.MjData]:
    example = EXAMPLES[name]
    path = Path(__file__).resolve().parent / "models" / example.path
    spec = mj.MjSpec.from_file(str(path))
    # Newer MuJoCo releases require the discrete integrator for implicit flex
    # elasticity; 3.3.7 uses the integrator authored in the upstream model.
    if name == "jelly" and hasattr(mj.mjtIntegrator, "mjINT_DISCRETE"):
        spec.option.integrator = mj.mjtIntegrator.mjINT_DISCRETE
    model = spec.compile()
    if not original_colors:
        apply_palette(model, example.accents)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


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
