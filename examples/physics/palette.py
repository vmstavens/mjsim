"""Carbon and flame colors, applied before creating a renderer or viewer.

Only rendering fields are changed. Collision geometry, masses, friction,
constraints, and solver parameters retain the upstream values.
"""

from collections.abc import Mapping

import mujoco as mj
import numpy as np


def rgba(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0])


CARBON = rgba("111214")
CHARCOAL = rgba("191b1e")
RAISED = rgba("22252a")
OFF_WHITE = rgba("f2f2ee")
COOL_GRAY = rgba("adb0b5")
LIGHT_ORANGE = rgba("ff8045")
FLAME = rgba("ff6b2b")
GRAPHITE = rgba("36393e")


def rgb8(color: np.ndarray) -> tuple[int, int, int]:
    """Convert a palette color for browser lights, grids, and frame overlays."""
    return tuple(int(round(value * 255)) for value in color[:3])


def _role_color(
    name: str, default: np.ndarray, overrides: Mapping[str, np.ndarray]
) -> np.ndarray:
    for fragment, color in reversed(list(overrides.items())):
        if fragment in name:
            return color
    return default


def apply_palette(
    model: mj.MjModel,
    accents: tuple[str, ...] = (),
    *,
    geom_colors: Mapping[str, np.ndarray] | None = None,
    site_colors: Mapping[str, np.ndarray] | None = None,
    flex_colors: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Use neutral structures and orange named bodies/geoms and flex objects.

    Name fragments also match replicated bodies. Existing transparency is kept.
    Materials are detached so an upstream texture cannot tint the chosen colors.
    Call once, before opening the viewer (textures are uploaded on creation).
    """
    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]
        names = (model.geom(i).name or "") + " " + (model.body(body_id).name or "")
        color = OFF_WHITE if model.body_weldid[body_id] else COOL_GRAY
        if model.geom_type[i] == mj.mjtGeom.mjGEOM_PLANE:
            color = CHARCOAL
        elif any(fragment in names for fragment in accents):
            color = FLAME
        color = _role_color(names, color, geom_colors or {})
        alpha = model.geom_rgba[i, 3]
        material = model.geom_matid[i]
        if material >= 0:
            alpha = min(alpha, model.mat_rgba[material, 3])
        model.geom_rgba[i] = color
        model.geom_rgba[i, 3] = alpha
    model.geom_matid[:] = -1

    for colors, materials in (
        (model.flex_rgba, model.flex_matid),
        (model.skin_rgba, model.skin_matid),
    ):
        for i, material in enumerate(materials):
            if material >= 0:
                colors[i, 3] = min(colors[i, 3], model.mat_rgba[material, 3])

    for colors, color in (
        (model.flex_rgba, FLAME),
        (model.skin_rgba, LIGHT_ORANGE),
        (model.tendon_rgba, LIGHT_ORANGE),
        (model.site_rgba, COOL_GRAY),
    ):
        colors[:, :3] = color[:3]
    model.flex_matid[:] = -1
    model.skin_matid[:] = -1
    roles = {
        "start": OFF_WHITE,
        "actual": OFF_WHITE,
        "loadcell": OFF_WHITE,
        "mocap": OFF_WHITE,
        "goal": FLAME,
        "target": FLAME,
        "tip": LIGHT_ORANGE,
        "entry": LIGHT_ORANGE,
        "exit": OFF_WHITE,
        "ball_site": FLAME,
        "box_site": FLAME,
        **(site_colors or {}),
    }
    for i in range(model.nsite):
        model.site_rgba[i, :3] = _role_color(
            model.site(i).name or "", COOL_GRAY, roles
        )[:3]
    for i in range(model.nflex):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_FLEX, i) or ""
        model.flex_rgba[i, :3] = _role_color(name, FLAME, flex_colors or {})[:3]
    # Preserve light intensity while removing an asset's colored wash.
    model.light_diffuse[:] = model.light_diffuse.mean(axis=1, keepdims=True)
    model.vis.rgba.haze[:] = CARBON
    model.vis.rgba.slidercrank[:] = LIGHT_ORANGE
    model.vis.rgba.crankbroken[:] = OFF_WHITE
    for fields, color in (
        (
            (
                "force",
                "contactforce",
                "contactpoint",
                "actuatorpositive",
                "selectpoint",
            ),
            FLAME,
        ),
        (
            (
                "joint",
                "connect",
                "constraint",
                "inertia",
                "com",
                "camera",
                "light",
                "bv",
                "frustum",
            ),
            COOL_GRAY,
        ),
        (
            (
                "contactfriction",
                "contactgap",
                "contacttorque",
                "actuator",
                "actuatornegative",
                "bvactive",
                "rangefinder",
            ),
            OFF_WHITE,
        ),
    ):
        for field in fields:
            value = getattr(model.vis.rgba, field, None)
            if value is not None and value.size >= 3:
                value[:3] = color[:3]

    # MuJoCo renamed tex_rgb to tex_data when adding configurable channels.
    pixels = getattr(model, "tex_data", None)
    if pixels is None:
        pixels = model.tex_rgb
    for i in range(model.ntex):
        if model.tex_type[i] != mj.mjtTexture.mjTEXTURE_SKYBOX:
            continue
        channels = model.tex_nchannel[i] if hasattr(model, "tex_nchannel") else 3
        start = model.tex_adr[i]
        size = model.tex_width[i] * model.tex_height[i] * channels
        sky = pixels[start : start + size].reshape(-1, channels)
        brightness = sky[:, :3].mean(axis=1, keepdims=True) / 255
        sky[:, :3] = (CARBON[:3] + brightness * (RAISED[:3] - CARBON[:3])) * 255


def style_plot(figure, axes) -> None:
    """Apply the same surfaces and text contrast to Matplotlib diagnostics."""
    figure.set_facecolor(CARBON[:3])
    for axis in np.asarray(axes, dtype=object).reshape(-1):
        axis.set_facecolor(CHARCOAL[:3])
        axis.tick_params(colors=COOL_GRAY[:3])
        axis.xaxis.label.set_color(OFF_WHITE[:3])
        axis.yaxis.label.set_color(OFF_WHITE[:3])
        axis.title.set_color(OFF_WHITE[:3])
        for spine in axis.spines.values():
            spine.set_color(GRAPHITE[:3])
        axis.grid(True, color=GRAPHITE[:3], alpha=0.7)
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(RAISED[:3])
            legend.get_frame().set_edgecolor(GRAPHITE[:3])
            for label in legend.get_texts():
                label.set_color(OFF_WHITE[:3])
