"""Manual bronchoscope control at an opened airway inlet, with tip video.

Run from any directory. A detailed visual airway and a simplified rigid-flex
collision surface share a frame. Rod stiffness and contact tuning are illustrative.
"""

from __future__ import annotations

import argparse
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from queue import Empty, SimpleQueue

import mujoco as mj
import numpy as np
from physics.palette import apply_palette

ASSETS = Path(__file__).resolve().parent / "assets"
TENDONS = ("xp", "xm", "yp", "ym")
BASE_POSITION = np.array([0.21, 0.0, 0.49])
SHAFT_LENGTH = 0.28
TIP_LENGTH = 0.06
MAX_TENSION = 0.6  # newtons, positive commands only pull
INSERTION_RANGE = (-0.25, 0.10)  # metres: negative advances, positive retracts
INSERTION_DEPTH = 0.06
INSERTION_SECONDS = 24.0
INSERTION_WAIT = 2.0
ENTRY_GAP = 0.012
CAMERA_NAME = "tip_camera"
CAMERA_WIDTH, CAMERA_HEIGHT = 400, 300


def default_airway() -> Path:
    """Prefer the airways directory, retaining support for the current asset."""
    original = ASSETS / "airways" / "hollow_airway.obj"
    if original.is_file():
        return original
    candidates = sorted(
        path
        for path in (ASSETS / "airways").glob("*.obj")
        if "collision" not in path.stem
    )
    if candidates:
        return candidates[0]
    return ASSETS / "hollow_airway.obj"


def default_collision(airway: Path) -> Path | None:
    path = airway.with_name(f"{airway.stem}_collision.obj")
    return path if path.is_file() else None


def add_airway(
    root: ET.Element, path: Path, scale: float, *, open_inlet: bool = False
) -> dict[str, bytes]:
    """Translate the supplied OBJ's single MTL material into MuJoCo assets.

    Kd, d/Tr, Ks, Ns and a simple map_Kd are supported. MuJoCo's shininess
    uses a different scale from Wavefront, so Ns is mapped approximately.
    UVs are retained. Optional inlet trimming uses a virtual copy of the OBJ;
    no asset files are rewritten. Inlet fitting assumes this upright airway.
    """
    vertices = []
    libraries = []
    materials = set()
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(v) * scale for v in fields[1:4]])
            elif fields[0] == "mtllib":
                libraries.append(path.parent / " ".join(fields[1:]))
            elif fields[0] == "usemtl":
                materials.add(" ".join(fields[1:]))
    if not vertices:
        raise ValueError(f"No vertices in {path}")
    if len(materials) > 1:
        raise ValueError("This example supports a single OBJ material")
    points = np.asarray(vertices)
    lower, upper = points.min(axis=0), points.max(axis=0)
    # Preserve source orientation, center horizontally, put the lowest point
    # above the floor. Millimetres are assumed by the default scale of 0.001.
    offset = -(lower + upper) / 2
    offset[2] = 0.015 - lower[2]

    # The supplied hollow mesh has a sealed top. Trim the top 5 mm from a VFS
    # copy, exposing the inner and outer tracheal walls without editing the OBJ.
    # Fit a downward entry axis from the upper 40 mm of the trachea.
    cut_z = upper[2] - 0.005
    if open_inlet:
        band = points[np.abs(points[:, 2] - cut_z) < 0.00075]
        if len(band) < 3:
            raise ValueError("Cannot locate the upper tracheal inlet")
        entry = (band.min(axis=0) + band.max(axis=0)) / 2
        entry[2] = cut_z
        lower_band = points[
            (np.abs(points[:, 2] - (cut_z - 0.04)) < 0.00075)
            & (np.linalg.norm(points[:, :2] - entry[:2], axis=1) < 0.02)
        ]
        if len(lower_band) < 3:
            raise ValueError("Cannot estimate the upper tracheal direction")
        lower_center = (lower_band.min(axis=0) + lower_band.max(axis=0)) / 2
        up = entry - lower_center
        up /= np.linalg.norm(up)
        # Put the fitted frame on a site for model construction and inspection.
        x = np.cross([0.0, 1.0, 0.0], up)
        x /= np.linalg.norm(x)
        rotation = np.column_stack((x, np.cross(up, x), up))
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, rotation.ravel())
        ET.SubElement(
            root.find("worldbody"),
            "site",
            name="airway_entry",
            pos=" ".join(map(str, entry + offset)),
            quat=" ".join(map(str, quat)),
        )

    properties = {}
    selected = next(iter(materials), None)
    texture_directory = path.parent
    for library in libraries:
        if not library.is_file():
            warnings.warn(f"Missing MTL: {library}; using neutral gray", stacklevel=2)
            continue
        active = False
        for line in library.read_text().splitlines():
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if fields[0] == "newmtl":
                active = selected is None or " ".join(fields[1:]) == selected
            elif active:
                properties[fields[0]] = fields[1:]
                texture_directory = library.parent

    diffuse = properties.get("Kd", ["0.4"] * 3)
    alpha = float(properties.get("d", [1.0])[0])
    if "Tr" in properties and "d" not in properties:
        alpha = 1.0 - float(properties["Tr"][0])
    asset = root.find("asset")
    material = ET.SubElement(
        asset,
        "material",
        name="airway_material",
        rgba=" ".join([*diffuse, str(alpha)]),
        specular=str(np.mean([float(v) for v in properties.get("Ks", [0.0])])),
        shininess=str(np.clip(float(properties.get("Ns", [0.0])[0]) / 1000, 0, 1)),
    )
    virtual_assets = {}
    if "map_Kd" in properties:
        texture = texture_directory / " ".join(properties["map_Kd"])
        if texture.is_file():
            # MuJoCo accepts PNG textures, while this asset ships a JPEG.
            # OpenCV is already a project dependency; keep conversion in memory.
            import cv2

            pixels = cv2.imdecode(
                np.frombuffer(texture.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if pixels is None:
                raise ValueError(f"Cannot decode airway texture: {texture}")
            ok, encoded = cv2.imencode(".png", pixels)
            if not ok:
                raise ValueError(f"Cannot convert airway texture: {texture}")
            virtual_assets["airway_texture.png"] = encoded.tobytes()
            ET.SubElement(
                asset,
                "texture",
                name="airway_texture",
                type="2d",
                file="airway_texture.png",
            )
            material.set("texture", "airway_texture")
        else:
            warnings.warn(
                f"Missing MTL texture: {texture}; using Kd color", stacklevel=2
            )
    mesh_file = str(path)
    if open_inlet:
        keep_vertex = points[:, 2] <= cut_z
        kept_lines = []
        with path.open() as stream:
            for line in stream:
                if line.startswith("f "):
                    indices = [int(field.split("/")[0]) for field in line.split()[1:]]
                    indices = [i - 1 if i > 0 else len(points) + i for i in indices]
                    if not all(keep_vertex[i] for i in indices):
                        continue
                kept_lines.append(line)
        mesh_file = "airway_open.obj"
        virtual_assets[mesh_file] = "".join(kept_lines).encode()
    ET.SubElement(
        asset,
        "mesh",
        name="airway_mesh",
        file=mesh_file,
        scale=f"{scale} {scale} {scale}",
        inertia="shell",
        maxhullvert="64",
    )
    ET.SubElement(
        root.find("worldbody"),
        "geom",
        name="airway",
        type="mesh",
        mesh="airway_mesh",
        material="airway_material",
        pos=" ".join(map(str, offset)),
        contype="0",
        conaffinity="0",
        density="0",
    )
    return virtual_assets


def build_model(
    airway: Path | None,
    scale: float = 0.001,
    *,
    collision: Path | None = None,
    collisions: bool = True,
) -> mj.MjModel:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Airway scale must be finite and positive")
    root = ET.fromstring("""
    <mujoco model="airway_bronchoscope">
      <compiler angle="radian"/>
      <option timestep="0.001" integrator="implicitfast"/>
      <statistic center="0 0 0.32" extent="0.8"/>
      <visual><global offwidth="1200" offheight="900"/>
        <map znear="0.000125" zfar="4"/>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/>
      </visual>
      <default>
        <geom friction="0.5 0.01 0.001" solref="0.008 1"/>
        <site size="0.0005" rgba="1 0.5 0.2 0" group="3"/>
      </default>
      <asset/>
      <worldbody>
        <light pos="0 -0.3 1" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="0.6 0.5 0.01"/>
      </worldbody>
      <tendon/>
      <actuator/>
    </mujoco>
    """)
    virtual_assets = {}
    if airway is not None:
        virtual_assets = add_airway(
            root, Path(airway).expanduser().resolve(), scale, open_inlet=True
        )
    world = root.find("worldbody")
    if airway is not None and collisions:
        collision = collision or default_collision(Path(airway))
        if collision is not None:
            visual_geom = world.find("geom[@name='airway']")
            surface = ET.SubElement(
                world,
                "flexcomp",
                name="airway_wall",
                type="mesh",
                dim="2",
                rigid="true",
                file=str(Path(collision).expanduser().resolve()),
                scale=f"{scale} {scale} {scale}",
                pos=visual_geom.get("pos"),
                radius="0.0001",
                rgba="0.1 0.8 0.3 0",
                group="5",
            )
            ET.SubElement(
                surface,
                "contact",
                selfcollide="none",
                contype="2",
                conaffinity="1",
                condim="3",
                friction="0.3 0.005 0.0001",
                solref="0.004 1",
                solimp="0.95 0.99 0.001",
            )
    entry = world.find("site[@name='airway_entry']")
    base_position = BASE_POSITION
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    if entry is not None:
        base_quat = np.fromstring(entry.get("quat"), sep=" ")
        rotation = np.empty(9)
        mj.mju_quat2Mat(rotation, base_quat)
        up = rotation.reshape(3, 3)[:, 2]
        base_position = np.fromstring(entry.get("pos"), sep=" ") + up * (
            SHAFT_LENGTH + TIP_LENGTH + ENTRY_GAP
        )
    base = ET.SubElement(
        world,
        "body",
        name="base",
        pos=" ".join(map(str, base_position)),
        quat=" ".join(map(str, base_quat)),
        gravcomp="1",
    )
    ET.SubElement(
        base,
        "joint",
        name="insertion",
        type="slide",
        axis="0 0 1",
        range=f"{INSERTION_RANGE[0]} {INSERTION_RANGE[1]}",
        damping="2",
    )
    ET.SubElement(
        base,
        "joint",
        name="rotation",
        type="hinge",
        axis="0 0 1",
        range="-3.1416 3.1416",
        damping="0.01",
    )
    ET.SubElement(
        base,
        "geom",
        name="handle",
        type="cylinder",
        size="0.014 0.025",
        pos="0 0 0.025",
        mass="0.12",
    )
    ET.SubElement(
        base,
        "geom",
        name="rotation_marker",
        type="box",
        size="0.008 0.002 0.002",
        pos="0.014 0 0.025",
        mass="0.001",
    )
    ET.SubElement(base, "site", name="base_site", pos="0 0 0")
    actuators = root.find("actuator")
    ET.SubElement(
        actuators,
        "position",
        name="base_height",
        joint="insertion",
        kp="1500",
        kv="30",
        ctrlrange=f"{INSERTION_RANGE[0]} {INSERTION_RANGE[1]}",
    )
    ET.SubElement(
        actuators,
        "position",
        name="base_rotation",
        joint="rotation",
        kp="0.8",
        kv="0.08",
        ctrlrange="-3.1416 3.1416",
    )

    parent = base
    previous_length = 0.0
    # Discrete elastic rod: ball joints provide bending and twist; passive
    # joint springs restore the straight shape. No joint angle commands.
    for section, count, length, radius, stiffness, damping in (
        ("shaft", 20, SHAFT_LENGTH, 0.003, 0.12, 0.002),
        ("tip", 8, TIP_LENGTH, 0.0024, 0.012, 0.0004),
    ):
        segment = length / count
        for i in range(count):
            # Compensate link weight so this soft demonstration rod follows
            # the inclined entry axis without gravitational sag into the wall.
            body = ET.SubElement(
                parent,
                "body",
                name=f"{section}_{i}",
                pos=f"0 0 {-previous_length}",
                gravcomp="1",
            )
            if section != "shaft" or i != 0:
                ET.SubElement(
                    body,
                    "joint",
                    name=f"{section}_joint_{i}",
                    type="ball",
                    stiffness=str(stiffness),
                    damping=str(damping),
                    armature="0.000001",
                    limited="true",
                    range="0 0.5",
                )
            ET.SubElement(
                body,
                "geom",
                name=f"{section}_geom_{i}",
                type="capsule",
                size=str(radius),
                fromto=f"0 0 0 0 0 {-segment}",
                density="1100",
            )
            if section == "shaft" and i == count - 1:
                ET.SubElement(body, "site", name="tip_root", pos=f"0 0 {-segment}")
            if (section == "shaft" and i == count - 1) or section == "tip":
                station = 0 if section == "shaft" else i + 1
                for name, x, y in (
                    ("xp", 1, 0),
                    ("xm", -1, 0),
                    ("yp", 0, 1),
                    ("ym", 0, -1),
                ):
                    ET.SubElement(
                        body,
                        "site",
                        name=f"guide_{name}_{station}",
                        pos=f"{x * 0.0019} {y * 0.0019} {-segment}",
                    )
            parent, previous_length = body, segment
    ET.SubElement(
        parent,
        "site",
        name="scope_tip",
        pos=f"0 0 {-previous_length}",
        rgba="1 0.5 0.2 1",
        size="0.0025",
        group="0",
    )
    camera_pos = f"0 0 {-previous_length - 0.0032}"
    ET.SubElement(
        parent,
        "camera",
        name=CAMERA_NAME,
        pos=camera_pos,
        fovy="90",
        ipd="0",
        mode="fixed",
    )
    ET.SubElement(
        parent,
        "light",
        name="tip_light",
        pos=camera_pos,
        dir="0 0 -1",
        type="spot",
        cutoff="75",
        exponent="3",
        castshadow="false",
        diffuse="0.65 0.65 0.65",
        ambient="0.02 0.02 0.02",
        attenuation="1 0 100",
    )
    for name in TENDONS:
        tendon = ET.SubElement(
            root.find("tendon"), "spatial", name=f"tendon_{name}", width="0.00035"
        )
        for station in range(9):
            ET.SubElement(tendon, "site", site=f"guide_{name}_{station}")
        # Negative gear converts positive actuator force to tendon shortening.
        ET.SubElement(
            actuators,
            "motor",
            name=f"pull_{name}",
            tendon=f"tendon_{name}",
            gear="-1",
            ctrlrange=f"0 {MAX_TENSION}",
        )

    model = mj.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"), assets=virtual_assets
    )
    # Keep the imported material/texture: the shared palette detaches materials.
    airway_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "airway")
    material_id = model.geom_matid[airway_id] if airway_id >= 0 else -1
    apply_palette(model, ("tip_", "rotation_marker"))
    if airway_id >= 0:
        model.geom_matid[airway_id] = material_id
        model.geom_rgba[airway_id] = [1, 1, 1, 1]
    return model


class BronchoscopeSim:
    def __init__(
        self,
        airway: Path | None,
        scale: float = 0.001,
        *,
        collision: Path | None = None,
        collisions: bool = True,
    ):
        self.model = build_model(
            airway, scale, collision=collision, collisions=collisions
        )
        self.data = mj.MjData(self.model)
        self.airway_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "airway")
        self.wall_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_FLEX, "airway_wall")
        self.camera_id = self.model.camera(CAMERA_NAME).id
        self.reset()
        self.insertion_depth = self.checked_insertion_depth()

    def reset(self) -> None:
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        self.begin_insertion()

    def begin_insertion(self) -> None:
        self.insertion_start = self.data.time
        self.insertion_height = float(self.data.ctrl[0])

    def checked_insertion_depth(self) -> float:
        """Check a straight approach against the real mesh triangles.

        This is a geometric path check, not contact response. Probe the wall
        radially along the route and stop before a narrowing or bend that the
        6 mm shaft cannot clear. Rays use the visual mesh, not its convex hull.
        """
        if self.airway_id < 0:
            return INSERTION_DEPTH
        entry_id = self.model.site("airway_entry").id
        entry = self.data.site_xpos[entry_id]
        frame = self.data.site_xmat[entry_id].reshape(3, 3)
        forward_hit = mj.mj_rayMesh(
            self.model,
            self.data,
            self.airway_id,
            entry + ENTRY_GAP * frame[:, 2],
            -frame[:, 2],
        )
        limit = INSERTION_DEPTH
        if forward_hit >= 0:
            limit = min(limit, max(0.0, forward_hit - 0.008))
        # The inlet is cut horizontally, while the shaft is slightly tilted;
        # start below its rim so transverse rays lie wholly inside the tube.
        for depth in np.arange(0.004, limit - ENTRY_GAP + 0.001, 0.001):
            point = entry - depth * frame[:, 2]
            distances = [
                mj.mj_rayMesh(
                    self.model,
                    self.data,
                    self.airway_id,
                    point,
                    np.cos(angle) * frame[:, 0] + np.sin(angle) * frame[:, 1],
                )
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ]
            if min(distances) < 0.004 or max(distances) > 0.04:
                return max(0.0, ENTRY_GAP + depth - 0.004)
        return limit

    def command(self, height=0.0, rotation=0.0, tensions=(0.0, 0.0, 0.0, 0.0)) -> None:
        values = np.asarray([height, rotation, *tensions], dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Expected finite height, rotation and four tendon tensions"
            )
        self.data.ctrl[:] = np.clip(
            values,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

    def step(self, count: int = 1) -> None:
        mj.mj_step(self.model, self.data, nstep=count)
        mj.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or np.any(self.data.warning.number):
            raise RuntimeError(
                "Simulation produced a non-finite state or MuJoCo warning"
            )

    def demo(self) -> None:
        elapsed = self.data.time - self.insertion_start - INSERTION_WAIT
        u = np.clip(elapsed / INSERTION_SECONDS, 0, 1)
        fraction = u**3 * (10 - 15 * u + 6 * u**2)
        target = min(self.insertion_height, -self.insertion_depth)
        self.command(
            height=self.insertion_height + fraction * (target - self.insertion_height)
        )

    def camera_image(self, renderer: mj.Renderer) -> np.ndarray:
        renderer.update_scene(self.data, camera=CAMERA_NAME)
        return renderer.render()

    def wall_contact_stats(self) -> tuple[int, float, float]:
        """Return wall contact count, summed normal force (N), max penetration (m)."""
        count, normal_force, penetration = 0, 0.0, 0.0
        if self.wall_id >= 0:
            wrench = np.empty(6)
            for i, contact in enumerate(self.data.contact):
                if self.wall_id in contact.flex:
                    mj.mj_contactForce(self.model, self.data, i, wrench)
                    count += 1
                    normal_force += max(0.0, float(wrench[0]))
                    penetration = max(penetration, -float(contact.dist))
        return count, normal_force, penetration

    def status(self, auto: bool) -> str:
        depth = -float(self.data.joint("insertion").qpos[0]) * 1000
        count, force, _ = self.wall_contact_stats()
        return (
            f"Tip camera | insertion {depth:.1f} / {-INSERTION_RANGE[0] * 1000:.0f} mm | "
            f"wall: {count} contacts, {force:.3f} N | "
            f"{'auto' if auto else 'manual'} | C: camera  V: inset  T: insert  P: pause  B: walls"
        )


def run_viewer(
    sim: BronchoscopeSim, auto: bool = False, show_collision: bool = False
) -> None:
    import mujoco.viewer

    keys = SimpleQueue()
    paused = False
    inset = True
    print(
        "Up/Down: base height; Left/Right: base rotation; W/S: +/-X bend; A/D: +/-Y bend"
    )
    print(
        "Space: release tendons; R: reset to manual; P: pause; T: toggle insertion; C: tip camera; V: camera inset."
    )
    with (
        mj.Renderer(sim.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH) as renderer,
        mujoco.viewer.launch_passive(
            sim.model, sim.data, key_callback=keys.put
        ) as viewer,
    ):
        viewer.cam.azimuth = 105
        viewer.cam.elevation = -18
        viewer.cam.distance = 1.0
        viewer.cam.lookat[:] = [0, 0, 0.32]
        viewer.opt.flexgroup[5] = True
        if sim.wall_id >= 0:
            sim.model.flex_rgba[sim.wall_id, 3] = 0.35 if show_collision else 0.0
        next_camera_frame = 0.0
        while viewer.is_running():
            start = time.monotonic()
            with viewer.lock():
                while True:
                    try:
                        key = keys.get_nowait()
                    except Empty:
                        break
                    if key == ord("P"):
                        paused = not paused
                    elif key == ord("T"):
                        auto = not auto
                        if auto:
                            sim.begin_insertion()
                    elif key == ord("C"):
                        if viewer.cam.type == mj.mjtCamera.mjCAMERA_FIXED:
                            viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
                        else:
                            viewer.cam.type = mj.mjtCamera.mjCAMERA_FIXED
                            viewer.cam.fixedcamid = sim.camera_id
                    elif key == ord("V"):
                        inset = not inset
                        if not inset:
                            viewer.clear_images()
                    elif key == ord("B") and sim.wall_id >= 0:
                        alpha = sim.model.flex_rgba[sim.wall_id, 3]
                        sim.model.flex_rgba[sim.wall_id, 3] = 0.0 if alpha else 0.35
                    elif key == ord("R"):
                        sim.reset()
                        auto = False
                    else:
                        ctrl = sim.data.ctrl.copy()
                        if key in (265, 264):  # GLFW up/down
                            ctrl[0] += 0.005 if key == 265 else -0.005
                        elif key in (263, 262):
                            ctrl[1] += 0.1 if key == 263 else -0.1
                        elif key in map(ord, "WSAD"):
                            index = "WSAD".index(chr(key))
                            ctrl[2 + (index ^ 1)] = 0
                            ctrl[2 + index] += 0.05
                        elif key == 32:
                            ctrl[2:] = 0
                        else:
                            continue
                        auto = False
                        sim.command(ctrl[0], ctrl[1], ctrl[2:])
                if not paused:
                    if auto:
                        sim.demo()
                    sim.step(10)
            if time.monotonic() >= next_camera_frame:
                if inset:
                    image = sim.camera_image(renderer)
                    viewport = viewer.viewport
                    # Fit the inset even when the user shrinks the window.
                    factor = min(
                        1.0,
                        max(1, viewport.width - 24) / CAMERA_WIDTH,
                        max(1, viewport.height - 24) / CAMERA_HEIGHT,
                    )
                    width, height = (
                        max(1, int(CAMERA_WIDTH * factor)),
                        max(1, int(CAMERA_HEIGHT * factor)),
                    )
                    if (width, height) != (CAMERA_WIDTH, CAMERA_HEIGHT):
                        import cv2

                        image = cv2.resize(image, (width, height))
                    viewer.set_images(
                        (
                            mj.MjrRect(
                                viewport.left + viewport.width - width - 12,
                                viewport.bottom + 12,
                                width,
                                height,
                            ),
                            image,
                        )
                    )
                viewer.set_texts(
                    (
                        mj.mjtFontScale.mjFONTSCALE_100,
                        mj.mjtGridPos.mjGRID_TOPLEFT,
                        sim.status(auto),
                        "",
                    )
                )
                next_camera_frame = time.monotonic() + 0.1
            viewer.sync()
            time.sleep(max(0, 0.01 - (time.monotonic() - start)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airway", type=Path, default=default_airway())
    parser.add_argument(
        "--airway-scale",
        type=float,
        default=0.001,
        help="OBJ units to metres (default: mm)",
    )
    parser.add_argument("--headless", action="store_true")
    collision_options = parser.add_mutually_exclusive_group()
    collision_options.add_argument(
        "--collision",
        type=Path,
        help="collision-only OBJ in the visual asset's frame/units",
    )
    collision_options.add_argument(
        "--no-collision",
        action="store_true",
        help="disable airway wall contact for comparison",
    )
    parser.add_argument(
        "--show-collision",
        action="store_true",
        help="show the collision surface overlay",
    )
    parser.add_argument("--duration", type=float, default=28.0)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="opt in to the slow automatic insertion demo",
    )
    parser.add_argument(
        "--manual", action="store_true", help="start in manual control (the default)"
    )
    parser.add_argument(
        "--camera-output",
        type=Path,
        help="save the final tip-camera PNG in headless mode",
    )
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if args.auto and args.manual:
        parser.error("--auto and --manual are mutually exclusive")
    if args.camera_output and not args.headless:
        parser.error("--camera-output requires --headless")
    sim = BronchoscopeSim(
        args.airway,
        args.airway_scale,
        collision=args.collision,
        collisions=not args.no_collision,
    )
    print(
        f"Airway: {args.airway}; checked insertion: {sim.insertion_depth * 1000:.0f} mm. "
        f"Wall collisions: {'enabled' if sim.wall_id >= 0 else 'disabled'}."
    )
    if args.headless:
        steps = int(np.ceil(args.duration / sim.model.opt.timestep))
        for start in range(0, steps, 10):
            if args.auto:
                sim.demo()
            sim.step(min(10, steps - start))
        tip = sim.data.site_xpos[sim.model.site("scope_tip").id]
        print(
            f"Completed {sim.data.time:.3f} s without warnings; tip: {tip.round(4)} m; "
            f"wall contacts/force/penetration: {sim.wall_contact_stats()}"
        )
        if args.camera_output:
            import cv2

            args.camera_output.parent.mkdir(parents=True, exist_ok=True)
            with mj.Renderer(
                sim.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH
            ) as renderer:
                image = sim.camera_image(renderer)
                if not cv2.imwrite(
                    str(args.camera_output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                ):
                    raise RuntimeError(
                        f"Cannot save camera image: {args.camera_output}"
                    )
            print(f"Tip-camera image: {args.camera_output}")
    else:
        run_viewer(sim, args.auto, args.show_collision)


if __name__ == "__main__":
    main()
