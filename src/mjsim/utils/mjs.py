import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union
from xml.dom import minidom

import mujoco as mj


def _spec_from_string(xml_str: str) -> mj.MjSpec:
    """
    Build an MjSpec from XML while keeping the original XML string around.
    """
    spec = mj.MjSpec().from_string(xml_str)
    try:
        setattr(spec, "_xml_string", xml_str)
    except Exception:
        # Some MjSpec implementations disallow setting new attributes; ignore in that case.
        pass
    if not hasattr(spec, "to_xml_string"):
        try:
            # Some bindings allow adding attributes to the class even if instances are frozen.
            def _to_xml_string(self, xml: str = xml_str) -> str:
                return getattr(self, "_xml_string", xml)

            setattr(type(spec), "to_xml_string", _to_xml_string)
        except Exception:
            # As a last resort, wrap in a lightweight proxy with the expected method.
            class _SpecProxy:
                def __init__(self, inner, xml):
                    self._inner = inner
                    self._xml = xml

                def __getattr__(self, name):
                    return getattr(self._inner, name)

                def to_xml_string(self):
                    return self._xml

            return _SpecProxy(spec, xml_str)
    return spec


def empty_scene(sim_name: str = "mj_sim", multiccd: bool = False) -> mj.MjSpec:
    multiccd = "enable" if multiccd else "disable"

    _XML = f"""
        <mujoco model="{sim_name}">

        <compiler angle="radian" autolimits="true" />
        <option timestep="0.002"
            integrator="implicitfast"
            solver="Newton"
            gravity="0 0 -9.82"
            cone="elliptic"
            sdf_iterations="5"
            sdf_initpoints="30"
            noslip_iterations="2"
        >
            <flag multiccd="enable" nativeccd="{multiccd}" />
        </option>

        <statistic center="0.3 0 0.3" extent="0.8" meansize="0.08" />

        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0" />
            <rgba haze="0.15 0.25 0.35 1" />
            <global azimuth="120" elevation="-20" offwidth="2000" offheight="2000" />
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
                height="3072" />
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
                rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
                reflectance="0.2" />
        </asset>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
            <geom name="floor" size="0 0 0.5" type="plane" material="groundplane"
                solimp="0.0 0.0 0.0 0.0 1" />
        </worldbody>
    </mujoco>
    """
    return _spec_from_string(_XML)


def _safe_mj_name(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_:/.-]+", "_", name).strip("_")
    return safe or "mesh"


def _vector(value: Sequence[float] | float, length: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * length
    if isinstance(value, str):
        msg = f"{name} must be a number or a sequence of {length} numbers"
        raise TypeError(msg)

    values = [float(x) for x in value]
    if len(values) != length:
        msg = f"{name} must contain {length} values, got {len(values)}"
        raise ValueError(msg)
    return values


def _optional_vector(
    value: Sequence[float] | None,
    length: int,
    name: str,
) -> list[float] | None:
    if value is None:
        return None
    return _vector(value, length, name)


def _geom_type(value: mj.mjtGeom | int | str, name: str) -> mj.mjtGeom | int:
    if not isinstance(value, str):
        return value

    key = value.strip().upper()
    if not key.startswith("MJGEOM_"):
        key = f"MJGEOM_{key}"
    key = key.replace("MJGEOM_", "mjGEOM_", 1)

    if not hasattr(mj.mjtGeom, key):
        msg = f"{name} must be a mujoco.mjtGeom value or valid geom type string"
        raise ValueError(msg)
    return getattr(mj.mjtGeom, key)


def _load_triangle_mesh(path: Path):
    import open3d as o3d

    loaded_mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if not loaded_mesh.has_vertices():
        msg = f"Mesh file {path} has no vertices"
        raise ValueError(msg)
    if not loaded_mesh.has_triangles():
        msg = f"Mesh file {path} has no triangles"
        raise ValueError(msg)
    return loaded_mesh


def _clean_triangle_mesh(triangle_mesh):
    triangle_mesh.remove_duplicated_vertices()
    triangle_mesh.remove_duplicated_triangles()
    triangle_mesh.remove_degenerate_triangles()
    triangle_mesh.remove_unreferenced_vertices()
    triangle_mesh.compute_vertex_normals()
    return triangle_mesh


def _decimate_triangle_mesh(
    triangle_mesh,
    decimation_ratio: float,
    decimation_method: Literal["quadric", "clustering", "none"],
    preserve_volume: bool,
    max_error: float | None,
    voxel_size: float | None,
):
    import numpy as np
    import open3d as o3d

    mesh_copy = o3d.geometry.TriangleMesh(triangle_mesh)
    if decimation_method == "none" or decimation_ratio >= 1.0:
        return _clean_triangle_mesh(mesh_copy)

    current_triangles = len(mesh_copy.triangles)
    target_triangles = max(4, int(current_triangles * decimation_ratio))

    if decimation_method == "quadric":
        decimated_mesh = mesh_copy.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles,
            maximum_error=np.inf if max_error is None else max_error,
            boundary_weight=1.0 if preserve_volume else 0.0,
        )
    elif decimation_method == "clustering":
        if voxel_size is None:
            diagonal = float(
                np.linalg.norm(mesh_copy.get_max_bound() - mesh_copy.get_min_bound())
            )
            voxel_size = max(diagonal, 1.0) * 0.01 * np.sqrt(1.0 / decimation_ratio)
        decimated_mesh = mesh_copy.simplify_vertex_clustering(voxel_size=voxel_size)
    else:
        msg = "decimation_method must be 'quadric', 'clustering', or 'none'"
        raise ValueError(msg)

    return _clean_triangle_mesh(decimated_mesh)


def _mesh_arrays(triangle_mesh) -> tuple[list[float], list[int]]:
    import numpy as np

    vertices = np.asarray(triangle_mesh.vertices, dtype=np.float64).reshape(-1)
    triangles = np.asarray(triangle_mesh.triangles, dtype=np.int32).reshape(-1)
    return vertices.tolist(), triangles.tolist()


def _collision_cache_path(
    source_path: Path,
    cache_dir: Path,
    decimation_ratio: float,
    decimation_method: str,
    preserve_volume: bool,
    max_error: float | None,
    voxel_size: float | None,
) -> Path:
    stat = source_path.stat()
    cache_key = "|".join(
        [
            "mjsim-mesh-v1",
            str(source_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(decimation_ratio),
            decimation_method,
            str(preserve_volume),
            str(max_error),
            str(voxel_size),
        ]
    )
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    return cache_dir / f"{_safe_mj_name(source_path.stem)}_collision_{digest}.obj"


def mesh(
    path: str | Path,
    model_name: str | None = None,
    name: str | None = None,
    scale: Sequence[float] | float = 1.0,
    pos: Sequence[float] | None = None,
    euler: Sequence[float] | None = None,
    quat: Sequence[float] | None = None,
    geom_pos: Sequence[float] | None = None,
    geom_euler: Sequence[float] | None = None,
    geom_quat: Sequence[float] | None = None,
    freejoint: bool = False,
    decimation_ratio: float = 0.05,
    decimation_method: Literal["quadric", "clustering", "none"] = "quadric",
    preserve_volume: bool = True,
    max_error: float | None = None,
    voxel_size: float | None = None,
    cache_dir: str | Path = ".cache",
    use_cache: bool = True,
    overwrite_cache: bool = False,
    visual: bool = True,
    collision: bool = True,
    visual_rgba: Sequence[float] = (0.9, 0.9, 0.95, 1.0),
    collision_rgba: Sequence[float] = (0.9, 0.0, 0.0, 0.3),
    collision_geom_type: mj.mjtGeom | int | str = mj.mjtGeom.mjGEOM_MESH,
    density: float = 1.0,
    friction: Sequence[float] = (0.1, 0.005, 0.0001),
    solref: Sequence[float] = (0.02, 1.0),
    solimp: Sequence[float] = (0.9, 0.95, 0.001, 0.5, 2.0),
    contype: int = 1,
    conaffinity: int = 1,
    visual_group: int = 2,
    collision_group: int = 3,
) -> mj.MjSpec:
    """
    Create an attachable MuJoCo mesh spec with separate visual and collision geoms.

    The source mesh is used for the visual geom. A simplified collision mesh is
    generated with Open3D and written to ``cache_dir`` so later runs can reuse it.

    Args:
        path: Mesh file to load. Any format supported by Open3D can be used.
        model_name: MuJoCo model name. Defaults to the sanitized file stem.
        name: Body and geom name stem. Defaults to ``model_name``.
        scale: Scalar or XYZ scale passed to both MuJoCo mesh assets.
        pos: Optional body position.
        euler: Optional body Euler rotation.
        quat: Optional body quaternion rotation.
        geom_pos: Optional local position for both visual and collision geoms.
        geom_euler: Optional local Euler rotation for both geoms.
        geom_quat: Optional local quaternion rotation for both geoms.
        freejoint: Add a free joint to the mesh body.
        decimation_ratio: Fraction of triangles to keep in the collision mesh.
        decimation_method: ``"quadric"``, ``"clustering"``, or ``"none"``.
        preserve_volume: Uses Open3D's boundary weight for quadric decimation.
        max_error: Optional quadric decimation error bound.
        voxel_size: Optional clustering voxel size.
        cache_dir: Directory for decimated collision OBJ files.
        use_cache: Reuse an existing collision cache file when available.
        overwrite_cache: Regenerate the collision cache even when it exists.
        visual: Add the high-resolution visual geom.
        collision: Add the simplified collision geom.
        visual_rgba: RGBA for the visual geom.
        collision_rgba: RGBA for the collision geom.
        collision_geom_type: MuJoCo geom type for collisions, e.g. ``"mesh"`` or
            ``"sdf"``.
        density: Density for the collision geom.
        friction: Collision geom friction.
        solref: Collision geom solver reference.
        solimp: Collision geom solver impedance.
        contype: Collision type bitmask.
        conaffinity: Collision affinity bitmask.
        visual_group: MuJoCo visualization group for the visual geom.
        collision_group: MuJoCo visualization group for the collision geom.

    Returns:
        An ``MjSpec`` containing a single body with mesh assets and geoms.
    """
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        msg = f"Mesh file does not exist: {source_path}"
        raise FileNotFoundError(msg)
    if not 0.0 < decimation_ratio <= 1.0:
        msg = f"decimation_ratio must be in the interval (0, 1], got {decimation_ratio}"
        raise ValueError(msg)
    if euler is not None and quat is not None:
        msg = "Pass either euler or quat for the body rotation, not both"
        raise ValueError(msg)
    if geom_euler is not None and geom_quat is not None:
        msg = "Pass either geom_euler or geom_quat for the geom rotation, not both"
        raise ValueError(msg)
    if not visual and not collision:
        msg = "At least one of visual or collision must be enabled"
        raise ValueError(msg)

    mesh_name = _safe_mj_name(name or model_name or source_path.stem)
    spec = mj.MjSpec()
    spec.modelname = _safe_mj_name(model_name or mesh_name)

    visual_mesh = _clean_triangle_mesh(_load_triangle_mesh(source_path))

    cache_path = _collision_cache_path(
        source_path=source_path,
        cache_dir=Path(cache_dir).expanduser(),
        decimation_ratio=decimation_ratio,
        decimation_method=decimation_method,
        preserve_volume=preserve_volume,
        max_error=max_error,
        voxel_size=voxel_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_path.exists() and not overwrite_cache:
        collision_mesh = _load_triangle_mesh(cache_path)
    else:
        collision_mesh = _decimate_triangle_mesh(
            triangle_mesh=visual_mesh,
            decimation_ratio=decimation_ratio,
            decimation_method=decimation_method,
            preserve_volume=preserve_volume,
            max_error=max_error,
            voxel_size=voxel_size,
        )

        import open3d as o3d

        if not o3d.io.write_triangle_mesh(
            str(cache_path),
            collision_mesh,
            write_ascii=True,
            write_vertex_normals=False,
        ):
            msg = f"Failed to write collision mesh cache: {cache_path}"
            raise OSError(msg)

    scale_xyz = _vector(scale, 3, "scale")
    body_kwargs = {"name": mesh_name}
    body_pos = _optional_vector(pos, 3, "pos")
    body_euler = _optional_vector(euler, 3, "euler")
    body_quat = _optional_vector(quat, 4, "quat")
    if body_pos is not None:
        body_kwargs["pos"] = body_pos
    if body_euler is not None:
        body_kwargs["euler"] = body_euler
    if body_quat is not None:
        body_kwargs["quat"] = body_quat

    body = spec.worldbody.add_body(**body_kwargs)
    if freejoint:
        body.add_freejoint()

    geom_kwargs = {}
    local_pos = _optional_vector(geom_pos, 3, "geom_pos")
    local_euler = _optional_vector(geom_euler, 3, "geom_euler")
    local_quat = _optional_vector(geom_quat, 4, "geom_quat")
    if local_pos is not None:
        geom_kwargs["pos"] = local_pos
    if local_euler is not None:
        geom_kwargs["euler"] = local_euler
    if local_quat is not None:
        geom_kwargs["quat"] = local_quat

    if visual:
        visual_vertices, visual_triangles = _mesh_arrays(visual_mesh)
        visual_asset_name = f"{mesh_name}_visual_mesh"
        spec.add_mesh(
            name=visual_asset_name,
            uservert=visual_vertices,
            userface=visual_triangles,
            scale=scale_xyz,
        )
        body.add_geom(
            name=f"{mesh_name}_visual",
            type=mj.mjtGeom.mjGEOM_MESH,
            meshname=visual_asset_name,
            rgba=_vector(visual_rgba, 4, "visual_rgba"),
            density=0.0,
            contype=0,
            conaffinity=0,
            group=visual_group,
            **geom_kwargs,
        )

    if collision:
        collision_vertices, collision_triangles = _mesh_arrays(collision_mesh)
        collision_asset_name = f"{mesh_name}_collision_mesh"
        spec.add_mesh(
            name=collision_asset_name,
            uservert=collision_vertices,
            userface=collision_triangles,
            scale=scale_xyz,
        )
        body.add_geom(
            name=f"{mesh_name}_collision",
            type=_geom_type(collision_geom_type, "collision_geom_type"),
            meshname=collision_asset_name,
            rgba=_vector(collision_rgba, 4, "collision_rgba"),
            density=density,
            friction=_vector(friction, 3, "friction"),
            solref=_vector(solref, 2, "solref"),
            solimp=_vector(solimp, 5, "solimp"),
            contype=contype,
            conaffinity=conaffinity,
            group=collision_group,
            **geom_kwargs,
        )

    try:
        setattr(
            spec,
            "_mesh_cache_path",
            str(cache_path),
        )
    except Exception:
        pass

    return spec


def dlo(
    model_name: str,
    prefix: str,
    curve: str,
    count: List[int],
    size: float,
    initial: str,
    twist: float,
    bend: float,
    vmax: float,
    segment_size: float,
    segment_geom_type: mj.mjtGeom,  # Using Any since mj.mjtGeom might not be available
    **kwargs,
) -> mj.MjSpec:  # Using Any for mj.MjSpec return type
    """
    Create a Deformable Linear Object (DLO) model specification using MuJoCo's elasticity plugin.

    This function generates a cable-like deformable object composed of multiple segments
    with configurable physical properties including elasticity, friction, and visual appearance.

    Additional geom and joint arguments can be passed using prefixes:
    - geom_* for geometry attributes (e.g., geom_solimp, geom_margin)
    - joint_* for joint attributes (e.g., joint_stiffness, joint_axis)

    For full information check the documentation:
    https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-composite

    Args:
        model_name: Name of the model.
        prefix: Prefix for composite elements in the model.
        curve: Curve specification for the DLO shape (e.g., "0 s 0" for straight line).
        count: Number of segments in each dimension as [x_count, y_count, z_count].
        size: Overall size/length of the DLO.
        initial: Initial condition specification ("free", "ball", "none").
        twist: Twist stiffness coefficient for the DLO.
        bend: Bending stiffness coefficient for the DLO.
        vmax: Maximum velocity constraint. Use 0 for no constraint.
        segment_size: Size parameter for each segment geometry.
        segment_geom_type: Geometry type for segments (from mj.mjtGeom enum).
        **kwargs: Additional arguments with prefixes:
                - geom_*: Geometry attributes (e.g., geom_rgba, geom_mass, geom_friction)
                - joint_*: Joint attributes (e.g., joint_damping, joint_armature)

    Returns:
        mj.MjSpec: MuJoCo model specification object.

    Raises:
        AssertionError: If input validation fails for any parameter.

    Examples:
        >>> # Create a basic DLO with required arguments only
        >>> dlo_spec = test_dlo(
        ...     model_name="rope",
        ...     prefix="rope:",
        ...     curve="0 s 0",
        ...     count=[10, 1, 1],
        ...     size=1.0,
        ...     initial="free",
        ...     twist=60000.0,
        ...     bend=10000000.0,
        ...     vmax=0,
        ...     segment_size=0.002,
        ...     segment_geom_type=mj.mjtGeom.mjGEOM_CAPSULE
        ... )

        >>> # Create a DLO with additional geom and joint arguments
        >>> dlo_spec = test_dlo(
        ...     model_name="advanced_rope",
        ...     prefix="rope:",
        ...     curve="0 s 0",
        ...     count=[15, 1, 1],
        ...     size=1.5,
        ...     initial="free",
        ...     twist=50000.0,
        ...     bend=8000000.0,
        ...     vmax=0,
        ...     segment_size=0.003,
        ...     segment_geom_type=mj.mjtGeom.mjGEOM_CAPSULE,
        ...     geom_rgba=[0.8, 0.2, 0.2, 1.0],
        ...     geom_mass="0.015",
        ...     geom_friction="0.5 0.1 0.1",
        ...     geom_condim="4",
        ...     geom_solref="0.00002 0.8",
        ...     geom_solimp="0.9 0.95 0.001",
        ...     joint_damping="0.02",
        ...     joint_armature="0.0005",
        ...     joint_stiffness="100"
        ... )
    """
    # Separate geom and joint arguments from kwargs
    geom_args = {}
    joint_args = {}
    other_args = {}

    for key, value in kwargs.items():
        if key.startswith("geom_"):
            geom_args[key[5:]] = value  # Remove 'geom_' prefix
        elif key.startswith("joint_"):
            joint_args[key[6:]] = value  # Remove 'joint_' prefix
        else:
            other_args[key] = value

    if other_args:
        print(f"Warning: Unprefixed kwargs will be ignored: {list(other_args.keys())}")

    # Extract geometry type string from enum
    geom_type = segment_geom_type.__str__().split("_")[-1].lower()

    # Input validation
    assert isinstance(model_name, str) and model_name, (
        f"model_name must be a non-empty string, got {model_name=}"
    )
    assert isinstance(prefix, str) and prefix, (
        f"prefix must be a non-empty string, got {prefix=}"
    )
    assert isinstance(curve, str) and curve, (
        f"curve must be a non-empty string, got {curve=}"
    )

    # Validate count
    assert isinstance(count, list) and len(count) == 3, (
        f"count must be a list of 3 integers, got {count=}"
    )
    assert all(isinstance(x, int) and x > 0 for x in count), (
        f"All count values must be positive integers, got {count=}"
    )

    # Validate numeric parameters
    assert isinstance(size, (int, float)) and size > 0, (
        f"size must be positive, got {size=}"
    )
    assert isinstance(segment_size, (int, float)) and segment_size > 0, (
        f"segment_size must be positive, got {segment_size=}"
    )
    assert isinstance(twist, (int, float)) and twist >= 0, (
        f"twist must be non-negative, got {twist=}"
    )
    assert isinstance(bend, (int, float)) and bend >= 0, (
        f"bend must be non-negative, got {bend=}"
    )
    assert isinstance(vmax, (int, float)) and vmax >= 0, (
        f"vmax must be non-negative, got {vmax=}"
    )

    # Validate initial condition
    valid_initial_conditions = {"free", "ball", "none"}
    assert isinstance(initial, str), f"initial must be a string, got {initial=}"
    if initial not in valid_initial_conditions:
        print(
            f"Warning: {initial=} is not a common initial condition. "
            f"Common values are: {valid_initial_conditions}"
        )

    # Validate geometry type
    valid_geom_types = {"capsule", "box", "sphere", "cylinder", "ellipsoid", "mesh"}
    assert geom_type in valid_geom_types, (
        f"Invalid geometry type: {geom_type}. Must be one of {valid_geom_types}"
    )

    # Create XML structure using ElementTree
    mujoco = ET.Element("mujoco", model=model_name)

    # Add extension
    extension = ET.SubElement(mujoco, "extension")
    ET.SubElement(extension, "plugin", plugin="mujoco.elasticity.cable")

    # Add worldbody
    worldbody = ET.SubElement(mujoco, "worldbody")

    # Create composite element
    composite = ET.SubElement(
        worldbody,
        "composite",
        prefix=prefix,
        type="cable",
        curve=curve,
        count=" ".join(str(x) for x in count),
        size=str(size),
        initial=initial,
    )

    # Add plugin to composite
    plugin = ET.SubElement(composite, "plugin", plugin="mujoco.elasticity.cable")
    ET.SubElement(plugin, "config", key="twist", value=str(twist))
    ET.SubElement(plugin, "config", key="bend", value=str(bend))
    ET.SubElement(plugin, "config", key="vmax", value=str(vmax))

    # Create joint attributes with defaults and additional joint_args
    joint_attribs = {
        "kind": "main",
        "damping": str(joint_args.get("damping", "0.01")),  # Default value
        "armature": str(joint_args.get("armature", "0.001")),  # Default value
    }
    # Add all other joint arguments (excluding already used ones)
    for key, value in joint_args.items():
        if key not in ["damping", "armature"]:
            joint_attribs[key] = str(value)

    ET.SubElement(composite, "joint", **joint_attribs)

    # Create geom attributes with defaults and additional geom_args
    geom_attribs = {
        "type": geom_type,
        "size": str(segment_size),
        "rgba": " ".join(
            str(x) for x in geom_args.get("rgba", [0.2, 0.2, 0.2, 1.0])
        ),  # Default
        "mass": str(geom_args.get("mass", "0.02")),  # Default
        "friction": geom_args.get("friction", "0.3 0.3 0.3"),  # Default
        "condim": str(geom_args.get("condim", 4)),  # Default
        "solref": geom_args.get("solref", "0.00001 1"),  # Default
    }
    # Add all other geom arguments (excluding already used ones)
    for key, value in geom_args.items():
        if key not in ["rgba", "mass", "friction", "condim", "solref"]:
            geom_attribs[key] = str(value)

    ET.SubElement(composite, "geom", **geom_attribs)

    # Convert to string with pretty printing
    try:
        ET.indent(mujoco)  # Python 3.9+
        xml_str = ET.tostring(mujoco, encoding="unicode", method="xml")
    except AttributeError:
        # Fallback for Python < 3.9
        xml_str = minidom.parseString(
            ET.tostring(mujoco, encoding="unicode")
        ).toprettyxml(indent="  ")

    return _spec_from_string(xml_str)


def dqo(
    model_name: str,
    prefix: str,
    count: List[int],
    size: float,
    initial: str,
    stretch: float,
    bend: float,
    shear: float,
    segment_size: float,
    segment_geom_type: mj.mjtGeom,
    **kwargs,
) -> mj.MjSpec:
    """
    Create a Deformable Quadratic Object (DQO) model specification using MuJoCo's elasticity plugin.

    This function generates a cloth-like deformable object composed of a grid of segments
    with configurable physical properties including stretch, bend, and shear stiffness.

    Additional geom and joint arguments can be passed using prefixes:
    - geom_* for geometry attributes (e.g., geom_solimp, geom_margin)
    - joint_* for joint attributes (e.g., joint_stiffness, joint_axis)

    For full information check the documentation:
    https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-composite

    Args:
        model_name: Name of the model.
        prefix: Prefix for composite elements in the model.
        count: Number of segments in each dimension as [x_count, y_count, z_count].
               For cloth, typically use [width_segments, height_segments, 1].
        size: Overall size of the DQO (width, height).
        initial: Initial condition specification ("flat", "ball", "none", "free").
        stretch: Stretch stiffness coefficient for the cloth.
        bend: Bending stiffness coefficient for the cloth.
        shear: Shear stiffness coefficient for the cloth.
        segment_size: Size parameter for each segment geometry (thickness for cloth).
        segment_geom_type: Geometry type for segments (from mj.mjtGeom enum).
        **kwargs: Additional arguments with prefixes:
                - geom_*: Geometry attributes (e.g., geom_rgba, geom_mass, geom_friction)
                - joint_*: Joint attributes (e.g., joint_damping, joint_armature)

    Returns:
        mj.MjSpec: MuJoCo model specification object.

    Raises:
        AssertionError: If input validation fails for any parameter.

    Examples:
        >>> # Create a basic cloth with 10x10 segments
        >>> cloth_spec = dqo(
        ...     model_name="cloth",
        ...     prefix="cloth:",
        ...     count=[10, 10, 1],
        ...     size=1.0,
        ...     initial="flat",
        ...     stretch=10000.0,
        ...     bend=1000.0,
        ...     shear=5000.0,
        ...     segment_size=0.001,
        ...     segment_geom_type=mj.mjtGeom.mjGEOM_BOX
        ... )

        >>> # Create a cloth with additional geom and joint arguments
        >>> cloth_spec = dqo(
        ...     model_name="advanced_cloth",
        ...     prefix="cloth:",
        ...     count=[20, 20, 1],
        ...     size=2.0,
        ...     initial="ball",
        ...     stretch=15000.0,
        ...     bend=2000.0,
        ...     shear=8000.0,
        ...     segment_size=0.002,
        ...     segment_geom_type=mj.mjtGeom.mjGEOM_BOX,
        ...     geom_rgba=[0.2, 0.8, 0.2, 0.8],
        ...     geom_mass="0.005",
        ...     geom_friction="1.0 0.5 0.1",
        ...     geom_condim="4",
        ...     geom_solref="0.0001 0.9",
        ...     geom_solimp="0.8 0.9 0.001",
        ...     joint_damping="0.1",
        ...     joint_armature="0.0001"
        ... )

        >>> # Create a 3D volumetric deformable object
        >>> volume_spec = dqo(
        ...     model_name="jelly",
        ...     prefix="jelly:",
        ...     count=[8, 8, 8],
        ...     size=0.5,
        ...     initial="free",
        ...     stretch=50000.0,
        ...     bend=5000.0,
        ...     shear=25000.0,
        ...     segment_size=0.005,
        ...     segment_geom_type=mj.mjtGeom.mjGEOM_BOX,
        ...     geom_rgba=[0.8, 0.3, 0.8, 0.6]
        ... )
    """
    # Separate geom and joint arguments from kwargs
    geom_args = {}
    joint_args = {}
    other_args = {}

    for key, value in kwargs.items():
        if key.startswith("geom_"):
            geom_args[key[5:]] = value  # Remove 'geom_' prefix
        elif key.startswith("joint_"):
            joint_args[key[6:]] = value  # Remove 'joint_' prefix
        else:
            other_args[key] = value

    if other_args:
        print(f"Warning: Unprefixed kwargs will be ignored: {list(other_args.keys())}")

    # Extract geometry type string from enum
    geom_type = segment_geom_type.__str__().split("_")[-1].lower()

    # Input validation
    assert isinstance(model_name, str) and model_name, (
        f"model_name must be a non-empty string, got {model_name=}"
    )
    assert isinstance(prefix, str) and prefix, (
        f"prefix must be a non-empty string, got {prefix=}"
    )

    # Validate count - for DQO we typically want at least 2D grid
    assert isinstance(count, list) and len(count) == 3, (
        f"count must be a list of 3 integers, got {count=}"
    )
    assert all(isinstance(x, int) and x > 0 for x in count), (
        f"All count values must be positive integers, got {count=}"
    )
    # For cloth-like objects, at least two dimensions should be > 1
    dims_greater_than_one = sum(1 for x in count if x > 1)
    assert dims_greater_than_one >= 1, (
        f"For DQO, at least one dimension in count should be > 1, got {count=}"
    )

    # Validate numeric parameters
    assert isinstance(size, (int, float)) and size > 0, (
        f"size must be positive, got {size=}"
    )
    assert isinstance(segment_size, (int, float)) and segment_size > 0, (
        f"segment_size must be positive, got {segment_size=}"
    )
    assert isinstance(stretch, (int, float)) and stretch >= 0, (
        f"stretch must be non-negative, got {stretch=}"
    )
    assert isinstance(bend, (int, float)) and bend >= 0, (
        f"bend must be non-negative, got {bend=}"
    )
    assert isinstance(shear, (int, float)) and shear >= 0, (
        f"shear must be non-negative, got {shear=}"
    )

    # Validate initial condition
    valid_initial_conditions = {"flat", "ball", "none", "free"}
    assert isinstance(initial, str), f"initial must be a string, got {initial=}"
    if initial not in valid_initial_conditions:
        print(
            f"Warning: {initial=} is not a common initial condition. "
            f"Common values are: {valid_initial_conditions}"
        )

    # Validate geometry type
    valid_geom_types = {"box", "sphere", "capsule", "cylinder", "ellipsoid", "mesh"}
    assert geom_type in valid_geom_types, (
        f"Invalid geometry type: {geom_type}. Must be one of {valid_geom_types}"
    )

    # Create XML structure using ElementTree
    mujoco = ET.Element("mujoco", model=model_name)

    # Add extension
    ET.SubElement(mujoco, "extension")
    # ET.SubElement(extension, "plugin", plugin="mujoco.elasticity.grid")

    # Add worldbody
    worldbody = ET.SubElement(mujoco, "worldbody")

    # Create composite element - using "grid" type for cloth
    composite = ET.SubElement(
        worldbody,
        "composite",
        prefix=prefix,
        type="flex",
        count=" ".join(str(x) for x in count),
        size=str(size),
        initial=initial,
    )

    # Create joint attributes with defaults and additional joint_args
    joint_attribs = {
        "kind": "main",
        "damping": str(joint_args.get("damping", "0.05")),  # Higher default for cloth
        "armature": str(
            joint_args.get("armature", "0.0005")
        ),  # Lower default for cloth
    }
    # Add all other joint arguments (excluding already used ones)
    for key, value in joint_args.items():
        if key not in ["damping", "armature"]:
            joint_attribs[key] = str(value)

    ET.SubElement(composite, "joint", **joint_attribs)

    # Create geom attributes with defaults and additional geom_args
    geom_attribs = {
        "type": geom_type,
        "size": str(segment_size),
        "rgba": " ".join(
            str(x) for x in geom_args.get("rgba", [0.3, 0.5, 0.8, 0.7])
        ),  # Default semi-transparent blue
        "mass": str(geom_args.get("mass", "0.01")),  # Lower default mass for cloth
        "friction": geom_args.get("friction", "1.0 0.3 0.1"),  # Higher sliding friction
        "condim": str(geom_args.get("condim", 4)),  # Default
        "solref": geom_args.get("solref", "0.0001 0.9"),  # Different default for cloth
    }
    # Add all other geom arguments (excluding already used ones)
    for key, value in geom_args.items():
        if key not in ["rgba", "mass", "friction", "condim", "solref"]:
            geom_attribs[key] = str(value)

    ET.SubElement(composite, "geom", **geom_attribs)

    # Convert to string with pretty printing
    try:
        ET.indent(mujoco)  # Python 3.9+
        xml_str = ET.tostring(mujoco, encoding="unicode", method="xml")
    except AttributeError:
        # Fallback for Python < 3.9
        xml_str = minidom.parseString(
            ET.tostring(mujoco, encoding="unicode")
        ).toprettyxml(indent="  ")

    return _spec_from_string(xml_str)


def dco(**kwargs) -> mj.MjSpec:
    """
    Convenience alias for a cable-like deformable object.

    Any keyword arguments are forwarded to :func:`cable`.
    """
    return cable(**kwargs)


def dmo(**kwargs) -> mj.MjSpec:
    """
    Convenience alias for a cloth-like deformable object.

    Any keyword arguments are forwarded to :func:`cloth`.
    """
    return cloth(**kwargs)


def cable(
    model_name: str = "cable",
    prefix: Optional[str] = None,
    curve: str = "0 s 0",
    n_segments: int = 10,
    twist: float = 60000.0,
    bend: float = 10000000.0,
    vmax: float = 0,
    length: float = 1,
    segment_size: float = 0.002,
    mass: float = 0.2,
    initial: str = "free",
    **kw_args,
) -> mj.MjSpec:
    """
    Create a cable model with minimal required parameters.

    This is a simplified interface for creating cable-like deformable linear objects
    with sensible defaults. Additional parameters can be passed through kw_args
    using geom_* and joint_* prefixes for fine-grained control.

    Args:
        model_name: Name of the cable model. Defaults to "cable".
        prefix: Prefix for composite elements. If None, uses "model_name:".
        curve: Curve specification for cable shape. Defaults to "0 s 0" (straight line).
        n_segments: Number of segments in the cable. Must be > 0. Defaults to 10.
        twist: Twist stiffness coefficient. Defaults to 60000.0.
        bend: Bending stiffness coefficient. Defaults to 10000000.0.
        vmax: Maximum velocity constraint. Use 0 for no constraint. Defaults to 0.
        length: Total length of the cable. Must be > 0. Defaults to 1.
        segment_size: Radius of each cable segment. Defaults to 0.002.
        mass: Total mass of the cable. Must be > 0. Defaults to 0.2.
        initial: Initial condition. Defaults to "free".
        **kw_args: Additional arguments passed to test_dlo with geom_* and joint_* prefixes.
                   Examples: geom_rgba, geom_friction, joint_damping, etc.

    Returns:
        mj.MjSpec: MuJoCo model specification object.

    Raises:
        AssertionError: If input validation fails.

    Examples:
        >>> # Create a basic cable with default parameters
        >>> cable_spec = cable()

        >>> # Create a custom cable
        >>> cable_spec = cable(
        ...     model_name="my_cable",
        ...     n_segments=20,
        ...     length=2.0,
        ...     mass=0.3,
        ...     twist=50000.0
        ... )

        >>> # Create a cable with custom appearance and physics
        >>> cable_spec = cable(
        ...     n_segments=15,
        ...     length=1.5,
        ...     geom_rgba=[0.8, 0.2, 0.2, 1.0],
        ...     geom_friction="0.5 0.1 0.1",
        ...     joint_damping="0.02",
        ...     geom_solimp="0.9 0.95 0.001"
        ... )
    """
    # Input validation
    assert n_segments > 0, f"n_segments should be above 0, got {n_segments=}"
    assert length > 0, f"length should be above 0, got {length=}"
    assert mass > 0, f"mass should be above 0, got {mass=}"

    segment_mass = mass / n_segments
    prefix = f"{model_name}:" if prefix is None else prefix

    return dlo(
        model_name=model_name,
        prefix=prefix,
        curve=curve,
        count=[n_segments, 1, 1],
        size=length,
        initial=initial,
        twist=twist,
        bend=bend,
        vmax=vmax,
        segment_size=segment_size,
        segment_geom_type=mj.mjtGeom.mjGEOM_CAPSULE,  # Default to capsule geometry
        geom_mass=str(segment_mass),
        geom_rgba=kw_args.pop("geom_rgba", [0.2, 0.2, 0.2, 1]),  # Default color
        **kw_args,  # Pass any additional geom_*/joint_* arguments
    )


def cloth(
    model_name: str = "cloth",
    prefix: Optional[str] = None,
    width_segments: int = 10,
    height_segments: int = 10,
    width: float = 1.0,
    height: float = 1.0,
    stretch: float = 10000.0,
    bend: float = 1000.0,
    shear: float = 5000.0,
    thickness: float = 0.001,
    mass: float = 0.1,
    initial: str = "flat",
    **kwargs,
) -> mj.MjSpec:
    """
    Create a cloth model with explicit width and height control.
    """
    # Input validation
    assert width_segments > 0, (
        f"width_segments should be above 0, got {width_segments=}"
    )
    assert height_segments > 0, (
        f"height_segments should be above 0, got {height_segments=}"
    )
    assert width > 0, f"width should be above 0, got {width=}"
    assert height > 0, f"height should be above 0, got {height=}"
    assert mass > 0, f"mass should be above 0, got {mass=}"
    assert thickness > 0, f"thickness should be above 0, got {thickness=}"

    # Calculate segment mass
    total_segments = width_segments * height_segments
    segment_mass = mass / total_segments

    prefix = f"{model_name}:" if prefix is None else prefix

    # For cloth, we can use a size that approximates the aspect ratio
    # or use the larger dimension as size
    size = max(width, height)

    return dqo(
        model_name=model_name,
        prefix=prefix,
        count=[width_segments, height_segments, 1],
        size=size,
        initial=initial,
        stretch=stretch,
        bend=bend,
        shear=shear,
        segment_size=thickness,
        segment_geom_type=mj.mjtGeom.mjGEOM_BOX,
        geom_mass=str(segment_mass),
        geom_rgba=kwargs.pop("geom_rgba", [0.3, 0.5, 0.8, 0.7]),
        # You could also pass width/height as geom_args if the plugin supports them
        **kwargs,
    )


def deform_3d(
    model_name: str = "soft_block",
    prefix: str | None = None,
    resolution: int = 5,
    size: float = 0.25,
    stiffness: float = 2.0e4,
    shear: float = 1.0e4,
    bend: float = 2.0e3,
) -> mj.MjSpec:
    """
    Create a small volumetric deformable block.
    """
    prefix = f"{model_name}:" if prefix is None else prefix
    return dqo(
        model_name=model_name,
        prefix=prefix,
        count=[resolution, resolution, resolution],
        size=size,
        initial="free",
        stretch=stiffness,
        bend=bend,
        shear=shear,
        segment_size=size / (resolution * 2),
        segment_geom_type=mj.mjtGeom.mjGEOM_BOX,
    )


def deform_3d_custom(
    count: list[int],
    size: float,
    stretch: float,
    bend: float,
    shear: float,
    **kwargs,
) -> mj.MjSpec:
    """
    Flexible helper that forwards directly to :func:`dqo`.
    """
    return dqo(
        model_name=kwargs.pop("model_name", "custom_flex"),
        prefix=kwargs.pop("prefix", "flex:"),
        count=count,
        size=size,
        initial=kwargs.pop("initial", "free"),
        stretch=stretch,
        bend=bend,
        shear=shear,
        segment_size=kwargs.pop("segment_size", size / max(count)),
        segment_geom_type=kwargs.pop("segment_geom_type", mj.mjtGeom.mjGEOM_BOX),
        **kwargs,
    )


def replicate(
    model_name: str = "replicated",
    prefix: str = "replicate:",
    count: int = 1,
    sep: str = " ",
    euler: str = "0 0 0",
    pos: str = "0 0 0",
    geom_type: str = "box",
    geom_size: str = "0.01 0.01 0.01",
    geom_pos: str = "0 0 0",
    geom_friction: str = "0.2 0.2 0.2",
    geom_solref: str = "0.000000001 1",
    **kwargs,
) -> mj.MjSpec:
    """
    Create a replicated geometry model using MuJoCo's replicate composite type.

    This function generates multiple copies of a geometry with specified separation
    and transformation between each copy.

    Args:
        model_name: Name of the model.
        prefix: Prefix for composite elements.
        count: Number of replications.
        sep: Separation between replications (space-separated XYZ or "hole:prefix").
        euler: Euler angles for each replication (space-separated XYZ rotations in degrees).
        pos: Initial position of the composite.
        geom_type: Type of geometry ("box", "sphere", "capsule", "cylinder", "ellipsoid").
        geom_size: Size parameters for the geometry.
        geom_pos: Position offset for the geometry relative to its body.
        geom_friction: Friction coefficients (sliding, torsional, rolling).
        geom_solref: Solver reference parameters for constraint stabilization.
        **kwargs: Additional geom attributes (e.g., rgba, mass, condim).

    Returns:
        mj.MjSpec: MuJoCo model specification object.

    Examples:
        >>> # Create a simple chain of boxes
        >>> chain = replicate(
        ...     count=5,
        ...     sep="0.02 0 0",
        ...     geom_size="0.01 0.01 0.05"
        ... )

        >>> # Create a spiral pattern
        >>> spiral = replicate(
        ...     count=20,
        ...     sep="hole:spiral",
        ...     euler="0 0 18",  # 18 degree rotation each step
        ...     geom_type="capsule",
        ...     geom_size="0.005 0.02",
        ...     geom_rgba="1 0 0 1"
        ... )
    """
    # Input validation
    assert count > 0, f"count must be positive, got {count=}"
    assert geom_type in {"box", "sphere", "capsule", "cylinder", "ellipsoid"}, (
        f"Invalid geom_type: {geom_type}"
    )

    # Create XML structure
    mujoco = ET.Element("mujoco", model=model_name)
    worldbody = ET.SubElement(mujoco, "worldbody")

    # Create body with initial position
    body = ET.SubElement(worldbody, "body", pos=pos, euler=euler)

    # Create composite with replicate type
    composite = ET.SubElement(
        body,
        "composite",
        type="replicate",
        prefix=prefix,
        count=str(count),
        sep=sep,
        euler=euler,
    )

    # Create geom with specified attributes and any additional kwargs
    geom_attribs = {
        "type": geom_type,
        "pos": geom_pos,
        "size": geom_size,
        "friction": geom_friction,
        "solref": geom_solref,
    }

    # Add any additional geom attributes from kwargs
    for key, value in kwargs.items():
        geom_attribs[key] = str(value)

    ET.SubElement(composite, "geom", **geom_attribs)

    # Convert to string with pretty printing
    try:
        ET.indent(mujoco)
        xml_str = ET.tostring(mujoco, encoding="unicode", method="xml")
    except AttributeError:
        xml_str = minidom.parseString(ET.tostring(mujoco)).toprettyxml(indent="  ")

    return _spec_from_string(xml_str)


def pipe(
    length: float = 0.1,
    count: int = 30,
    segment_length: float = 0.1,
    radius: float = 0.004,
    thickness: float = 0.001,
    model_name: str = "pipe",
    **kwargs,
) -> mj.MjSpec:
    """
    Create a pipe-like structure using replicated box geometries.

    This is a specialized wrapper around replicate() that creates a pipe
    composed of multiple box segments arranged in a circular pattern.

    Args:
        length: Total length of the pipe segment.
        count: Number of segments in the pipe.
        segment_length: Length of each individual segment.
        radius: Radius of the pipe.
        thickness: Thickness of the pipe walls.
        model_name: Name of the pipe model.
        **kwargs: Additional arguments passed to replicate().

    Returns:
        mj.MjSpec: MuJoCo model specification object.

    Examples:
        >>> # Create a basic pipe
        >>> pipe_spec = pipe()

        >>> # Create a longer pipe with more segments
        >>> pipe_spec = pipe(
        ...     length=0.2,
        ...     count=50,
        ...     radius=0.01
        ... )

        >>> # Create a pipe with custom appearance
        >>> pipe_spec = pipe(
        ...     geom_rgba="0.8 0.2 0.2 1",
        ...     geom_friction="0.5 0.1 0.1"
        ... )
    """
    # Calculate geometry size for box segments
    geom_size = f"{radius} {thickness} {segment_length / 2}"

    # Use replicate to create the pipe structure
    return replicate(
        model_name=model_name,
        prefix="pipe:",
        count=count,
        sep="hole:pipe",  # Circular hole pattern
        euler="0 0 0",  # No additional rotation between segments
        pos="0 0 0",
        geom_type="box",
        geom_size=geom_size,
        geom_pos=f"0 -{radius + thickness / 2} 0",  # Position to form circle
        geom_friction=kwargs.pop("geom_friction", "0.2 0.2 0.2"),
        geom_solref=kwargs.pop("geom_solref", "0.000000001 1"),
        **kwargs,
    )


# def mesh()
