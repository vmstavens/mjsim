import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence
from xml.dom import minidom

import mujoco as mj
import numpy as np


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
            def _to_xml_string(self) -> str:
                if hasattr(self, "_xml_string"):
                    return self._xml_string
                return self.to_xml()

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


def _ensure_to_xml_string(spec: mj.MjSpec) -> mj.MjSpec:
    if not hasattr(spec, "to_xml_string"):
        try:
            setattr(type(spec), "to_xml_string", lambda self: self.to_xml())
        except Exception:
            pass
    return spec


def _parse_memory(value: int | str | None) -> int:
    if value is None:
        return -1
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        msg = f"memory must be an int, a size string, or None, got {type(value)}"
        raise TypeError(msg)

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMG]?)(?:I?B)?\s*", value.upper())
    if match is None:
        msg = f"Invalid memory value {value!r}. Use bytes or strings like '10M'."
        raise ValueError(msg)

    amount = float(match.group(1))
    unit = match.group(2)
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[unit]
    return int(amount * multiplier)


def _enum_value(value: int | str, enum_type: Any, prefix: str, name: str) -> int:
    if not isinstance(value, str):
        return int(value)

    key = value.strip().upper()
    key = key.replace("-", "_")
    if key.startswith(f"{prefix}_"):
        enum_name = key
    elif key.startswith("MJ"):
        enum_name = key
    else:
        enum_name = f"{prefix}_{key}"

    if not hasattr(enum_type, enum_name):
        valid = sorted(attr for attr in dir(enum_type) if attr.startswith(prefix))
        msg = f"{name} must be one of {valid}, got {value!r}"
        raise ValueError(msg)
    return int(getattr(enum_type, enum_name))


def _set_attrs(target: Any, attrs: dict[str, Any] | None, target_name: str) -> None:
    if attrs is None:
        return
    for key, value in attrs.items():
        if not hasattr(target, key):
            msg = f"{target_name} has no attribute {key!r}"
            raise AttributeError(msg)
        setattr(target, key, value)


def _set_enable_flag(option: Any, flag: int, enabled: bool) -> None:
    if enabled:
        option.enableflags |= flag
    else:
        option.enableflags &= ~flag


def _set_disable_flag(option: Any, flag: int, disabled: bool) -> None:
    if disabled:
        option.disableflags |= flag
    else:
        option.disableflags &= ~flag


def empty_scene(
    sim_name: str = "mj_sim",
    multiccd: bool = False,
    memory: int | str | None = "10M",
    solver: mj.mjtSolver | int | str = "Newton",
    integrator: mj.mjtIntegrator | int | str = "implicitfast",
    timestep: float = 0.002,
    tolerance: float | None = None,
    iterations: int | None = None,
    cone: mj.mjtCone | int | str = "elliptic",
    gravity: Sequence[float] = (0.0, 0.0, -9.82),
    sdf_iterations: int = 5,
    sdf_initpoints: int = 30,
    noslip_iterations: int = 2,
    enable_multiccd: bool = True,
    nativeccd: bool | None = None,
    compiler_angle: Literal["radian", "degree"] = "radian",
    autolimits: bool = True,
    statistic_center: Sequence[float] = (0.3, 0.0, 0.3),
    statistic_extent: float = 0.8,
    statistic_meansize: float = 0.08,
    offwidth: int = 2000,
    offheight: int = 2000,
    add_assets: bool = True,
    add_light: bool = True,
    add_floor: bool = True,
    floor_size: Sequence[float] = (0.0, 0.0, 0.5),
    floor_solimp: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 1.0),
    option_overrides: dict[str, Any] | None = None,
    compiler_overrides: dict[str, Any] | None = None,
    visual_overrides: dict[str, dict[str, Any]] | None = None,
    size_overrides: dict[str, int] | None = None,
) -> mj.MjSpec:
    """
    Build a reusable base MuJoCo scene with the ``MjSpec`` API.

    Args:
        sim_name: Model name.
        multiccd: Backwards-compatible switch for native CCD. The scene keeps
            MuJoCo's multi-CCD flag enabled by default; when ``nativeccd`` is not
            passed, this value controls whether native CCD is enabled.
        memory: Compiler memory budget. Accepts bytes or strings like ``"10M"``.
        solver: Solver enum, int, or string such as ``"Newton"`` or ``"CG"``.
        integrator: Integrator enum, int, or string such as ``"implicitfast"``.
        timestep: Simulation timestep.
        tolerance: Optional solver tolerance.
        iterations: Optional solver iteration limit.
        cone: Cone enum, int, or string such as ``"elliptic"``.
        gravity: XYZ gravity vector.
        sdf_iterations: SDF collision iterations.
        sdf_initpoints: SDF collision initialization points.
        noslip_iterations: No-slip solver iterations.
        enable_multiccd: Enables MuJoCo's multi-CCD flag.
        nativeccd: Enables native CCD when true and disables it when false. If
            omitted, ``multiccd`` is used for compatibility with the old API.
        compiler_angle: Angle unit for Euler values.
        autolimits: Compiler autolimits setting.
        statistic_center: Visualization statistic center.
        statistic_extent: Visualization statistic extent.
        statistic_meansize: Visualization statistic mean size.
        offwidth: Offscreen render width.
        offheight: Offscreen render height.
        add_assets: Add the default skybox, ground texture, and material.
        add_light: Add the default directional light.
        add_floor: Add the default ground plane.
        floor_size: Ground plane size.
        floor_solimp: Ground plane solver impedance.
        option_overrides: Extra attributes to set on ``spec.option``.
        compiler_overrides: Extra attributes to set on ``spec.compiler``.
        visual_overrides: Nested visual overrides, e.g.
            ``{"global_": {"azimuth": 90}}``.
        size_overrides: Extra integer attributes on the spec, such as
            ``{"nconmax": 2000}``.
    """
    spec = mj.MjSpec()
    spec.modelname = sim_name

    spec.compiler.degree = 1 if compiler_angle == "degree" else 0
    spec.compiler.autolimits = autolimits
    _set_attrs(spec.compiler, compiler_overrides, "spec.compiler")

    spec.option.timestep = float(timestep)
    spec.option.integrator = _enum_value(
        integrator, mj.mjtIntegrator, "mjINT", "integrator"
    )
    spec.option.solver = _enum_value(solver, mj.mjtSolver, "mjSOL", "solver")
    spec.option.cone = _enum_value(cone, mj.mjtCone, "mjCONE", "cone")
    spec.option.gravity = _vector(gravity, 3, "gravity")
    spec.option.sdf_iterations = int(sdf_iterations)
    spec.option.sdf_initpoints = int(sdf_initpoints)
    spec.option.noslip_iterations = int(noslip_iterations)
    if tolerance is not None:
        spec.option.tolerance = float(tolerance)
    if iterations is not None:
        spec.option.iterations = int(iterations)

    _set_enable_flag(
        spec.option,
        int(mj.mjtEnableBit.mjENBL_MULTICCD),
        enabled=enable_multiccd,
    )
    nativeccd_enabled = multiccd if nativeccd is None else nativeccd
    _set_disable_flag(
        spec.option,
        int(mj.mjtDisableBit.mjDSBL_NATIVECCD),
        disabled=not nativeccd_enabled,
    )
    _set_attrs(spec.option, option_overrides, "spec.option")

    spec.memory = _parse_memory(memory)
    if size_overrides is not None:
        for key, value in size_overrides.items():
            if not hasattr(spec, key):
                msg = f"MjSpec has no size attribute {key!r}"
                raise AttributeError(msg)
            setattr(spec, key, int(value))

    spec.stat.center = _vector(statistic_center, 3, "statistic_center")
    spec.stat.extent = float(statistic_extent)
    spec.stat.meansize = float(statistic_meansize)

    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.1, 0.1, 0.1]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
    spec.visual.global_.azimuth = 120
    spec.visual.global_.elevation = -20
    spec.visual.global_.offwidth = int(offwidth)
    spec.visual.global_.offheight = int(offheight)
    if visual_overrides is not None:
        for group, attrs in visual_overrides.items():
            target = getattr(spec.visual, group)
            _set_attrs(target, attrs, f"spec.visual.{group}")

    if add_assets:
        spec.add_texture(
            type=mj.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0.0, 0.0, 0.0],
            width=512,
            height=3072,
        )
        spec.add_texture(
            name="groundplane",
            type=mj.mjtTexture.mjTEXTURE_2D,
            builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
            mark=mj.mjtMark.mjMARK_EDGE,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            markrgb=[0.8, 0.8, 0.8],
            width=300,
            height=300,
        )
        spec.add_material(
            name="groundplane",
            textures=["", "groundplane"],
            texuniform=True,
            texrepeat=[5.0, 5.0],
            reflectance=0.2,
        )

    if add_light:
        spec.worldbody.add_light(
            pos=[0.0, 0.0, 1.5],
            dir=[0.0, 0.0, -1.0],
            type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
        )

    if add_floor:
        floor_kwargs = {
            "name": "floor",
            "size": _vector(floor_size, 3, "floor_size"),
            "type": mj.mjtGeom.mjGEOM_PLANE,
            "solimp": _vector(floor_solimp, 5, "floor_solimp"),
        }
        if add_assets:
            floor_kwargs["material"] = "groundplane"
        spec.worldbody.add_geom(**floor_kwargs)

    return _ensure_to_xml_string(spec)


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


def _float_list(
    value: Sequence[float] | float | str | None,
    length: int,
    default: Sequence[float],
) -> list[float]:
    if value is None:
        values = [float(x) for x in default]
    elif isinstance(value, str):
        values = [float(x) for x in value.split()]
    elif isinstance(value, (int, float)):
        values = [float(value)]
    else:
        values = [float(x) for x in value]

    if not values:
        values = [float(x) for x in default]
    if len(values) < length:
        values.extend(float(x) for x in default[len(values) : length])
    if len(values) > length:
        values = values[:length]
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


def _grid_points_elements_2d(
    count: tuple[int, int],
    spacing: tuple[float, float],
) -> tuple[np.ndarray, list[int]]:
    nx, ny = count
    sx, sy = spacing
    points = np.zeros((nx * ny, 3), dtype=float)

    index = 0
    for ix in range(nx):
        for iy in range(ny):
            points[index] = [
                sx * (ix - 0.5 * (nx - 1)),
                sy * (iy - 0.5 * (ny - 1)),
                0.0,
            ]
            index += 1

    elements: list[int] = []
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            v0 = ix * ny + iy
            v1 = (ix + 1) * ny + iy
            v2 = (ix + 1) * ny + (iy + 1)
            v3 = ix * ny + (iy + 1)
            elements.extend([v0, v1, v2, v0, v2, v3])

    return points, elements


def _grid_points_elements_3d(
    count: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, list[int]]:
    nx, ny, nz = count
    sx, sy, sz = spacing
    points = np.zeros((nx * ny * nz, 3), dtype=float)

    index = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                points[index] = [
                    sx * (ix - 0.5 * (nx - 1)),
                    sy * (iy - 0.5 * (ny - 1)),
                    sz * (iz - 0.5 * (nz - 1)),
                ]
                index += 1

    cube_to_tets = [
        [0, 3, 1, 7],
        [0, 1, 4, 7],
        [1, 3, 2, 7],
        [1, 2, 6, 7],
        [1, 5, 4, 7],
        [1, 6, 5, 7],
    ]
    elements: list[int] = []
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            for iz in range(nz - 1):
                vertices = [
                    ix * ny * nz + iy * nz + iz,
                    (ix + 1) * ny * nz + iy * nz + iz,
                    (ix + 1) * ny * nz + (iy + 1) * nz + iz,
                    ix * ny * nz + (iy + 1) * nz + iz,
                    ix * ny * nz + iy * nz + (iz + 1),
                    (ix + 1) * ny * nz + iy * nz + (iz + 1),
                    (ix + 1) * ny * nz + (iy + 1) * nz + (iz + 1),
                    ix * ny * nz + (iy + 1) * nz + (iz + 1),
                ]
                for tet in cube_to_tets:
                    elements.extend([vertices[i] for i in tet])

    return points, elements


def _parse_flex_selfcollide(value: str | int) -> int:
    if isinstance(value, int):
        return value
    key = value.strip().lower()
    mapping = {
        "none": getattr(mj.mjtFlexSelf, "mjFLEXSELF_NONE", 0),
        "self": getattr(mj.mjtFlexSelf, "mjFLEXSELF_NARROW", 1),
        "all": getattr(mj.mjtFlexSelf, "mjFLEXSELF_BVH", 2),
    }
    if key not in mapping:
        msg = f"Unknown flex selfcollide setting: {value}"
        raise ValueError(msg)
    return mapping[key]


def _parse_flex_elastic2d(value: str | int) -> int:
    if isinstance(value, int):
        return value
    key = value.strip().lower()
    mapping = {
        "none": 0,
        "bend": 1,
        "stretch": 2,
        "both": 3,
    }
    if key not in mapping:
        msg = f"Unknown flex elastic2d setting: {value}"
        raise ValueError(msg)
    return mapping[key]


def _flex_grid_spec(
    *,
    model_name: str,
    name: str,
    count: Sequence[int],
    spacing: Sequence[float],
    dim: int,
    mass: float,
    rgba: Sequence[float],
    radius: float,
    joint_damping: float,
    inertiabox: float,
    friction: Sequence[float] | str | None,
    solref: Sequence[float] | str | None,
    solimp: Sequence[float] | str | None,
    condim: int | None,
    margin: float | None,
    gap: float | None,
    selfcollide: str | int | None,
    edge_equality: bool,
    edge_damping: float | None,
    edge_stiffness: float | None,
    damping: float | None,
    young: float | None,
    poisson: float | None,
    thickness: float | None,
    elastic2d: str | int | None,
) -> mj.MjSpec:
    if dim not in (2, 3):
        msg = f"dim must be 2 or 3, got {dim}"
        raise ValueError(msg)

    count_vec = [int(x) for x in count]
    if len(count_vec) != 3:
        msg = f"count must contain 3 integers, got {count}"
        raise ValueError(msg)
    if dim == 2:
        count_vec[2] = 1
        if count_vec[0] < 2 or count_vec[1] < 2:
            msg = f"2D flex objects require at least 2x2 points, got {count_vec}"
            raise ValueError(msg)
        points, elements = _grid_points_elements_2d(
            (count_vec[0], count_vec[1]),
            (float(spacing[0]), float(spacing[1])),
        )
    else:
        if any(value < 2 for value in count_vec):
            msg = f"3D flex objects require at least 2x2x2 points, got {count_vec}"
            raise ValueError(msg)
        points, elements = _grid_points_elements_3d(
            (count_vec[0], count_vec[1], count_vec[2]),
            (float(spacing[0]), float(spacing[1]), float(spacing[2])),
        )

    if not elements:
        msg = "No flex elements were created"
        raise ValueError(msg)

    spec = mj.MjSpec()
    spec.modelname = _safe_mj_name(model_name)
    flex_name = _safe_mj_name(name)

    n_points = points.shape[0]
    body_mass = mass / n_points
    body_inertia = body_mass * (2.0 * inertiabox * inertiabox) / 3.0
    vertbody: list[str] = []

    for i, point in enumerate(points):
        body = spec.worldbody.add_body(name=f"{flex_name}_{i}", pos=point.tolist())
        body.ipos = np.array([0.0, 0.0, 0.0])
        body.mass = body_mass
        body.inertia = np.array([body_inertia, body_inertia, body_inertia])
        body.explicitinertial = 1

        for axis_index in range(3):
            axis = [0.0, 0.0, 0.0]
            axis[axis_index] = 1.0
            joint = body.add_joint(
                type=mj.mjtJoint.mjJNT_SLIDE,
                pos=[0.0, 0.0, 0.0],
                axis=axis,
            )
            joint.damping = joint_damping

        vertbody.append(body.name)

    flex = spec.add_flex(name=flex_name)
    flex.dim = dim
    flex.vert = np.zeros_like(points).reshape(-1).tolist()
    flex.elem = elements
    flex.vertbody = vertbody
    flex.rgba = np.array(rgba, dtype=np.float32)
    flex.radius = float(radius)

    if condim is not None:
        flex.condim = int(condim)
    if friction is not None:
        flex.friction = np.array(
            _float_list(friction, 3, (0.3, 0.3, 0.3)), dtype=float
        ).reshape(3, 1)
    if solref is not None:
        flex.solref = np.array(
            _float_list(solref, 2, (0.02, 1.0)), dtype=float
        ).reshape(2, 1)
    if solimp is not None:
        flex.solimp = np.array(
            _float_list(solimp, 5, (0.95, 0.99, 0.001, 0.5, 2.0)), dtype=float
        ).reshape(5, 1)
    if margin is not None:
        flex.margin = float(margin)
    if gap is not None:
        flex.gap = float(gap)
    if selfcollide is not None:
        flex.selfcollide = _parse_flex_selfcollide(selfcollide)

    for attr in ("edgeequality", "edge_equality"):
        try:
            setattr(flex, attr, int(bool(edge_equality)))
            break
        except AttributeError:
            continue

    if edge_damping is not None:
        flex.edgedamping = float(edge_damping)
    if dim == 1 and edge_stiffness is not None:
        flex.edgestiffness = float(edge_stiffness)
    if damping is not None:
        flex.damping = float(damping)
    if young is not None:
        flex.young = float(young)
    if poisson is not None:
        flex.poisson = float(poisson)
    if thickness is not None:
        flex.thickness = float(thickness)
    if elastic2d is not None:
        flex.elastic2d = _parse_flex_elastic2d(elastic2d)
    if dim == 3:
        flex.flatskin = 1

    spec.add_equality(
        type=mj.mjtEq.mjEQ_FLEX,
        objtype=mj.mjtObj.mjOBJ_FLEX,
        name1=flex_name,
    )
    return _ensure_to_xml_string(spec)


def _warn_if_mesh_likely_millimeters(
    source_path: Path,
    triangle_mesh,
    scale: Sequence[float],
    threshold: float,
    function_call: str = "mjsim.mesh",
) -> None:
    extents = triangle_mesh.get_max_bound() - triangle_mesh.get_min_bound()
    scaled_extents = [float(extents[i]) * abs(scale[i]) for i in range(3)]
    max_scaled_extent = max(scaled_extents)
    if max_scaled_extent < threshold:
        return

    unscaled = " ".join(f"{float(value):.3g}" for value in extents)
    scaled = " ".join(f"{value:.3g}" for value in scaled_extents)
    print(
        "\033[33m"
        "Warning: "
        f"mesh '{source_path}' has extents [{unscaled}] before scaling "
        f"and [{scaled}] after scaling. This looks like a mesh exported in "
        "millimeters while MuJoCo expects meters. Fix: pass scale=0.001 "
        f"to {function_call}(...)."
        "\033[0m"
    )


def _xml_bool(value: bool | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


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
    warn_if_millimeters: bool = True,
    millimeter_warning_threshold: float = 10.0,
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
        warn_if_millimeters: Warn when scaled mesh extents look like millimeters.
        millimeter_warning_threshold: Largest scaled extent, in meters, above
            which the mesh is considered suspicious.

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
    scale_xyz = _vector(scale, 3, "scale")
    if warn_if_millimeters:
        _warn_if_mesh_likely_millimeters(
            source_path=source_path,
            triangle_mesh=visual_mesh,
            scale=scale_xyz,
            threshold=millimeter_warning_threshold,
        )

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

        collision_mesh.triangle_normals = o3d.utility.Vector3dVector()
        if not o3d.io.write_triangle_mesh(
            str(cache_path),
            collision_mesh,
            write_ascii=True,
            write_vertex_normals=False,
            write_vertex_colors=False,
            write_triangle_uvs=False,
        ):
            msg = f"Failed to write collision mesh cache: {cache_path}"
            raise OSError(msg)
        print(f"Saved decimated collision mesh to: {cache_path}")

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


def deformable_mesh(
    path: str | Path,
    model_name: str | None = None,
    name: str | None = None,
    scale: Sequence[float] | float = 1.0,
    pos: Sequence[float] | None = None,
    euler: Sequence[float] | None = None,
    quat: Sequence[float] | None = None,
    dim: int = 2,
    dof: str = "quadratic",
    mass: float = 0.05,
    radius: float = 0.001,
    rgba: Sequence[float] = (0.0, 0.7, 0.7, 1.0),
    young: float = 10.0,
    poisson: float = 0.1,
    damping: float = 0.001,
    elastic2d: str | None = "stretch",
    selfcollide: str = "none",
    internal: bool | str = False,
    condim: int | None = None,
    solref: Sequence[float] | str | None = None,
    solimp: Sequence[float] | str | None = None,
    warn_if_millimeters: bool = True,
    millimeter_warning_threshold: float = 10.0,
) -> mj.MjSpec:
    """
    Create a deformable mesh using MuJoCo's ``flexcomp type="mesh"`` compiler.

    This is useful for OBJ surface meshes such as the Stanford bunny. The default
    parameters favor a soft, stable surface mesh that can be stepped directly
    with the default ``empty_scene`` timestep.
    """
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        msg = f"Mesh file does not exist: {source_path}"
        raise FileNotFoundError(msg)
    if euler is not None and quat is not None:
        msg = "Pass either euler or quat for the flexcomp rotation, not both"
        raise ValueError(msg)
    if dim not in (2, 3):
        msg = f"dim must be 2 or 3, got {dim}"
        raise ValueError(msg)
    if mass <= 0:
        msg = f"mass must be positive, got {mass}"
        raise ValueError(msg)
    if radius < 0:
        msg = f"radius must be non-negative, got {radius}"
        raise ValueError(msg)
    if young <= 0:
        msg = f"young must be positive, got {young}"
        raise ValueError(msg)
    if damping < 0:
        msg = f"damping must be non-negative, got {damping}"
        raise ValueError(msg)

    mesh_name = _safe_mj_name(name or model_name or source_path.stem)
    model_name = _safe_mj_name(model_name or mesh_name)
    scale_xyz = _vector(scale, 3, "scale")

    if warn_if_millimeters:
        _warn_if_mesh_likely_millimeters(
            source_path=source_path,
            triangle_mesh=_clean_triangle_mesh(_load_triangle_mesh(source_path)),
            scale=scale_xyz,
            threshold=millimeter_warning_threshold,
            function_call="mjsim.deformable_mesh",
        )

    mujoco = ET.Element("mujoco", model=model_name)
    ET.SubElement(mujoco, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        mujoco,
        "option",
        solver="CG",
        tolerance="1e-6",
        timestep="0.001",
        integrator="implicitfast",
    )
    ET.SubElement(mujoco, "size", memory="100M")

    worldbody = ET.SubElement(mujoco, "worldbody")
    flex_attrs = {
        "type": "mesh",
        "file": str(source_path.resolve()),
        "name": mesh_name,
        "dim": str(dim),
        "dof": str(dof),
        "mass": str(mass),
        "radius": str(radius),
        "rgba": " ".join(str(x) for x in _float_list(rgba, 4, (0.0, 0.7, 0.7, 1.0))),
        "scale": " ".join(str(x) for x in scale_xyz),
    }

    flex_pos = _optional_vector(pos, 3, "pos")
    flex_euler = _optional_vector(euler, 3, "euler")
    flex_quat = _optional_vector(quat, 4, "quat")
    if flex_pos is not None:
        flex_attrs["pos"] = " ".join(str(x) for x in flex_pos)
    if flex_euler is not None:
        flex_attrs["euler"] = " ".join(str(x) for x in flex_euler)
    if flex_quat is not None:
        flex_attrs["quat"] = " ".join(str(x) for x in flex_quat)

    flexcomp = ET.SubElement(worldbody, "flexcomp", **flex_attrs)

    elasticity_attrs = {
        "young": str(young),
        "poisson": str(poisson),
        "damping": str(damping),
    }
    if elastic2d is not None:
        elasticity_attrs["elastic2d"] = str(elastic2d)
    ET.SubElement(flexcomp, "elasticity", **elasticity_attrs)

    contact_attrs = {
        "selfcollide": str(selfcollide),
        "internal": _xml_bool(internal),
    }
    if condim is not None:
        contact_attrs["condim"] = str(condim)
    if solref is not None:
        contact_attrs["solref"] = " ".join(
            str(x) for x in _float_list(solref, 2, (0.01, 1.0))
        )
    if solimp is not None:
        contact_attrs["solimp"] = " ".join(
            str(x) for x in _float_list(solimp, 3, (0.95, 0.99, 0.0001))
        )
    ET.SubElement(flexcomp, "contact", **contact_attrs)

    try:
        ET.indent(mujoco)
        xml_str = ET.tostring(mujoco, encoding="unicode", method="xml")
    except AttributeError:
        xml_str = minidom.parseString(
            ET.tostring(mujoco, encoding="unicode")
        ).toprettyxml(indent="  ")

    return _spec_from_string(xml_str)


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
    # Separate geom, joint, and flex arguments from kwargs
    geom_args = {}
    joint_args = {}
    flex_args = {}

    for key, value in kwargs.items():
        if key.startswith("geom_"):
            geom_args[key[5:]] = value  # Remove 'geom_' prefix
        elif key.startswith("joint_"):
            joint_args[key[6:]] = value  # Remove 'joint_' prefix
        else:
            flex_args[key] = value

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

    dim = 3 if count[2] > 1 else 2
    spacing = flex_args.pop("spacing", None)
    if spacing is None:
        spacing = [
            size / max(count[0] - 1, 1),
            size / max(count[1] - 1, 1),
            size / max(count[2] - 1, 1) if dim == 3 else 0.0,
        ]
    else:
        spacing = _float_list(spacing, 3, (size, size, size))

    per_point_mass = float(geom_args.get("mass", 0.01))
    total_mass = float(flex_args.pop("mass", per_point_mass * np.prod(count)))
    rgba = _float_list(
        geom_args.get("rgba", [0.3, 0.5, 0.8, 0.7]),
        4,
        (0.3, 0.5, 0.8, 0.7),
    )

    return _flex_grid_spec(
        model_name=model_name,
        name=model_name if prefix is None else prefix.rstrip(":/"),
        count=count,
        spacing=spacing,
        dim=dim,
        mass=total_mass,
        rgba=rgba,
        radius=float(flex_args.pop("radius", segment_size)),
        joint_damping=float(joint_args.get("damping", 0.05)),
        inertiabox=float(flex_args.pop("inertiabox", max(segment_size, 0.001))),
        friction=geom_args.get("friction", "1.0 0.3 0.1"),
        solref=geom_args.get("solref", "0.0001 0.9"),
        solimp=geom_args.get("solimp", None),
        condim=int(geom_args.get("condim", 4)),
        margin=flex_args.pop("margin", None),
        gap=flex_args.pop("gap", None),
        selfcollide=flex_args.pop("selfcollide", "none"),
        edge_equality=bool(flex_args.pop("edge_equality", True)),
        edge_damping=flex_args.pop("edge_damping", 0.001),
        edge_stiffness=flex_args.pop("edge_stiffness", None),
        damping=flex_args.pop("damping", None),
        young=flex_args.pop("young", max(stretch, shear)),
        poisson=flex_args.pop("poisson", 0.2),
        thickness=flex_args.pop("thickness", segment_size if dim == 2 else None),
        elastic2d=flex_args.pop("elastic2d", "both" if dim == 2 else None),
    )


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
    width_segments: int = 5,
    height_segments: int = 5,
    width: float = 0.2,
    height: float = 0.2,
    stretch: float = 3.0e8,
    shear: float = 3.0e8,
    thickness: float = 0.1,
    mass: float = 1.0,
    pin_corner: bool = True,
    **kwargs,
) -> mj.MjSpec:
    """
    Create a cloth model using MuJoCo's grid flexcomp compiler.
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

    prefix = f"{model_name}:" if prefix is None else prefix
    cloth_name = _safe_mj_name(prefix.rstrip(":/"))
    spacing = kwargs.pop(
        "spacing",
        [
            width / max(width_segments - 1, 1),
            height / max(height_segments - 1, 1),
            max(
                width / max(width_segments - 1, 1),
                height / max(height_segments - 1, 1),
            ),
        ],
    )
    spacing = _float_list(spacing, 3, (0.05, 0.05, 0.05))

    radius = float(kwargs.pop("radius", 0.001))
    edge_equality = "true" if kwargs.pop("edge_equality", True) else "false"
    edge_damping = float(kwargs.pop("edge_damping", 0.01))
    edge_solref = " ".join(
        str(x)
        for x in _float_list(kwargs.pop("edge_solref", "0.00001 1"), 2, (1e-5, 1))
    )
    young = float(kwargs.pop("young", max(stretch, shear)))
    poisson = float(kwargs.pop("poisson", 0.1))
    elastic2d = kwargs.pop("elastic2d", "none")

    if kwargs:
        print(f"Warning: unused cloth kwargs: {list(kwargs.keys())}")

    mujoco = ET.Element("mujoco", model=model_name)
    worldbody = ET.SubElement(mujoco, "worldbody")
    body = ET.SubElement(worldbody, "body", name=f"{cloth_name}_pin", pos="0 0 0")
    flexcomp = ET.SubElement(
        body,
        "flexcomp",
        type="grid",
        count=f"{width_segments} {height_segments} 1",
        spacing=" ".join(str(x) for x in spacing),
        mass=str(mass),
        name=cloth_name,
        radius=str(radius),
    )
    ET.SubElement(
        flexcomp,
        "edge",
        equality=edge_equality,
        damping=str(edge_damping),
        solref=edge_solref,
    )
    ET.SubElement(
        flexcomp,
        "elasticity",
        poisson=str(poisson),
        thickness=str(thickness),
        young=str(young),
        elastic2d=str(elastic2d),
    )

    if pin_corner:
        equality = ET.SubElement(mujoco, "equality")
        ET.SubElement(equality, "connect", body1=f"{cloth_name}_0", anchor="0 0 0")

    try:
        ET.indent(mujoco)
        xml_str = ET.tostring(mujoco, encoding="unicode", method="xml")
    except AttributeError:
        xml_str = minidom.parseString(
            ET.tostring(mujoco, encoding="unicode")
        ).toprettyxml(indent="  ")

    return _spec_from_string(xml_str)


def jello(
    model_name: str = "jello",
    prefix: Optional[str] = None,
    count: Sequence[int] = (4, 4, 4),
    spacing: Sequence[float] | float = (0.1, 0.1, 0.1),
    mass: float = 25.0,
    radius: float = 0.0,
    rgba: Sequence[float] = (0.0, 0.7, 0.7, 1.0),
    condim: int = 3,
    solref: Sequence[float] | str = (0.01, 1.0),
    solimp: Sequence[float] | str = (0.95, 0.99, 0.0001),
    selfcollide: str = "none",
    young: float = 5.0e4,
    damping: float = 0.002,
    poisson: float = 0.2,
) -> mj.MjSpec:
    """
    Create a 3D jello-like deformable object using MuJoCo's grid flexcomp compiler.
    """
    count_vec = [int(x) for x in count]
    if len(count_vec) != 3:
        msg = f"count must contain 3 integers, got {count}"
        raise ValueError(msg)
    if any(x < 2 for x in count_vec):
        msg = f"jello requires at least a 2x2x2 grid, got {count_vec}"
        raise ValueError(msg)
    assert mass > 0, f"mass should be above 0, got {mass=}"

    if isinstance(spacing, (int, float)):
        spacing_vec = [float(spacing)] * 3
    else:
        spacing_vec = _float_list(spacing, 3, (0.1, 0.1, 0.1))

    prefix = f"{model_name}:" if prefix is None else prefix
    jello_name = _safe_mj_name(prefix.rstrip(":/"))

    mujoco = ET.Element("mujoco", model=model_name)
    worldbody = ET.SubElement(mujoco, "worldbody")
    flexcomp = ET.SubElement(
        worldbody,
        "flexcomp",
        type="grid",
        count=" ".join(str(x) for x in count_vec),
        spacing=" ".join(str(x) for x in spacing_vec),
        mass=str(mass),
        name=jello_name,
        radius=str(radius),
        rgba=" ".join(str(x) for x in _float_list(rgba, 4, (0.0, 0.7, 0.7, 1.0))),
        dim="3",
    )
    ET.SubElement(
        flexcomp,
        "contact",
        condim=str(condim),
        solref=" ".join(str(x) for x in _float_list(solref, 2, (0.01, 1.0))),
        solimp=" ".join(str(x) for x in _float_list(solimp, 3, (0.95, 0.99, 0.0001))),
        selfcollide=str(selfcollide),
    )
    ET.SubElement(
        flexcomp,
        "elasticity",
        young=str(young),
        damping=str(damping),
        poisson=str(poisson),
    )

    try:
        ET.indent(mujoco)
        xml_str = ET.tostring(mujoco, encoding="unicode", method="xml")
    except AttributeError:
        xml_str = minidom.parseString(
            ET.tostring(mujoco, encoding="unicode")
        ).toprettyxml(indent="  ")

    return _spec_from_string(xml_str)


def deform_3d(
    model_name: str = "soft_block",
    prefix: str | None = None,
    resolution: int = 5,
    size: float = 0.25,
    mass: float = 0.2,
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
        mass=mass,
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
