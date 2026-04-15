import math
import os

import pytest

if os.environ.get("MJSIM_LIGHT_IMPORT") == "1":
    pytest.skip(
        "Skipping MuJoCo-dependent tests in light mode", allow_module_level=True
    )


def test_cable_xml_roundtrip():
    mj = pytest.importorskip("mujoco")

    from mjsim.utils.mjs import cable

    spec = cable(model_name="unit_test_cable", n_segments=4, length=0.4)
    assert isinstance(spec, mj.MjSpec) or hasattr(spec, "_xml_string")

    if hasattr(spec, "to_xml_string"):
        xml = spec.to_xml_string()
    else:
        xml = getattr(spec, "_xml_string")

    assert "unit_test_cable" in xml
    assert "mujoco.elasticity.cable" in xml


def test_flex_deformables_compile_and_export_xml():
    mj = pytest.importorskip("mujoco")

    from mjsim.utils.mjs import cloth, deform_3d_custom, jello

    cloth_spec = cloth(width_segments=4, height_segments=4)
    cloth_model = cloth_spec.compile()
    assert cloth_model.nflex == 1
    assert cloth_model.flex_edgeequality[0] == 1
    assert cloth_model.flex_edgedamping[0] > 0
    assert "<flex" in cloth_spec.to_xml_string()

    scene = mj.MjSpec()
    scene.attach(cloth_spec, prefix="", frame=scene.worldbody.add_frame())
    attached_model = scene.compile()
    assert attached_model.nflex == 1
    assert (attached_model.eq_type[: attached_model.neq] == mj.mjtEq.mjEQ_FLEX).any()

    block = deform_3d_custom(
        count=[2, 2, 2], size=0.1, stretch=100.0, bend=10.0, shear=50.0
    )
    block_model = block.compile()
    assert block_model.nflex == 1
    assert "<flex" in block.to_xml_string()

    jello_spec = jello(count=(3, 3, 3), spacing=(0.03, 0.03, 0.03))
    jello_model = jello_spec.compile()
    assert jello_model.nflex == 1
    assert jello_model.flex_dim[0] == 3
    assert "<flex" in jello_spec.to_xml_string()


def test_empty_scene_uses_spec_options():
    mj = pytest.importorskip("mujoco")

    from mjsim.utils.mjs import empty_scene

    spec = empty_scene(
        memory="100M",
        solver="CG",
        timestep=0.001,
        tolerance=1e-6,
        nativeccd=True,
        option_overrides={"iterations": 50},
        size_overrides={"nconmax": 2000},
    )
    model = spec.compile()

    assert spec.memory == 100 * 1024 * 1024
    assert spec.nconmax == 2000
    assert model.opt.solver == mj.mjtSolver.mjSOL_CG
    assert model.opt.timestep == pytest.approx(0.001)
    assert model.opt.tolerance == pytest.approx(1e-6)
    assert model.opt.iterations == 50
    assert model.opt.disableflags & mj.mjtDisableBit.mjDSBL_NATIVECCD == 0
    assert model.ngeom == 1
    assert model.nlight == 1

    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    material_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, "groundplane")
    texture_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_TEXTURE, "groundplane")
    assert floor_id != -1
    assert material_id != -1
    assert texture_id != -1
    assert model.geom_matid[floor_id] == material_id
    assert model.mat_texid[material_id, 1] == texture_id
    assert 'texture="groundplane"' in spec.to_xml_string()


def test_mesh_helper_writes_collision_cache_and_compiles(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("open3d")

    from mjsim.utils.mjs import mesh

    mesh_path = tmp_path / "tetra.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 3 2",
                "f 1 2 4",
                "f 2 3 4",
                "f 3 1 4",
            ]
        ),
        encoding="utf-8",
    )

    spec = mesh(
        mesh_path,
        model_name="tetra_model",
        name="tetra",
        decimation_ratio=1.0,
        cache_dir=tmp_path / ".cache",
        freejoint=True,
    )

    cached_meshes = list((tmp_path / ".cache").glob("tetra_collision_*.obj"))
    assert len(cached_meshes) == 1

    model = spec.compile()
    assert model.nmesh == 2
    assert model.ngeom == 2
    assert model.njnt == 1


def test_mesh_helper_warns_when_mesh_looks_like_millimeters(tmp_path, capsys):
    pytest.importorskip("mujoco")
    pytest.importorskip("open3d")

    from mjsim.utils.mjs import mesh

    mesh_path = tmp_path / "large_part.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 20 0 0",
                "v 0 20 0",
                "v 0 0 20",
                "f 1 3 2",
                "f 1 2 4",
                "f 2 3 4",
                "f 3 1 4",
            ]
        ),
        encoding="utf-8",
    )

    mesh(mesh_path, decimation_ratio=1.0, cache_dir=tmp_path / ".cache")
    captured = capsys.readouterr()
    assert "looks like a mesh exported in millimeters" in captured.out
    assert "scale=0.001" in captured.out

    mesh(
        mesh_path,
        scale=0.001,
        decimation_ratio=1.0,
        cache_dir=tmp_path / ".cache",
        overwrite_cache=True,
    )
    captured = capsys.readouterr()
    assert "looks like a mesh exported in millimeters" not in captured.out


def test_deformable_mesh_helper_compiles_and_attaches(tmp_path):
    pytest.importorskip("mujoco")
    o3d = pytest.importorskip("open3d")

    from mjsim.utils.mjs import deformable_mesh, empty_scene

    mesh_path = tmp_path / "soft_sphere.obj"
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=5)
    sphere.triangle_normals = o3d.utility.Vector3dVector()
    o3d.io.write_triangle_mesh(
        str(mesh_path),
        sphere,
        write_ascii=True,
        write_vertex_normals=False,
        write_vertex_colors=False,
        write_triangle_uvs=False,
    )

    spec = deformable_mesh(mesh_path, name="soft_sphere")
    model = spec.compile()
    assert model.nflex == 1
    assert model.flex_dim[0] == 2
    assert "<flex" in spec.to_xml_string()
    assert "soft_sphere" in spec.to_xml_string()

    scene = empty_scene()
    scene.attach(spec, prefix="", frame=scene.worldbody.add_frame())
    attached_model = scene.compile()
    attached_data = mj.MjData(attached_model)
    for _ in range(5):
        mj.mj_step(attached_model, attached_data)

    assert attached_model.nflex == 1
    assert all(math.isfinite(float(value)) for value in attached_data.qpos)
