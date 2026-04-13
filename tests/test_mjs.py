import os

import pytest

if os.environ.get("MJSIM_LIGHT_IMPORT") == "1":
    pytest.skip(
        "Skipping MuJoCo-dependent tests in light mode", allow_module_level=True
    )


def test_cable_xml_roundtrip():
    mj = pytest.importorskip("mujoco")

    from mjsim.utils.mjs import cable, deform_3d_custom

    spec = cable(model_name="unit_test_cable", n_segments=4, length=0.4)
    assert isinstance(spec, mj.MjSpec) or hasattr(spec, "_xml_string")

    if hasattr(spec, "to_xml_string"):
        xml = spec.to_xml_string()
    else:
        xml = getattr(spec, "_xml_string")

    assert "unit_test_cable" in xml
    assert "<composite" in xml

    try:
        block = deform_3d_custom(
            count=[2, 2, 2], size=0.1, stretch=100.0, bend=10.0, shear=50.0
        )
    except ValueError as exc:
        # Flex composites require MuJoCo builds with flex support.
        if "invalid keyword: 'flex'" in str(exc):
            pytest.skip("MuJoCo build lacks flex composite support")
        raise
    assert "composite" in block.to_xml_string()


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
