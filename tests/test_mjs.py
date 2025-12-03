import pytest


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
import os
import pytest

if os.environ.get("MJSIM_LIGHT_IMPORT") == "1":
    pytest.skip("Skipping MuJoCo-dependent tests in light mode", allow_module_level=True)
