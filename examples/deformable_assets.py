"""
Generate deformable MuJoCo assets and export them as XML strings.
"""

from pathlib import Path

from mjsim.utils.mjs import cable, cloth, deform_3d


def main() -> None:
    cable_spec = cable(model_name="demo_cable", n_segments=8, length=1.0)
    cloth_spec = cloth(model_name="demo_cloth", width_segments=6, height_segments=4)
    block_spec = deform_3d(model_name="demo_block", resolution=4, size=0.15)

    export_root = Path("generated_assets")
    export_root.mkdir(exist_ok=True)
    (export_root / "demo_cable.xml").write_text(cable_spec.to_xml_string())
    (export_root / "demo_cloth.xml").write_text(cloth_spec.to_xml_string())
    (export_root / "demo_block.xml").write_text(block_spec.to_xml_string())
    print(f"Exported XML to {export_root}")


if __name__ == "__main__":
    main()
