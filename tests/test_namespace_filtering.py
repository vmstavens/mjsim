import pytest

pytest.importorskip("mujoco")

from mjsim.utils.mj import is_robot_entity


@pytest.mark.parametrize(
    ("entity_name", "namespace", "expected"),
    [
        ("ur10e/base_link", "ur10e", True),
        ("ur10e/gripper/base_link", "ur10e", False),
        ("/ur10e/gripper/base_link", "/ur10e/", False),
        ("ur10e", "ur10e", False),
        ("prefix/sub/object", "prefix/sub", True),
        ("prefix/sub/child/leaf", "prefix/sub", False),
        ("Prefix/Sub/Object", "prefix/sub", True),
    ],
)
def test_is_robot_entity_filters_nested_namespaces(entity_name, namespace, expected):
    assert is_robot_entity(entity_name, namespace) is expected
