from __future__ import annotations

import pytest


@pytest.fixture
def stubgen(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MJSIM_SKIP_STUBGEN", "1")
    import mjsim._stubgen

    return mjsim._stubgen


def test_target_modules_includes_importable_defaults(
    stubgen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    importable = {"mujoco", "ompl"}

    def fake_find_spec(module: str):
        return object() if module in importable else None

    monkeypatch.setattr(stubgen.util, "find_spec", fake_find_spec)

    assert list(stubgen._target_modules()) == ["mujoco", "ompl"]


def test_ensure_stubs_generates_targets_missing_from_stamp(
    stubgen,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    generated_targets: list[list[str]] = []
    stamp = tmp_path / ".stamp"
    stamp.write_text("mujoco")

    def fake_run_stubgen(stub_root, targets, *, quiet: bool) -> bool:
        generated_targets.append(list(targets))
        return True

    monkeypatch.delenv("MJSIM_SKIP_STUBGEN", raising=False)
    monkeypatch.setattr(stubgen, "_stub_root", lambda: tmp_path)
    monkeypatch.setattr(stubgen, "_target_modules", lambda: ["mujoco", "ompl"])
    monkeypatch.setattr(stubgen, "_run_stubgen", fake_run_stubgen)

    stubgen.ensure_stubs()

    assert generated_targets == [["ompl"]]
    assert stamp.read_text() == "mujoco\nompl"
