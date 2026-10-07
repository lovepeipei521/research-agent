from pathlib import Path


def test_skill_manager_can_start_with_builtin_skills_only(monkeypatch, tmp_path: Path) -> None:
    from poirot.backend.agents.skill import build_skill_manager

    monkeypatch.setenv("POIROT_SKILL_ENABLED", "true")
    monkeypatch.setenv("POIROT_SKILL_INCLUDE_BUILTIN", "true")
    monkeypatch.setenv("POIROT_SKILL_DIRS", str(tmp_path / "missing-user-skills"))
    monkeypatch.setenv("POIROT_SKILL_DB_PATH", str(tmp_path / "skills.db"))

    manager = build_skill_manager()

    assert manager is not None
    manager.load_startup()
    assert manager.list_skills()
