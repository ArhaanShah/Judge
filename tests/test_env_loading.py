from __future__ import annotations

from src.utils import load_env_file


def test_env_file_loads_without_overwriting(monkeypatch, tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text("NEW_VALUE='from-file'\nEXISTING=from-file\n", encoding="utf-8")
    monkeypatch.delenv("NEW_VALUE", raising=False)
    monkeypatch.setenv("EXISTING", "from-process")
    load_env_file(path)
    assert __import__("os").environ["NEW_VALUE"] == "from-file"
    assert __import__("os").environ["EXISTING"] == "from-process"
