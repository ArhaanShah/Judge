from __future__ import annotations

import json

import pytest

from src.run_judge import _increment_counter, provider_preflight


def test_persistent_counter_aborts_before_call_801(tmp_path) -> None:
    path = tmp_path / "counter.json"
    path.write_text(json.dumps({"api_calls": 799}), encoding="utf-8")
    assert _increment_counter(path, 800) == 800
    with pytest.raises(RuntimeError, match="cap"):
        _increment_counter(path, 800)


def test_preflight_requires_operator_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "test-only")
    config = {"judge": {"provider": "cohere", "model": "command-a-plus-05-2026"}, "budget": {"required_available_calls": 800, "hard_api_call_cap": 800}}
    with pytest.raises(RuntimeError, match="confirm"):
        provider_preflight(config, trial_confirmed=False, available_call_budget=800)
