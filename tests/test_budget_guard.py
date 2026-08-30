from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.attempt_budget import AttemptBudget
from src.run_judge import provider_preflight


def test_persistent_counter_aborts_before_call_801(tmp_path) -> None:
    path = tmp_path / "counter.json"
    budget = AttemptBudget(path, 800, "test_model", "test_hash")
    path.write_text(json.dumps({"api_attempts": 799}), encoding="utf-8")
    assert budget.increment() == 800
    with pytest.raises(RuntimeError, match="cap"):
        budget.increment()


def test_preflight_requires_operator_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "test-only")
    config = {"judge": {"provider": "cohere", "model": "command-a-plus-05-2026"}, "budget": {"required_available_calls": 800, "hard_api_call_cap": 800}}
    with pytest.raises(RuntimeError, match="confirm"):
        provider_preflight(config, trial_confirmed=False, available_call_budget=800)
