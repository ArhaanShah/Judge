from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.providers.base import ProviderError, RetriableProviderError
from src.providers.groq_provider import GroqJudgeProvider, RESPONSE_FORMAT
from src.run_judge import provider_preflight


class FakeCompletions:
    def __init__(self, response=None, exc=None) -> None:
        self.kwargs = None
        self.response = response or SimpleNamespace(
            id="groq-req-1",
            model="llama-3.3-70b-versatile",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"winner":"A","brief_reason":"A is correct."}'
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
            model_dump=lambda: {"id": "groq-req-1"},
        )
        self.exc = exc

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.response


class FakeClient:
    def __init__(self, completions=None) -> None:
        self.chat = SimpleNamespace(completions=completions or FakeCompletions())


def test_groq_provider_parameters_and_parsing() -> None:
    completions = FakeCompletions()
    provider = GroqJudgeProvider(client=FakeClient(completions))

    response = provider.judge(
        "prompt",
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_output_tokens=256,
    )

    assert completions.kwargs["model"] == "llama-3.3-70b-versatile"
    assert completions.kwargs["messages"] == [{"role": "user", "content": "prompt"}]
    assert completions.kwargs["response_format"] == RESPONSE_FORMAT
    assert completions.kwargs["temperature"] == 0
    assert completions.kwargs["max_tokens"] == 256
    assert response.provider == "groq"
    assert response.returned_model == "llama-3.3-70b-versatile"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 8
    assert response.brief_reason == "A is correct."


def test_groq_provider_rejects_max_tokens() -> None:
    provider = GroqJudgeProvider(
        client=FakeClient(
            FakeCompletions(
                response=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="length",
                            message=SimpleNamespace(content='{"winner":"A"}'),
                        )
                    ]
                )
            )
        )
    )

    with pytest.raises(ProviderError, match="incomplete Groq response: length"):
        provider.judge(
            "prompt",
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_output_tokens=256,
        )


def test_groq_provider_retries_on_429() -> None:
    exc = RuntimeError("rate limit")
    setattr(exc, "status_code", 429)
    provider = GroqJudgeProvider(client=FakeClient(FakeCompletions(exc=exc)))

    with pytest.raises(RetriableProviderError):
        provider.judge(
            "prompt",
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_output_tokens=256,
        )


def test_groq_preflight_checks(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    config = {
        "judge": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "budget": {"required_available_calls": 480, "hard_api_call_cap": 500},
    }
    provider_preflight(config, trial_confirmed=True, available_call_budget=500)

    with pytest.raises(RuntimeError, match="confirm"):
        provider_preflight(config, trial_confirmed=False, available_call_budget=500)
