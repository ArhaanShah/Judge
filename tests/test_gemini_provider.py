from __future__ import annotations

from types import SimpleNamespace
import pytest

from src.providers.base import ProviderError, RetriableProviderError
from src.providers.gemini_provider import (
    GeminiJudgeProvider,
    RESPONSE_SCHEMA,
)
from src.run_judge import provider_preflight


class FakeModels:
    def __init__(self, response=None, exc=None) -> None:
        self.kwargs = None
        self.exc = exc
        self.response = response or SimpleNamespace(
            response_id="gemini-req-1",
            model_version="gemini-3.7-flash",
            text='{"winner":"B","brief_reason":"B is mathematically sound."}',
            candidates=[
                SimpleNamespace(
                    finish_reason="STOP",
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text='{"winner":"B","brief_reason":"B is mathematically sound."}'
                            )
                        ]
                    ),
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=15,
                candidates_token_count=9,
            ),
            model_dump=lambda: {"id": "gemini-req-1"},
        )

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.response


class FakeGenAIClient:
    def __init__(self, models=None) -> None:
        self.models = models or FakeModels()


def test_gemini_provider_parameters_and_parsing() -> None:
    models = FakeModels(
        response=SimpleNamespace(
            response_id="gemini-req-1",
            model_version="gemini-3.5-flash-lite",
            text='{"winner":"B","brief_reason":"B is mathematically sound."}',
            candidates=[
                SimpleNamespace(
                    finish_reason="STOP",
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text='{"winner":"B","brief_reason":"B is mathematically sound."}'
                            )
                        ]
                    ),
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=15,
                candidates_token_count=9,
            ),
            model_dump=lambda: {"id": "gemini-req-1"},
        )
    )
    client = FakeGenAIClient(models=models)
    provider = GeminiJudgeProvider(client=client)

    response = provider.judge(
        "prompt text",
        model="gemini-3.5-flash-lite",
        temperature=0.0,
        max_output_tokens=256,
    )

    assert models.kwargs["model"] == "gemini-3.5-flash-lite"
    assert models.kwargs["contents"] == "prompt text"
    config = models.kwargs["config"]
    assert config.temperature == 0.0
    assert config.max_output_tokens == 256
    assert config.response_mime_type == "application/json"
    assert config.response_schema == RESPONSE_SCHEMA

    assert response.provider == "gemini"
    assert response.requested_model == "gemini-3.5-flash-lite"
    assert response.brief_reason == "B is mathematically sound."
    assert response.prompt_tokens == 15
    assert response.completion_tokens == 9
    assert response.raw_response_text.startswith('{"winner":"B"')


def test_gemini_provider_rejects_max_tokens() -> None:
    models = FakeModels(
        response=SimpleNamespace(
            response_id="gemini-req-2",
            model_version="gemini-3.5-flash-lite",
            text='{"winner":"B"}',
            candidates=[
                SimpleNamespace(
                    finish_reason="MAX_TOKENS",
                )
            ],
            usage_metadata=None,
        )
    )
    client = FakeGenAIClient(models=models)
    provider = GeminiJudgeProvider(client=client)

    with pytest.raises(
        ProviderError,
        match="incomplete Gemini response: MAX_TOKENS",
    ):
        provider.judge(
            "prompt text",
            model="gemini-3.5-flash-lite",
            temperature=0.0,
            max_output_tokens=256,
        )


def test_gemini_provider_retries_on_503() -> None:
    models = FakeModels(
        exc=RuntimeError("503 UNAVAILABLE. Model experiencing high demand.")
    )
    client = FakeGenAIClient(models=models)
    provider = GeminiJudgeProvider(client=client)

    with pytest.raises(RetriableProviderError):
        provider.judge(
            "prompt text",
            model="gemini-3.5-flash-lite",
            temperature=0.0,
            max_output_tokens=256,
        )


def test_gemini_preflight_checks(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    config = {
        "judge": {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
        "budget": {"required_available_calls": 480, "hard_api_call_cap": 500},
    }
    # Succeeds with confirmation and budget
    provider_preflight(config, trial_confirmed=True, available_call_budget=500)

    # Fails without operator confirmation
    with pytest.raises(RuntimeError, match="confirm"):
        provider_preflight(config, trial_confirmed=False, available_call_budget=500)
