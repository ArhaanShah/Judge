from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.providers.base import ProviderError
from src.providers.cohere_provider import (
    CohereJudgeProvider,
    RESPONSE_FORMAT,
)


class FakeClient:
    def __init__(self, response=None) -> None:
        self.kwargs = None
        self.response = response or SimpleNamespace(
            id="request-1",
            model="command-a-plus-05-2026",
            finish_reason="COMPLETE",
            message=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="thinking",
                        thinking="Compare the candidates.",
                    ),
                    SimpleNamespace(
                        type="text",
                        text=(
                            '{"winner":"A",'
                            '"brief_reason":"A is correct."}'
                        ),
                    ),
                ]
            ),
            usage=SimpleNamespace(
                billed_units=SimpleNamespace(
                    input_tokens=12,
                    output_tokens=8,
                )
            ),
            model_dump=lambda: {"id": "request-1"},
        )

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_cohere_provider_uses_structured_schema() -> None:
    client = FakeClient()
    provider = CohereJudgeProvider(client=client)

    response = provider.judge(
        "prompt",
        model="command-a-plus-05-2026",
        temperature=0,
        max_output_tokens=256,
    )

    assert client.kwargs["response_format"] == RESPONSE_FORMAT
    assert client.kwargs["temperature"] == 0
    assert client.kwargs["max_tokens"] == 256
    assert client.kwargs["thinking"] == {"type": "disabled"}
    assert response.raw_response_text.startswith('{"winner":"A"')
    assert response.brief_reason == "A is correct."


def test_cohere_provider_rejects_max_tokens() -> None:
    client = FakeClient(
        SimpleNamespace(
            finish_reason="MAX_TOKENS",
            message=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="thinking",
                        thinking="unfinished",
                    )
                ]
            ),
        )
    )
    provider = CohereJudgeProvider(client=client)

    with pytest.raises(
        ProviderError,
        match="incomplete Cohere response: MAX_TOKENS",
    ):
        provider.judge(
            "prompt",
            model="command-a-plus-05-2026",
            temperature=0,
            max_output_tokens=256,
        )