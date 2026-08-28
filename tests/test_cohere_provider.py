from __future__ import annotations

from types import SimpleNamespace

from src.providers.cohere_provider import CohereJudgeProvider, RESPONSE_FORMAT


class FakeClient:
    def __init__(self) -> None:
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="request-1",
            model="command-a-plus-05-2026",
            message=SimpleNamespace(content=[SimpleNamespace(text='{"winner":"A","brief_reason":"A is correct."}')]),
            usage=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=12, output_tokens=8)),
            model_dump=lambda: {"id": "request-1"},
        )


def test_cohere_provider_uses_frozen_structured_schema() -> None:
    client = FakeClient()
    provider = CohereJudgeProvider(client=client)
    response = provider.judge("prompt", model="command-a-plus-05-2026", temperature=0, max_output_tokens=256)
    assert client.kwargs["response_format"] == RESPONSE_FORMAT
    assert client.kwargs["temperature"] == 0
    assert client.kwargs["max_tokens"] == 256
    assert response.raw_response_text.startswith('{"winner":"A"')
    assert response.brief_reason == "A is correct."
