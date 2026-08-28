from __future__ import annotations

from typing import Any

from .base import JudgeProvider
from .mock import MockJudgeProvider
from .openai_provider import OpenAIJudgeProvider


def make_provider(judge_config: dict[str, Any]) -> JudgeProvider:
    name = str(judge_config["provider"]).lower()
    if name == "mock":
        return MockJudgeProvider()
    if name == "openai":
        return OpenAIJudgeProvider(
            timeout_seconds=float(judge_config.get("timeout_seconds", 120))
        )
    if name == "cohere":
        from .cohere_provider import CohereJudgeProvider
        return CohereJudgeProvider(
            timeout_seconds=float(judge_config.get("timeout_seconds", 120))
        )
    raise ValueError(f"unsupported judge provider: {name}")
