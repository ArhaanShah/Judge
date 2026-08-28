from __future__ import annotations

import time

from .base import JudgeResponse, ProviderError, RetriableProviderError


class OpenAIJudgeProvider:
    name = "openai"

    def __init__(self, *, timeout_seconds: float = 120) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the OpenAI extra with: pip install -e '.[openai]'"
            ) from exc
        self._client = OpenAI(timeout=timeout_seconds, max_retries=0)

    def judge(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> JudgeResponse:
        try:
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        except ImportError as exc:
            raise RuntimeError("the openai package is unavailable") from exc
        started = time.perf_counter()
        try:
            response = self._client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                store=False,
            )
        except (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        ) as exc:
            raise RetriableProviderError(str(exc)) from exc
        except APIStatusError as exc:
            raise ProviderError(str(exc)) from exc
        usage = getattr(response, "usage", None)
        return JudgeResponse(
            provider=self.name,
            requested_model=model,
            returned_model=getattr(response, "model", None),
            request_id=getattr(response, "id", None),
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            raw_response_text=response.output_text,
            latency_seconds=time.perf_counter() - started,
            error_status=(
                None
                if getattr(response, "status", "completed") == "completed"
                else str(getattr(response, "status", "unknown"))
            ),
        )
