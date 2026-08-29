from __future__ import annotations

import json
import os
import time
from typing import Any

from .base import JudgeResponse, ProviderError, RetriableProviderError
from ..utils import load_env_file


RESPONSE_FORMAT = {"type": "json_object"}


def _value(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _raw_payload(response: Any) -> Any:
    if hasattr(response, "model_dump_json"):
        try:
            return json.loads(response.model_dump_json())
        except Exception:
            pass
    for method in ("model_dump", "dict"):
        if hasattr(response, method):
            try:
                return getattr(response, method)()
            except Exception:
                pass
    return repr(response)


class GroqJudgeProvider:
    name = "groq"

    def __init__(self, timeout_seconds: float = 120, client: Any = None) -> None:
        load_env_file()
        api_key = os.environ.get("GROQ_API_KEY")
        if client is None and not api_key:
            raise ProviderError("GROQ_API_KEY is not set")
        if client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise ProviderError("install the 'groq' extra to use Groq") from exc
            client = Groq(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self.client = client

    def judge(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> JudgeResponse:
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=RESPONSE_FORMAT,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
        except Exception as exc:
            status = _value(exc, "status_code", "http_status")
            if status == 429 or (isinstance(status, int) and status >= 500):
                retry_after = _value(exc, "retry_after")
                error = RetriableProviderError(str(exc))
                setattr(error, "retry_after", retry_after)
                raise error from exc
            raise ProviderError(str(exc)) from exc

        choices = _value(response, "choices") or []
        choice = choices[0] if choices else None
        finish_reason = _value(choice, "finish_reason")
        if finish_reason and str(finish_reason).lower() not in ("stop", "complete"):
            raise ProviderError(f"incomplete Groq response: {finish_reason}")

        message = _value(choice, "message")
        text = str(_value(message, "content") or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        if not text:
            raise ProviderError("Groq response contained no text verdict")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError("Groq response was not valid JSON") from exc
        if not isinstance(parsed, dict) or parsed.get("winner") not in ("A", "B", "tie"):
            raise ProviderError("Groq response contained an invalid winner")
        if not isinstance(parsed.get("brief_reason"), str):
            raise ProviderError("Groq response contained an invalid brief_reason")

        usage = _value(response, "usage")
        return JudgeResponse(
            provider=self.name,
            requested_model=model,
            returned_model=_value(response, "model") or model,
            request_id=_value(response, "id"),
            prompt_tokens=_value(usage, "prompt_tokens"),
            completion_tokens=_value(usage, "completion_tokens"),
            raw_response_text=text,
            latency_seconds=time.perf_counter() - started,
            raw_response=_raw_payload(response),
            brief_reason=parsed.get("brief_reason"),
            http_status=200,
        )
