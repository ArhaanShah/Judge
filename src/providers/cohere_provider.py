from __future__ import annotations

import json
import os
import time
from typing import Any

from .base import JudgeResponse, ProviderError, RetriableProviderError
from ..utils import load_env_file


RESPONSE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B", "tie"]},
            "brief_reason": {"type": "string"},
        },
        "required": ["winner", "brief_reason"],
    },
}


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


class CohereJudgeProvider:
    name = "cohere"

    def __init__(self, timeout_seconds: float = 120, client: Any = None) -> None:
        load_env_file()
        api_key = os.environ.get("COHERE_API_KEY")
        if client is None and not api_key:
            raise ProviderError("COHERE_API_KEY is not set")
        if client is None:
            try:
                import cohere
            except ImportError as exc:
                raise ProviderError("install the 'cohere' extra to use Cohere") from exc
            client = cohere.ClientV2(
                api_key=api_key,
                timeout=timeout_seconds,
                log_warning_experimental_features=False,
            )
        self.client = client

    def judge(self, prompt: str, *, model: str, temperature: float, max_output_tokens: int) -> JudgeResponse:
        started = time.perf_counter()
        try:
            response = self.client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=RESPONSE_FORMAT,
                temperature=temperature,
                max_tokens=max_output_tokens,
                thinking={"type": "disabled"},
            )
        except Exception as exc:
            status = _value(exc, "status_code", "http_status")
            if status == 429 or (isinstance(status, int) and status >= 500):
                retry_after = _value(exc, "retry_after")
                error = RetriableProviderError(str(exc))
                setattr(error, "retry_after", retry_after)
                raise error from exc
            raise ProviderError(str(exc)) from exc
        finish_reason_value = _value(response, "finish_reason")
        finish_reason = (
            _value(finish_reason_value, "value") or finish_reason_value
        )

        if str(finish_reason).upper() != "COMPLETE":
            raise ProviderError(
                f"incomplete Cohere response: "
                f"{finish_reason or 'missing finish reason'}"
            )

        message = _value(response, "message")
        content = _value(message, "content") or []

        text = next(
            (
                str(_value(item, "text"))
                for item in content
                if str(_value(item, "type") or "").lower() == "text"
                and _value(item, "text")
            ),
            "",
        )

        if not text.strip():
            raise ProviderError("Cohere response contained no text verdict")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError("Cohere response was not valid JSON") from exc

        if not isinstance(parsed, dict) or parsed.get("winner") not in (
            "A",
            "B",
            "tie",
        ):
            raise ProviderError("Cohere response contained an invalid winner")

        if not isinstance(parsed.get("brief_reason"), str):
            raise ProviderError(
                "Cohere response contained an invalid brief_reason"
            )
        usage = _value(response, "usage")
        billed = _value(usage, "billed_units") or usage
        return JudgeResponse(
            provider=self.name,
            requested_model=model,
            returned_model=_value(response, "model") or model,
            request_id=_value(response, "id", "request_id"),
            prompt_tokens=_value(billed, "input_tokens", "prompt_tokens"),
            completion_tokens=_value(billed, "output_tokens", "completion_tokens"),
            raw_response_text=text,
            latency_seconds=time.perf_counter() - started,
            raw_response=_raw_payload(response),
            brief_reason=parsed.get("brief_reason") if isinstance(parsed, dict) else None,
            http_status=200,
        )
