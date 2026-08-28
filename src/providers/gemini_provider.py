from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from .base import JudgeResponse, ProviderError, RetriableProviderError
from ..utils import load_env_file

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "winner": {"type": "STRING", "enum": ["A", "B", "tie"]},
        "brief_reason": {"type": "STRING"},
    },
    "required": ["winner", "brief_reason"],
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
    for method in ("model_dump", "to_dict", "dict"):
        if hasattr(response, method):
            try:
                return getattr(response, method)()
            except Exception:
                pass
    return repr(response)


def _extract_retry_after(exc: Any) -> float | None:
    val = getattr(exc, "retry_after", None)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    response_json = getattr(exc, "response_json", None) or getattr(exc, "error_details", None)
    if isinstance(response_json, dict):
        error = response_json.get("error", response_json)
        details = error.get("details", []) if isinstance(error, dict) else []
        for detail in details:
            if isinstance(detail, dict) and "retryDelay" in detail:
                raw_delay = str(detail["retryDelay"]).rstrip("s")
                try:
                    return float(raw_delay) + 1.0
                except (ValueError, TypeError):
                    pass
    msg = str(exc)
    match = re.search(r"[Pp]lease retry in ([\d\.]+)s", msg)
    if match:
        try:
            return float(match.group(1)) + 1.0
        except ValueError:
            pass
    return None


class GeminiJudgeProvider:
    name = "gemini"

    def __init__(self, timeout_seconds: float = 120, client: Any = None) -> None:
        load_env_file()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if client is None and not api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        if client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ProviderError("install the 'gemini' extra to use Gemini") from exc
            client = genai.Client(
                api_key=api_key,
                http_options={"timeout": int(timeout_seconds * 1000)},
            )
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
            from google.genai import errors, types
        except ImportError:
            errors = None
            types = None

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
        }
        if types is not None and hasattr(types, "AutomaticFunctionCallingConfig"):
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )

        config = types.GenerateContentConfig(**config_kwargs) if types is not None else config_kwargs

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            status = _value(exc, "code", "status_code", "http_status")
            retry_after = _extract_retry_after(exc)
            error_msg = str(exc)

            is_server_error = errors is not None and isinstance(exc, errors.ServerError)
            is_retriable_status = status in (429, 500, 502, 503, 504) or (
                isinstance(status, int) and status >= 500
            )
            is_demand_error = "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "ResourceExhausted" in error_msg or "timed out" in error_msg.lower()

            if is_server_error or is_retriable_status or is_demand_error:
                error = RetriableProviderError(error_msg)
                if retry_after:
                    setattr(error, "retry_after", retry_after)
                raise error from exc
            raise ProviderError(error_msg) from exc

        candidates = _value(response, "candidates") or []
        first_candidate = candidates[0] if candidates else None
        finish_reason = _value(first_candidate, "finish_reason")
        if finish_reason:
            finish_str = str(_value(finish_reason, "name") or finish_reason).upper()
            if finish_str not in ("STOP", "FINISH_REASON_UNSPECIFIED", "COMPLETE", "NONE"):
                raise ProviderError(f"incomplete Gemini response: {finish_str}")

        text = getattr(response, "text", None)
        if not text and first_candidate:
            content = _value(first_candidate, "content")
            parts = _value(content, "parts") or []
            text = "".join(str(_value(part, "text") or "") for part in parts)

        cleaned_text = str(text or "").strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        if not cleaned_text:
            raise ProviderError("Gemini response contained no text verdict")

        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            raise ProviderError("Gemini response was not valid JSON") from exc

        if not isinstance(parsed, dict) or parsed.get("winner") not in ("A", "B", "tie"):
            raise ProviderError("Gemini response contained an invalid winner")

        if not isinstance(parsed.get("brief_reason"), str):
            raise ProviderError("Gemini response contained an invalid brief_reason")

        usage = _value(response, "usage_metadata")
        prompt_tokens = _value(usage, "prompt_token_count", "input_tokens")
        completion_tokens = _value(usage, "candidates_token_count", "output_tokens")

        return JudgeResponse(
            provider=self.name,
            requested_model=model,
            returned_model=_value(response, "model_version", "model") or model,
            request_id=_value(response, "response_id", "id"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_response_text=cleaned_text,
            latency_seconds=time.perf_counter() - started,
            raw_response=_raw_payload(response),
            brief_reason=parsed.get("brief_reason"),
            http_status=200,
        )
