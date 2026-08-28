from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


class RetriableProviderError(RuntimeError):
    """A transport, rate-limit, or transient server failure."""


class ProviderError(RuntimeError):
    """A non-retriable provider request failure."""


@dataclass(frozen=True)
class JudgeResponse:
    provider: str
    requested_model: str
    returned_model: str | None
    request_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    raw_response_text: str
    latency_seconds: float
    error_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JudgeProvider(Protocol):
    name: str

    def judge(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> JudgeResponse:
        ...
