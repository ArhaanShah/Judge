from __future__ import annotations

import hashlib
import time

from .base import JudgeResponse
from ..utils import token_count


class MockJudgeProvider:
    name = "mock"

    def judge(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> JudgeResponse:
        started = time.perf_counter()
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        verdict = "A" if int(digest[0], 16) % 2 == 0 else "B"
        text = f"Deterministic mock response.\nVERDICT: {verdict}"
        return JudgeResponse(
            provider=self.name,
            requested_model=model,
            returned_model="deterministic-mock-v1",
            request_id=f"mock_{digest[:16]}",
            prompt_tokens=token_count(prompt, model),
            completion_tokens=token_count(text, model),
            raw_response_text=text,
            latency_seconds=time.perf_counter() - started,
        )
