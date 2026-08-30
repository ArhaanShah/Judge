import json
import os
import time
import hashlib
from typing import Any
import httpx

from .base import JudgeProvider, JudgeResponse, RetriableProviderError, ProviderError

class NIMModelProfile:
    def __init__(self, config: dict[str, Any]):
        self.model_slug = config.get("model", "z-ai/glm-5.2")
        self.temperature = config.get("temperature", 0.0)
        self.top_p = config.get("top_p")
        self.max_tokens = config.get("max_tokens", 16384)
        self.seed = config.get("seed", 20260828)
        self.stream = config.get("stream", True)
        
        self.reasoning_mode = config.get("reasoning_mode")
        self.reasoning_effort = config.get("reasoning_effort")
        self.reasoning_budget = config.get("reasoning_budget")
        self.extra_body = config.get("extra_body", {})
        self.reasoning_must_be_verified = config.get("reasoning_must_be_verified", False)


class NvidiaNimProvider(JudgeProvider):
    name = "nvidia_nim"

    def __init__(self, config: dict[str, Any], timeout_seconds: float = 120.0):
        self.config = config
        self.profile = NIMModelProfile(config)
        self.timeout_seconds = timeout_seconds
        
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ProviderError("NVIDIA_API_KEY environment variable is required")
            
        self.base_url = config.get("base_url", "https://integrate.api.nvidia.com/v1/chat/completions")

    def judge(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> JudgeResponse:
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self.profile.stream else "application/json"
        }
        
        payload = {
            "model": model or self.profile.model_slug,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.profile.temperature if temperature is None else temperature,
            "max_tokens": self.profile.max_tokens if max_output_tokens is None else max_output_tokens,
            "stream": self.profile.stream,
        }
        
        if self.profile.top_p is not None:
            payload["top_p"] = self.profile.top_p
        if self.profile.seed is not None:
            payload["seed"] = self.profile.seed
        if self.profile.reasoning_effort is not None:
            payload["reasoning_effort"] = self.profile.reasoning_effort
        if self.profile.reasoning_budget is not None:
            payload["reasoning_budget"] = self.profile.reasoning_budget
            
        if self.profile.extra_body:
            payload.update(self.profile.extra_body)
            
        start_time = time.time()
        
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                if self.profile.stream:
                    return self._handle_stream(client, payload, headers, start_time)
                else:
                    return self._handle_sync(client, payload, headers, start_time)
        except httpx.TimeoutException as e:
            raise RetriableProviderError(f"Request timed out: {e}")
        except httpx.NetworkError as e:
            raise RetriableProviderError(f"Network error: {e}")

    def _handle_sync(self, client: httpx.Client, payload: dict, headers: dict, start_time: float) -> JudgeResponse:
        response = client.post(self.base_url, json=payload, headers=headers)
        latency = time.time() - start_time
        
        if response.status_code in (429, 500, 502, 503, 504):
            raise RetriableProviderError(f"HTTP {response.status_code}: {response.text}")
        if response.status_code != 200:
            raise ProviderError(f"HTTP {response.status_code}: {response.text}")
            
        data = response.json()
        
        message = data["choices"][0].get("message", {})
        final_content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")
        finish_reason = data["choices"][0].get("finish_reason")
        usage = data.get("usage", {})
        
        return self._build_response(
            payload, response.headers, latency, final_content, reasoning_content,
            usage, finish_reason, response.status_code
        )

    def _handle_stream(self, client: httpx.Client, payload: dict, headers: dict, start_time: float) -> JudgeResponse:
        final_content = ""
        reasoning_content = ""
        finish_reason = None
        usage = {}
        
        try:
            with client.stream("POST", self.base_url, json=payload, headers=headers) as response:
                if response.status_code in (429, 500, 502, 503, 504):
                    response.read()
                    raise RetriableProviderError(f"HTTP {response.status_code}: {response.text}")
                if response.status_code != 200:
                    response.read()
                    raise ProviderError(f"HTTP {response.status_code}: {response.text}")
                    
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                        
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                        
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                            reasoning_content += delta["reasoning_content"]
                        if "content" in delta and delta["content"] is not None:
                            final_content += delta["content"]
                        if choices[0].get("finish_reason"):
                            finish_reason = choices[0]["finish_reason"]
                            
        except httpx.ReadError as e:
            raise RetriableProviderError(f"Stream read error: {e}")
            
        latency = time.time() - start_time
        return self._build_response(
            payload, response.headers, latency, final_content, reasoning_content,
            usage, finish_reason, response.status_code
        )

    def _build_response(
        self, payload: dict, headers: httpx.Headers, latency: float, 
        final_content: str, reasoning_content: str, usage: dict, 
        finish_reason: str, status_code: int
    ) -> JudgeResponse:
    
        raw_response = {
            "reasoning_present": bool(reasoning_content),
            "reasoning_character_count": len(reasoning_content) if reasoning_content else 0,
            "reasoning_sha256": hashlib.sha256(reasoning_content.encode("utf-8")).hexdigest() if reasoning_content else None,
            "reasoning_tokens_if_reported": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            "finish_reason": finish_reason,
            "retry_after": headers.get("retry-after"),
            "x_ratelimit_limit": headers.get("x-ratelimit-limit"),
            "x_ratelimit_remaining": headers.get("x-ratelimit-remaining"),
            "x_ratelimit_reset": headers.get("x-ratelimit-reset"),
        }
        
        return JudgeResponse(
            provider=self.name,
            requested_model=payload["model"],
            returned_model=None, # Not explicitly available in standard response chunk outside of 'model'
            request_id=headers.get("x-request-id"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw_response_text=final_content,
            latency_seconds=latency,
            raw_response=raw_response,
            http_status=status_code
        )
