"""API client with retry logic, backoff, and robust error handling."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from eval_harness.config import EvalConfig

logger = logging.getLogger(__name__)


@dataclass
class ClientResponse:
    """Typed response from the API client."""

    content: str
    latency_ms: float
    error_category: str = "none"
    raw_response: dict[str, Any] | None = None
    retries_used: int = 0


@dataclass
class ErrorRecord:
    """Structured error log entry."""

    timestamp: str
    error_type: str
    status_code: int | None
    message: str
    retry_count: int
    prompt_preview: str


class EvalClient:
    """OpenAI-compatible API client with retry, backoff, and error handling.

    Handles:
    - Rate limiting (429) with Retry-After header support
    - Server overload (503) with exponential backoff
    - Truncated JSON on 200 responses
    - Slow/chunked responses via per-request timeout
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.server = config.server
        self.retry = config.retry
        self.session = requests.Session()
        self.error_log: list[ErrorRecord] = []

    @property
    def endpoint(self) -> str:
        return f"{self.server.base_url}/v1/chat/completions"

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build the request payload."""
        payload: dict[str, Any] = {
            "model": overrides.get("model", self.server.model),
            "messages": messages,
            "temperature": overrides.get("temperature", self.server.temperature),
            "max_tokens": overrides.get("max_tokens", self.server.max_tokens),
            "top_p": overrides.get("top_p", self.server.top_p),
        }
        stop = overrides.get("stop", self.server.stop)
        if stop is not None:
            payload["stop"] = stop
        return payload

    def _log_error(
        self,
        error_type: str,
        status_code: int | None,
        message: str,
        retry_count: int,
        prompt_preview: str,
    ) -> None:
        record = ErrorRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            error_type=error_type,
            status_code=status_code,
            message=message,
            retry_count=retry_count,
            prompt_preview=prompt_preview[:120],
        )
        self.error_log.append(record)
        logger.warning(
            "API error: type=%s status=%s retry=%d msg=%s",
            error_type,
            status_code,
            retry_count,
            message[:200],
        )

    def _backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Calculate backoff delay with jitter."""
        if retry_after is not None:
            return retry_after + random.uniform(0, 1)
        delay = min(
            self.retry.base_delay * (2**attempt) + random.uniform(0, 1),
            self.retry.max_delay,
        )
        return delay

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        **overrides: Any,
    ) -> ClientResponse:
        """Send a chat completion request with retry logic.

        Args:
            messages: List of chat messages (role/content dicts).
            **overrides: Override any server config param (model, temperature, etc.).

        Returns:
            ClientResponse with content, latency, and error info.
        """
        payload = self._build_payload(messages, **overrides)
        prompt_preview = messages[-1].get("content", "")[:120] if messages else ""
        timeout = overrides.get("timeout", self.server.timeout_seconds)

        last_error_category = "api_error"

        for attempt in range(self.retry.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self.session.post(
                    self.endpoint,
                    json=payload,
                    timeout=timeout,
                )
                latency_ms = (time.perf_counter() - start) * 1000

                # --- Retryable status codes (429, 503) ---
                if resp.status_code in self.retry.retry_status_codes:
                    error_type = (
                        "rate_limit" if resp.status_code == 429 else "server_error"
                    )
                    last_error_category = error_type
                    retry_after = None
                    if "Retry-After" in resp.headers:
                        try:
                            retry_after = float(resp.headers["Retry-After"])
                        except ValueError:
                            pass
                    self._log_error(
                        error_type,
                        resp.status_code,
                        resp.text[:200],
                        attempt,
                        prompt_preview,
                    )
                    if attempt < self.retry.max_retries:
                        time.sleep(self._backoff_delay(attempt, retry_after))
                        continue
                    return ClientResponse(
                        content="",
                        latency_ms=latency_ms,
                        error_category=error_type,
                        retries_used=attempt,
                    )

                # --- Non-200 / non-retryable errors ---
                if resp.status_code != 200:
                    self._log_error(
                        "http_error",
                        resp.status_code,
                        resp.text[:200],
                        attempt,
                        prompt_preview,
                    )
                    return ClientResponse(
                        content="",
                        latency_ms=latency_ms,
                        error_category="api_error",
                        retries_used=attempt,
                    )

                # --- 200 OK: try to parse JSON ---
                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    # Truncated JSON — retry
                    last_error_category = "truncated_json"
                    self._log_error(
                        "truncated_json",
                        200,
                        f"Invalid JSON: {resp.text[:100]}...",
                        attempt,
                        prompt_preview,
                    )
                    if attempt < self.retry.max_retries:
                        time.sleep(self._backoff_delay(attempt))
                        continue
                    return ClientResponse(
                        content="",
                        latency_ms=latency_ms,
                        error_category="truncated_json",
                        retries_used=attempt,
                    )

                # --- Extract content ---
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    self._log_error(
                        "malformed_response",
                        200,
                        f"Missing content in response: {str(data)[:200]}",
                        attempt,
                        prompt_preview,
                    )
                    if attempt < self.retry.max_retries:
                        time.sleep(self._backoff_delay(attempt))
                        continue
                    return ClientResponse(
                        content="",
                        latency_ms=latency_ms,
                        error_category="malformed_response",
                        retries_used=attempt,
                    )

                return ClientResponse(
                    content=content,
                    latency_ms=latency_ms,
                    raw_response=data,
                    retries_used=attempt,
                )

            except requests.exceptions.Timeout:
                latency_ms = (time.perf_counter() - start) * 1000
                last_error_category = "timeout"
                self._log_error(
                    "timeout",
                    None,
                    f"Request timed out after {timeout}s",
                    attempt,
                    prompt_preview,
                )
                if attempt < self.retry.max_retries:
                    time.sleep(self._backoff_delay(attempt))
                    continue
                return ClientResponse(
                    content="",
                    latency_ms=latency_ms,
                    error_category="timeout",
                    retries_used=attempt,
                )

            except requests.exceptions.ConnectionError as e:
                latency_ms = (time.perf_counter() - start) * 1000
                last_error_category = "connection_error"
                self._log_error(
                    "connection_error",
                    None,
                    str(e)[:200],
                    attempt,
                    prompt_preview,
                )
                if attempt < self.retry.max_retries:
                    time.sleep(self._backoff_delay(attempt))
                    continue
                return ClientResponse(
                    content="",
                    latency_ms=latency_ms,
                    error_category="connection_error",
                    retries_used=attempt,
                )

        # Should not reach here, but just in case
        return ClientResponse(
            content="",
            latency_ms=0,
            error_category=last_error_category,
            retries_used=self.retry.max_retries,
        )

    def dump_error_log(self, path: str) -> None:
        """Write collected errors to a JSONL file."""
        from pathlib import Path as P

        P(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in self.error_log:
                f.write(json.dumps(record.__dict__) + "\n")
