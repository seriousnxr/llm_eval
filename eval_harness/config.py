"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ServerConfig:
    """Model server connection settings."""

    base_url: str = "http://localhost:8000"
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    stop: list[str] | None = None
    timeout_seconds: int = 30


@dataclass
class RetryConfig:
    """Retry / backoff settings."""

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    retry_status_codes: list[int] = field(default_factory=lambda: [429, 503])


@dataclass
class ConcurrencyConfig:
    """Concurrency and batching settings."""

    max_workers: int = 10
    batch_size: int = 50


@dataclass
class OutputConfig:
    """Output directory settings."""

    results_dir: str = "results"


@dataclass
class EvalConfig:
    """Top-level evaluation configuration."""

    server: ServerConfig = field(default_factory=ServerConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        return cls(
            server=ServerConfig(**raw.get("server", {})),
            retry=RetryConfig(**raw.get("retry", {})),
            concurrency=ConcurrencyConfig(**raw.get("concurrency", {})),
            output=OutputConfig(**raw.get("output", {})),
        )

    @classmethod
    def default(cls) -> EvalConfig:
        """Return default configuration."""
        return cls()

    def ensure_output_dir(self) -> Path:
        """Create the results directory if it doesn't exist and return it."""
        results = Path(self.output.results_dir)
        results.mkdir(parents=True, exist_ok=True)
        return results
