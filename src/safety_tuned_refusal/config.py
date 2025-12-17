"""
Configuration defaults and constants shared across the toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_ALIGNED_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = DEFAULT_ALIGNED_MODEL
    max_new_tokens: int = 256
    temperature: float = 0.7
    n_samples: int = 3
    device: str | None = None  # auto-detect when None


@dataclass(frozen=True)
class JudgeConfig:
    model_name: str = DEFAULT_JUDGE_MODEL
    rate_limit_rpm: int = 15
    api_key_env: str = "GOOGLE_API_KEY"
