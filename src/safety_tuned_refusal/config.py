"""
Configuration defaults and constants shared across the toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_ALIGNED_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"
DEFAULT_JUDGE_PROVIDER = "gemini"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = DEFAULT_ALIGNED_MODEL
    max_new_tokens: int = 256
    temperature: float = 0.7
    n_samples: int = 3
    device: str | None = None  # auto-detect when None


@dataclass(frozen=True)
class JudgeConfig:
    provider: str = DEFAULT_JUDGE_PROVIDER  # "gemini" | "openrouter"
    model_name: str = DEFAULT_JUDGE_MODEL
    rate_limit_rpm: int = 15
    api_key_env: str = "GOOGLE_API_KEY"
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None
