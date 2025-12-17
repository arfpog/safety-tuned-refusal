"""
Safety-tuned refusal analysis toolkit.

This package exposes utilities for:
- generating prompts across identity axes
- running an aligned model to produce responses
- judging refusals vs hedging vs direct answers
- probing model hidden states for identity and safety signals
"""

__all__ = [
    "config",
    "prompts",
    "inference",
    "judge",
    "probes",
    "metrics",
]
