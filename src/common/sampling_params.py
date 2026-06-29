"""
sampling_params.py — how the engine picks the next token, per request.

OWNED BY: Ajay (M1). Read by the engine (Harshith) inside the sampler.
Do not change field names without announcing to the team.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    """Per-request sampling configuration."""
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1                       # -1 means "disabled"
    stop: list[str] = field(default_factory=list)
