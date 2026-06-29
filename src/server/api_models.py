"""
api_models.py — the JSON shapes the HTTP API accepts and returns.

OWNED BY: Ajay (M1). These are Pydantic models: FastAPI uses them to validate
incoming JSON and to serialize responses. They are the PUBLIC API contract,
separate from the internal Request object in common/request.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Body of POST /generate and POST /generate/stream."""
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1)
    stop: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    """Body returned by POST /generate."""
    request_id: str
    text: str
    output_tokens: list[int]
    finish_reason: str | None


class HealthResponse(BaseModel):
    status: str


class StatsResponse(BaseModel):
    """Body returned by GET /stats. Live system metrics."""
    num_waiting: int
    num_running: int
    free_blocks: int
    total_blocks: int
