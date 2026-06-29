"""
request.py — the Request object and its lifecycle states.

OWNED BY: Ajay (M1). This is the job ticket that follows a request through the
whole system. Everyone reads it; be careful who WRITES each field:

    - M1 (Ajay)     sets: request_id, prompt, prompt_tokens, sampling,
                          arrival_time, state(initial)
    - M3 (Tarun)    sets: state (WAITING -> RUNNING -> FINISHED)
    - M2 (Harshith) appends to: output_tokens; sets finish_reason
    - M4 (Pavan)    keyed by: request_id (owns the block table externally)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from common.sampling_params import SamplingParams


class RequestState(str, Enum):
    """Where a request is in its lifecycle."""
    WAITING = "waiting"     # in the queue, not yet admitted by the scheduler
    RUNNING = "running"     # currently being generated, step by step
    FINISHED = "finished"   # hit EOS, a stop string, or max_tokens
    ABORTED = "aborted"     # client disconnected or error


@dataclass
class Request:
    """One user request as it travels through the whole system."""
    prompt: str
    sampling: SamplingParams

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    arrival_time: float = field(default_factory=time.monotonic)

    # Token bookkeeping. M2 (Harshith) fills these as generation proceeds.
    prompt_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)

    state: RequestState = RequestState.WAITING
    finish_reason: str | None = None      # "stop", "length", "eos", "abort"

    @property
    def is_finished(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.ABORTED)

    @property
    def is_prefill(self) -> bool:
        """True while the prompt is still being processed (no output yet).
        The engine's mixed prefill+decode forward pass (forward.py) uses this
        to decide which attention path a request takes this step."""
        return len(self.output_tokens) == 0

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)
