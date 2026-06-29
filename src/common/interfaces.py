"""
interfaces.py — the contracts every component must satisfy.

OWNED BY: Ajay (M1). The `...` bodies mean "no code yet — you fill it in."

WHO IMPLEMENTS WHAT:
    - M2 (Harshith) -> Engine          in src/engine/engine.py
    - M3 (Tarun)    -> Scheduler        in src/scheduler/base_scheduler.py
    - M4 (Pavan)    -> KVCacheManager    in src/kv_cache/block_allocator.py

RULE: do not change a method signature without announcing it to the whole team.
Everyone depends on these shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from common.request import Request


# ---------------------------------------------------------------------------
# Result types passed between components
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    """What one engine step produces. Returned by Engine.step()."""
    new_token_ids: dict[str, int]         # request_id -> newly generated token id
    finished_ids: list[str]               # request_ids that finished this step


@dataclass
class SchedulerOutput:
    """What the scheduler decides for one iteration. Returned by Scheduler.step()."""
    scheduled: list[Request]              # requests to run this step
    num_waiting: int = 0                  # bookkeeping for M5's metrics
    num_running: int = 0


# ---------------------------------------------------------------------------
# Interface 1: KVCacheManager  (Pavan / M4 implements)
# ---------------------------------------------------------------------------

@runtime_checkable
class KVCacheManager(Protocol):
    """The block-based KV cache — PagedAttention-lite's core."""

    def can_allocate(self, request: Request) -> bool:
        """True if there's at least one free block to admit/grow this request.
        The scheduler MUST call this before admitting or extending a request."""
        ...

    def allocate(self, request: Request) -> None:
        """Reserve the initial block(s) for a newly admitted request."""
        ...

    def append_token(self, request_id: str) -> None:
        """Account for one more generated token; grab a new physical block
        if the request just crossed a block boundary."""
        ...

    def get_block_table(self, request_id: str) -> list[int]:
        """Return the ordered physical block ids holding this request's K/V.
        The engine calls this every step to know where to read K/V from."""
        ...

    def free(self, request_id: str) -> None:
        """Return all of this request's blocks to the free pool."""
        ...

    # --- metrics (M5 reads these) ---
    def num_free_blocks(self) -> int: ...
    def num_total_blocks(self) -> int: ...


# ---------------------------------------------------------------------------
# Interface 2: Scheduler  (Tarun / M3 implements)
# ---------------------------------------------------------------------------

@runtime_checkable
class Scheduler(Protocol):
    """Decides which requests run on each engine step."""

    def add_request(self, request: Request) -> None:
        """Enqueue a newly arrived request (called from M1's HTTP side)."""
        ...

    def abort_request(self, request_id: str) -> None:
        """Remove a request (client disconnected, etc.)."""
        ...

    def step(self) -> SchedulerOutput:
        """Run admission/eviction logic and return the requests the engine
        should process on the next forward pass."""
        ...

    def has_unfinished(self) -> bool:
        """True while any request is still waiting or running."""
        ...


# ---------------------------------------------------------------------------
# Interface 3: Engine  (Harshith / M2 implements)
# ---------------------------------------------------------------------------

@runtime_checkable
class Engine(Protocol):
    """Runs the model. One step = one forward pass = one new token per request."""

    def step(self, scheduled: list[Request]) -> StepOutput:
        """Run one forward pass over the scheduled requests; return the new
        token per request and which requests finished."""
        ...
