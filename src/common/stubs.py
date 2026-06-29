"""
stubs.py — working fakes that obey the interfaces.

OWNED BY: Ajay (M1). These are real, runnable stand-ins so each person can
test in isolation BEFORE the real components exist:

    - Ajay     : build the HTTP server against DummyEngine + DummyScheduler.
    - Harshith : drop your real Engine in for DummyEngine.
    - Tarun    : drop your real Scheduler in for DummyScheduler.
    - Pavan    : drop your real KVCacheManager in for DummyKVCacheManager.

The dummies also double as a behavioral reference: your real component should
reproduce the same lifecycle (admit -> step -> finish -> free).

tests/conftest.py should expose these as fixtures.
"""

from __future__ import annotations

from common.request import Request, RequestState
from common.interfaces import StepOutput, SchedulerOutput


class DummyKVCacheManager:
    """Pretends to have plenty of memory. Tracks block ids but nothing real."""

    def __init__(self, total_blocks: int = 1000) -> None:
        self._total = total_blocks
        self._tables: dict[str, list[int]] = {}
        self._next_block = 0

    def can_allocate(self, request: Request) -> bool:
        return self._next_block < self._total

    def allocate(self, request: Request) -> None:
        self._tables[request.request_id] = [self._next_block]
        self._next_block += 1

    def append_token(self, request_id: str) -> None:
        self._tables.setdefault(request_id, []).append(self._next_block)
        self._next_block += 1

    def get_block_table(self, request_id: str) -> list[int]:
        return self._tables.get(request_id, [])

    def free(self, request_id: str) -> None:
        self._tables.pop(request_id, None)

    def num_free_blocks(self) -> int:
        return self._total - self._next_block

    def num_total_blocks(self) -> int:
        return self._total


class DummyEngine:
    """Generates fake tokens. Each request finishes after max_tokens outputs."""

    def step(self, scheduled: list[Request]) -> StepOutput:
        new_tokens: dict[str, int] = {}
        finished: list[str] = []
        for req in scheduled:
            fake_token = 42                       # stand-in token id
            req.output_tokens.append(fake_token)
            new_tokens[req.request_id] = fake_token
            if len(req.output_tokens) >= req.sampling.max_tokens:
                req.state = RequestState.FINISHED
                req.finish_reason = "length"
                finished.append(req.request_id)
        return StepOutput(new_token_ids=new_tokens, finished_ids=finished)


class DummyScheduler:
    """First-come-first-served, no real memory limits. Runs everything waiting."""

    def __init__(self, max_batch_size: int = 8) -> None:
        self._max_batch = max_batch_size
        self._waiting: list[Request] = []
        self._running: list[Request] = []

    def add_request(self, request: Request) -> None:
        request.state = RequestState.WAITING
        self._waiting.append(request)

    def abort_request(self, request_id: str) -> None:
        self._waiting = [r for r in self._waiting if r.request_id != request_id]
        self._running = [r for r in self._running if r.request_id != request_id]

    def step(self) -> SchedulerOutput:
        self._running = [r for r in self._running if not r.is_finished]
        while self._waiting and len(self._running) < self._max_batch:
            req = self._waiting.pop(0)
            req.state = RequestState.RUNNING
            self._running.append(req)
        return SchedulerOutput(
            scheduled=list(self._running),
            num_waiting=len(self._waiting),
            num_running=len(self._running),
        )

    def has_unfinished(self) -> bool:
        return bool(self._waiting or self._running)
