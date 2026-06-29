"""
base_scheduler.py — Abstract base scheduler implementing common interface.
Owner: Tarun (Member 3 — Scheduler Engineer)
"""

from __future__ import annotations

import abc
import time
from typing import List, Optional

from src.common.interfaces import SchedulerInterface
from src.common.request import Request, RequestState
from src.scheduler.scheduler_metrics import SchedulerMetrics


class BaseScheduler(SchedulerInterface, abc.ABC):
    """
    Abstract base class for all schedulers.
    Provides shared state (waiting/running/finished queues) and metric hooks.
    Concrete subclasses must implement `schedule()`.
    """

    def __init__(
        self,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 4096,
        max_batch_size: int = 32,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_batch_size = max_batch_size

        # Core queues
        self._waiting: list[Request] = []       # Admitted but not yet running
        self._running: list[Request] = []       # Currently in a forward pass
        self._finished: list[Request] = []      # Done (success or error)

        self.metrics = SchedulerMetrics()
        self._step_index: int = 0

    # ------------------------------------------------------------------
    # Public queue-management helpers (used by subclasses & engine)
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Enqueue a new request for scheduling."""
        request.state = RequestState.WAITING
        request.arrival_time = request.arrival_time or time.monotonic()
        self._waiting.append(request)

    def get_running(self) -> list[Request]:
        """Return the currently running batch (read-only view)."""
        return list(self._running)

    def get_finished(self) -> list[Request]:
        """Return finished requests and clear the finished list."""
        done = list(self._finished)
        self._finished.clear()
        return done

    def num_waiting(self) -> int:
        return len(self._waiting)

    def num_running(self) -> int:
        return len(self._running)

    # ------------------------------------------------------------------
    # Lifecycle hooks called by the engine
    # ------------------------------------------------------------------

    def on_step_begin(self) -> None:
        """Called by the engine at the start of every forward-pass step."""
        self._step_start_ts = time.monotonic()

    def on_step_end(self, finished_ids: list[str]) -> None:
        """
        Called by the engine at the end of every step.
        `finished_ids` — request IDs that completed this step.
        """
        # Move finished requests out of running
        still_running: list[Request] = []
        for req in self._running:
            if req.request_id in finished_ids:
                req.state = RequestState.FINISHED
                req.finish_time = time.monotonic()
                self._finished.append(req)
            else:
                still_running.append(req)
        self._running = still_running

        # Record step metrics
        elapsed = time.monotonic() - self._step_start_ts
        self.metrics.record_step(
            step_index=self._step_index,
            batch_size=len(self._running),
            queue_depth=len(self._waiting),
            iteration_latency_s=elapsed,
            tokens_generated=len(finished_ids),
        )
        self._step_index += 1

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def schedule(self, kv_allocator) -> list[Request]:
        """
        Decide which requests run on the next forward pass.

        Args:
            kv_allocator: An object with `.can_allocate(request)` and
                          `.allocate(request)` methods (Member 4's BlockAllocator).

        Returns:
            The list of Request objects to feed into the engine this step.
        """

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable scheduler name for logging/metrics."""