"""
queue_manager.py — Waiting / Running / Finished queue management.
Owner: Tarun (Member 3 — Scheduler Engineer)

Provides a single, thread-safe-ish container for all three queues so
the scheduler core stays clean.  The engine and server interact with
the scheduler through this manager via the BaseScheduler interface;
direct access to QueueManager is scheduler-internal only.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Iterator

from src.common.request import Request, RequestState

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Manages the three lifecycle queues for LLM inference requests:

      waiting  — admitted to the system, not yet scheduled for a step
      running  — currently in the active forward-pass batch
      finished — completed (success or error); drained by the server
    """

    def __init__(self) -> None:
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []
        self._finished: list[Request] = []

        # fast lookup: request_id → Request
        self._id_map: dict[str, Request] = {}

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(self, request: Request) -> None:
        """Add a new request to the waiting queue."""
        if request.request_id in self._id_map:
            raise ValueError(f"Duplicate request_id: {request.request_id}")
        request.state = RequestState.WAITING
        request.arrival_time = request.arrival_time or time.monotonic()
        self._waiting.append(request)
        self._id_map[request.request_id] = request
        logger.debug("Enqueued %s", request.request_id)

    def enqueue_front(self, request: Request) -> None:
        """Re-enqueue a preempted request at the front (high priority)."""
        request.state = RequestState.WAITING
        self._waiting.appendleft(request)
        # id_map already has it
        logger.debug("Re-enqueued (front) %s", request.request_id)

    # ------------------------------------------------------------------
    # Promote waiting → running
    # ------------------------------------------------------------------

    def promote_to_running(self, requests: list[Request]) -> None:
        """Move the given requests from waiting to running."""
        promote_ids = {r.request_id for r in requests}
        self._waiting = deque(
            r for r in self._waiting if r.request_id not in promote_ids
        )
        for req in requests:
            req.state = RequestState.RUNNING
            self._running.append(req)

    # ------------------------------------------------------------------
    # Demote running → waiting (preemption)
    # ------------------------------------------------------------------

    def preempt(self, request: Request, front: bool = True) -> None:
        """Move a running request back to waiting (preemption / swap-out)."""
        self._running = [r for r in self._running if r.request_id != request.request_id]
        request.state = RequestState.WAITING
        if front:
            self._waiting.appendleft(request)
        else:
            self._waiting.append(request)
        logger.info("Preempted %s", request.request_id)

    # ------------------------------------------------------------------
    # Mark finished
    # ------------------------------------------------------------------

    def mark_finished(self, finished_ids: list[str]) -> list[Request]:
        """
        Move requests with the given IDs from running → finished.
        Returns the list of newly finished requests.
        """
        done: list[Request] = []
        still_running: list[Request] = []
        for req in self._running:
            if req.request_id in finished_ids:
                req.state = RequestState.FINISHED
                req.finish_time = time.monotonic()
                self._finished.append(req)
                done.append(req)
            else:
                still_running.append(req)
        self._running = still_running
        return done

    # ------------------------------------------------------------------
    # Drain finished
    # ------------------------------------------------------------------

    def drain_finished(self) -> list[Request]:
        """Return and clear the finished list."""
        done = list(self._finished)
        self._finished.clear()
        # Remove from id_map so the same ID can be reused (if desired)
        for req in done:
            self._id_map.pop(req.request_id, None)
        return done

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def waiting_snapshot(self) -> list[Request]:
        return list(self._waiting)

    def running_snapshot(self) -> list[Request]:
        return list(self._running)

    def get_request(self, request_id: str) -> Request | None:
        return self._id_map.get(request_id)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_finished(self) -> int:
        return len(self._finished)

    def __repr__(self) -> str:
        return (
            f"QueueManager(waiting={self.num_waiting}, "
            f"running={self.num_running}, "
            f"finished={self.num_finished})"
        )