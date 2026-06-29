"""
static_scheduler.py — Static batching baseline scheduler.
Owner: Tarun (Member 3 — Scheduler Engineer)

Static batching:
  Collect up to `max_batch_size` requests, run them ALL to completion
  (every request generates tokens until its stop condition), then start
  the next batch.  No request is added mid-flight.

This is the classic "offline batching" approach, used here as the
baseline for Member 5's benchmark comparisons.
"""

from __future__ import annotations

import logging
import time

from src.common.request import Request, RequestState
from src.scheduler.base_scheduler import BaseScheduler
from src.scheduler.policies import fcfs_sort

logger = logging.getLogger(__name__)


class StaticBatchingScheduler(BaseScheduler):
    """
    Static (offline) batching scheduler.

    Lifecycle per batch:
      1. Wait until the running set is empty.
      2. Pull up to `max_batch_size` requests from the waiting queue.
      3. Allocate KV blocks for all of them up front.
      4. Run until every request in the batch is finished.
      5. Repeat.

    If KV allocation fails for a request during batch assembly, that
    request is skipped and left in the waiting queue (no preemption).
    """

    def __init__(
        self,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 4096,
        max_batch_size: int = 32,
    ) -> None:
        super().__init__(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_batch_size=max_batch_size,
        )
        self._batch_start_time: float | None = None

    def name(self) -> str:
        return "StaticBatchingScheduler"

    # ------------------------------------------------------------------
    # Core scheduling loop
    # ------------------------------------------------------------------

    def schedule(self, kv_allocator) -> list[Request]:
        """
        Return the current running batch.  If the batch is empty (all
        finished or first call), assemble a new one from the waiting queue.

        Args:
            kv_allocator: BlockAllocator from Member 4.

        Returns:
            List of requests in the active batch.
        """
        # ---- Step 1: if running batch is not empty, just keep going --------
        if self._running:
            # Allocate additional KV blocks for requests that crossed a
            # block boundary (static scheduler still needs this each step).
            for req in self._running:
                if kv_allocator.can_allocate(req):
                    # allocate only if the request needs a new block
                    # (allocator is idempotent / no-ops if nothing needed)
                    kv_allocator.allocate(req)
            return list(self._running)

        # ---- Step 2: batch is empty — assemble next batch ------------------
        if not self._waiting:
            return []

        sorted_waiting = fcfs_sort(self._waiting)
        new_batch: list[Request] = []
        failed: list[Request] = []
        batched_tokens = 0

        for req in sorted_waiting:
            if len(new_batch) >= self.max_batch_size:
                break
            if batched_tokens + req.num_prompt_tokens > self.max_num_batched_tokens:
                continue  # static scheduler can skip non-fitting requests

            if not kv_allocator.can_allocate(req):
                logger.warning(
                    "Static scheduler: skipping %s — no KV blocks",
                    req.request_id,
                )
                failed.append(req)
                continue

            kv_allocator.allocate(req)
            req.state = RequestState.RUNNING
            new_batch.append(req)
            batched_tokens += req.num_prompt_tokens

        # Remove admitted requests from waiting
        admitted_ids = {r.request_id for r in new_batch}
        self._waiting = [r for r in self._waiting if r.request_id not in admitted_ids]
        self._running = new_batch
        self._batch_start_time = time.monotonic()

        self.metrics.record_admission(
            admitted=len(new_batch),
            preempted=0,
            waiting=len(self._waiting),
            running=len(self._running),
        )

        logger.info(
            "[%s] New batch assembled: size=%d, waiting=%d",
            self.name(),
            len(self._running),
            len(self._waiting),
        )

        return list(self._running)