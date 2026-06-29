"""
continuous_scheduler.py — Continuous batching scheduler.
Owner: Tarun (Member 3 — Scheduler Engineer)

Design choice — Prefill vs Decode policy
-----------------------------------------
We use **pure interleaving**: on every step, we first try to admit new
waiting requests (prefill), then run all currently-running requests
(decode) together in the same forward pass.

Rationale:
- Maximises GPU utilisation by keeping the batch full at all times.
- Avoids "prefill stall" (where long prefills block decode progress)
  because prefill is bounded by max_num_batched_tokens.
- Simpler to implement correctly than chunked-prefill while still
  delivering most of the throughput benefit.

Stretch goal (preemption) is implemented as swap-out: if the KV
allocator cannot give a new block to a running request, that request
is moved back to the front of the waiting queue.
"""

from __future__ import annotations

import logging
from collections import deque

from src.common.request import Request, RequestState
from src.scheduler.base_scheduler import BaseScheduler
from src.scheduler.policies import fcfs_sort

logger = logging.getLogger(__name__)


class ContinuousBatchingScheduler(BaseScheduler):
    """
    Continuous-batching scheduler.

    Every step:
      1. Remove requests finished by the engine (via on_step_end).
      2. Admit waiting requests into the running set subject to:
         - max_batch_size
         - max_num_batched_tokens
         - KV block availability (at least one free block per new request)
      3. For running requests that need a new KV block this step,
         allocate it; preempt (swap-out) any that can't be served.
      4. Return the running set to the engine.
    """

    def __init__(
        self,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 4096,
        max_batch_size: int = 32,
        enable_preemption: bool = True,
        block_size: int = 16,
    ) -> None:
        super().__init__(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_batch_size=max_batch_size,
        )
        self.enable_preemption = enable_preemption
        self.block_size = block_size

        # Use a deque for O(1) front-insertion (preempted requests go back here)
        self._waiting: deque[Request] = deque()

    def name(self) -> str:
        return "ContinuousBatchingScheduler"

    def add_request(self, request: Request) -> None:
        """Enqueue a new request (FCFS order)."""
        import time
        request.state = RequestState.WAITING
        request.arrival_time = request.arrival_time or time.monotonic()
        self._waiting.append(request)

    # ------------------------------------------------------------------
    # Core scheduling loop
    # ------------------------------------------------------------------

    def schedule(self, kv_allocator) -> list[Request]:
        """
        Build the batch for the next engine step.

        Args:
            kv_allocator: BlockAllocator from Member 4 with:
                - can_allocate(request) -> bool
                - allocate(request) -> None
                - free(request) -> None

        Returns:
            List of requests to run this step.
        """
        # ---- Step 1: handle any newly-finished requests --------------------
        # (on_step_end already moved them out of self._running)

        # ---- Step 2: allocate new blocks for running requests that crossed
        #              a block boundary this decode step -----------------------
        preempted: list[Request] = []
        for req in list(self._running):
            if self._needs_new_block(req):
                if kv_allocator.can_allocate(req):
                    kv_allocator.allocate(req)
                elif self.enable_preemption:
                    logger.warning(
                        "Preempting request %s (no free KV blocks)", req.request_id
                    )
                    kv_allocator.free(req)
                    req.state = RequestState.WAITING
                    self._running.remove(req)
                    preempted.append(req)
                # else: leave in running; engine will stall — shouldn't happen
                #       if max_batch_size is set conservatively.

        # Preempted requests go to the front of the waiting queue (priority)
        for req in reversed(preempted):
            self._waiting.appendleft(req)

        # ---- Step 3: admit new requests from the waiting queue -------------
        # Sort waiting by FCFS (arrival_time)
        sorted_waiting = fcfs_sort(list(self._waiting))

        batched_tokens = sum(r.num_prompt_tokens for r in self._running)
        new_running: list[Request] = []

        for req in sorted_waiting:
            if len(self._running) + len(new_running) >= self.max_batch_size:
                break
            if len(self._running) + len(new_running) >= self.max_num_seqs:
                break

            req_tokens = req.num_prompt_tokens  # prefill cost
            if batched_tokens + req_tokens > self.max_num_batched_tokens:
                break  # FCFS — don't skip ahead

            if not kv_allocator.can_allocate(req):
                # Can't serve even one block — stop admitting
                logger.debug(
                    "Admission blocked for %s: no KV blocks available",
                    req.request_id,
                )
                break

            kv_allocator.allocate(req)
            req.state = RequestState.RUNNING
            new_running.append(req)
            batched_tokens += req_tokens

        # Remove admitted requests from the waiting deque
        admitted_ids = {r.request_id for r in new_running}
        self._waiting = deque(r for r in self._waiting if r.request_id not in admitted_ids)
        self._running.extend(new_running)

        # ---- Step 4: emit metrics snapshot ---------------------------------
        self.metrics.record_admission(
            admitted=len(new_running),
            preempted=len(preempted),
            waiting=len(self._waiting),
            running=len(self._running),
        )

        logger.debug(
            "[%s] running=%d waiting=%d admitted=%d preempted=%d",
            self.name(),
            len(self._running),
            len(self._waiting),
            len(new_running),
            len(preempted),
        )

        return list(self._running)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _needs_new_block(self, req: Request) -> bool:
        """
        True when the request has just crossed into a new KV block
        (i.e. its total token count is an exact multiple of block_size).
        The engine increments req.num_generated_tokens before we check.
        """
        total = req.num_prompt_tokens + req.num_generated_tokens
        return total > 0 and total % self.block_size == 0