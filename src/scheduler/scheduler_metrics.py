"""
scheduler_metrics.py — Per-step scheduler metrics.
Owner: Tarun (Member 3 — Scheduler Engineer)

Emits metrics that Member 5 (Jay / benchmarks) consumes:
  - active_batch_size
  - queue_depth
  - iteration_latency_s
  - tokens_generated (per step)
  - admitted_per_step
  - preempted_per_step
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StepMetric:
    step_index: int
    timestamp: float
    batch_size: int
    queue_depth: int
    iteration_latency_s: float
    tokens_generated: int
    admitted: int = 0
    preempted: int = 0


class SchedulerMetrics:
    """
    Lightweight in-memory metrics store for the scheduler.

    Usage (inside the scheduler)::

        self.metrics.record_step(step_index=i, batch_size=32, ...)
        self.metrics.record_admission(admitted=4, preempted=1, ...)

    Member 5 can call `metrics.dump()` to get all recorded steps as
    a list of dicts suitable for pandas or CSV export.
    """

    def __init__(self, max_history: int = 10_000) -> None:
        self._steps: list[StepMetric] = []
        self._pending_admission: dict[str, int] = {}
        self._max_history = max_history

        # Running totals
        self.total_tokens_generated: int = 0
        self.total_requests_admitted: int = 0
        self.total_requests_preempted: int = 0
        self.total_steps: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        step_index: int,
        batch_size: int,
        queue_depth: int,
        iteration_latency_s: float,
        tokens_generated: int,
    ) -> None:
        metric = StepMetric(
            step_index=step_index,
            timestamp=time.monotonic(),
            batch_size=batch_size,
            queue_depth=queue_depth,
            iteration_latency_s=iteration_latency_s,
            tokens_generated=tokens_generated,
        )
        if len(self._steps) >= self._max_history:
            self._steps.pop(0)
        self._steps.append(metric)

        self.total_tokens_generated += tokens_generated
        self.total_steps += 1

        logger.debug(
            "step=%d  batch=%d  queue=%d  lat=%.4fs  toks=%d",
            step_index,
            batch_size,
            queue_depth,
            iteration_latency_s,
            tokens_generated,
        )

    def record_admission(
        self,
        admitted: int,
        preempted: int,
        waiting: int,
        running: int,
    ) -> None:
        """Called by the scheduler after each admission pass."""
        self.total_requests_admitted += admitted
        self.total_requests_preempted += preempted

        # Patch the most recent step if it exists
        if self._steps:
            self._steps[-1].admitted = admitted
            self._steps[-1].preempted = preempted

        if admitted or preempted:
            logger.info(
                "Admission: +%d admitted, -%d preempted | running=%d waiting=%d",
                admitted,
                preempted,
                running,
                waiting,
            )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def dump(self) -> list[dict]:
        """Return all recorded steps as a list of dicts (for Jay's analysis)."""
        return [
            {
                "step_index": s.step_index,
                "timestamp": s.timestamp,
                "batch_size": s.batch_size,
                "queue_depth": s.queue_depth,
                "iteration_latency_s": s.iteration_latency_s,
                "tokens_generated": s.tokens_generated,
                "admitted": s.admitted,
                "preempted": s.preempted,
            }
            for s in self._steps
        ]

    def summary(self) -> dict:
        """High-level summary stats."""
        if not self._steps:
            return {}

        latencies = [s.iteration_latency_s for s in self._steps]
        batch_sizes = [s.batch_size for s in self._steps]

        return {
            "total_steps": self.total_steps,
            "total_tokens_generated": self.total_tokens_generated,
            "total_requests_admitted": self.total_requests_admitted,
            "total_requests_preempted": self.total_requests_preempted,
            "avg_iteration_latency_s": sum(latencies) / len(latencies),
            "max_iteration_latency_s": max(latencies),
            "avg_batch_size": sum(batch_sizes) / len(batch_sizes),
            "max_batch_size": max(batch_sizes),
        }

    def reset(self) -> None:
        self._steps.clear()
        self.total_tokens_generated = 0
        self.total_requests_admitted = 0
        self.total_requests_preempted = 0
        self.total_steps = 0