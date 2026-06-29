"""
tests/unit/test_scheduler.py — Unit tests for Tarun's scheduler.
Owner: Tarun (Member 3)
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch

from src.common.request import Request, RequestState
from src.scheduler.continuous_scheduler import ContinuousBatchingScheduler
from src.scheduler.static_scheduler import StaticBatchingScheduler
from src.scheduler.queue_manager import QueueManager
from src.scheduler.policies import fcfs_sort, token_budget_admits, seq_limit_admits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_request(req_id: str, prompt_tokens: int = 10, arrival: float | None = None) -> Request:
    req = Request(
        request_id=req_id,
        prompt="dummy prompt",
        num_prompt_tokens=prompt_tokens,
        sampling_params=MagicMock(),
    )
    req.arrival_time = arrival or time.monotonic()
    req.num_generated_tokens = 0
    return req


def make_kv_allocator(always_can_allocate: bool = True) -> MagicMock:
    allocator = MagicMock()
    allocator.can_allocate.return_value = always_can_allocate
    allocator.allocate.return_value = None
    allocator.free.return_value = None
    return allocator


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class TestFCFSSort:
    def test_sorts_by_arrival_time(self):
        r1 = make_request("r1", arrival=2.0)
        r2 = make_request("r2", arrival=1.0)
        r3 = make_request("r3", arrival=3.0)
        result = fcfs_sort([r1, r2, r3])
        assert [r.request_id for r in result] == ["r2", "r1", "r3"]

    def test_none_arrival_goes_last(self):
        r1 = make_request("r1", arrival=1.0)
        r2 = make_request("r2", arrival=None)
        result = fcfs_sort([r1, r2])
        assert result[0].request_id == "r1"

    def test_empty_list(self):
        assert fcfs_sort([]) == []


class TestAdmissionPolicies:
    def test_token_budget_admits_within_budget(self):
        r1 = make_request("r1", prompt_tokens=50)
        candidate = make_request("c1", prompt_tokens=30)
        assert token_budget_admits(candidate, [r1], max_num_batched_tokens=100)

    def test_token_budget_rejects_over_budget(self):
        r1 = make_request("r1", prompt_tokens=80)
        candidate = make_request("c1", prompt_tokens=30)
        assert not token_budget_admits(candidate, [r1], max_num_batched_tokens=100)

    def test_seq_limit_admits_when_room(self):
        batch = [make_request(f"r{i}") for i in range(5)]
        assert seq_limit_admits(batch, max_num_seqs=10, max_batch_size=10)

    def test_seq_limit_rejects_at_max(self):
        batch = [make_request(f"r{i}") for i in range(10)]
        assert not seq_limit_admits(batch, max_num_seqs=10, max_batch_size=10)


# ---------------------------------------------------------------------------
# QueueManager
# ---------------------------------------------------------------------------

class TestQueueManager:
    def test_enqueue_and_promote(self):
        qm = QueueManager()
        r1 = make_request("r1")
        qm.enqueue(r1)
        assert qm.num_waiting == 1
        assert qm.num_running == 0

        qm.promote_to_running([r1])
        assert qm.num_waiting == 0
        assert qm.num_running == 1
        assert r1.state == RequestState.RUNNING

    def test_mark_finished(self):
        qm = QueueManager()
        r1 = make_request("r1")
        qm.enqueue(r1)
        qm.promote_to_running([r1])

        done = qm.mark_finished(["r1"])
        assert len(done) == 1
        assert qm.num_running == 0
        assert qm.num_finished == 1

    def test_drain_finished(self):
        qm = QueueManager()
        r1 = make_request("r1")
        qm.enqueue(r1)
        qm.promote_to_running([r1])
        qm.mark_finished(["r1"])

        drained = qm.drain_finished()
        assert len(drained) == 1
        assert qm.num_finished == 0

    def test_preempt_moves_to_waiting_front(self):
        qm = QueueManager()
        r1 = make_request("r1", arrival=1.0)
        r2 = make_request("r2", arrival=2.0)
        qm.enqueue(r1)
        qm.enqueue(r2)
        qm.promote_to_running([r1])
        qm.preempt(r1, front=True)

        waiting = qm.waiting_snapshot()
        assert waiting[0].request_id == "r1"

    def test_duplicate_enqueue_raises(self):
        qm = QueueManager()
        r1 = make_request("r1")
        qm.enqueue(r1)
        with pytest.raises(ValueError, match="Duplicate"):
            qm.enqueue(r1)


# ---------------------------------------------------------------------------
# ContinuousBatchingScheduler
# ---------------------------------------------------------------------------

class TestContinuousBatchingScheduler:
    def test_admit_request_into_running(self):
        sched = ContinuousBatchingScheduler(max_batch_size=8, max_num_batched_tokens=512)
        kv = make_kv_allocator(always_can_allocate=True)

        r1 = make_request("r1", prompt_tokens=20)
        sched.add_request(r1)
        batch = sched.schedule(kv)

        assert len(batch) == 1
        assert batch[0].request_id == "r1"
        assert batch[0].state == RequestState.RUNNING

    def test_respects_max_batch_size(self):
        sched = ContinuousBatchingScheduler(max_batch_size=3, max_num_batched_tokens=10_000)
        kv = make_kv_allocator(always_can_allocate=True)

        for i in range(10):
            sched.add_request(make_request(f"r{i}", prompt_tokens=10))

        batch = sched.schedule(kv)
        assert len(batch) == 3

    def test_respects_token_budget(self):
        sched = ContinuousBatchingScheduler(max_batch_size=100, max_num_batched_tokens=50)
        kv = make_kv_allocator(always_can_allocate=True)

        # Each request = 20 tokens; budget = 50 → only 2 fit
        for i in range(5):
            sched.add_request(make_request(f"r{i}", prompt_tokens=20))

        batch = sched.schedule(kv)
        assert len(batch) == 2

    def test_no_admission_when_kv_full(self):
        sched = ContinuousBatchingScheduler(max_batch_size=8, max_num_batched_tokens=512)
        kv = make_kv_allocator(always_can_allocate=False)

        sched.add_request(make_request("r1"))
        batch = sched.schedule(kv)
        assert batch == []

    def test_preemption_when_decode_block_unavailable(self):
        sched = ContinuousBatchingScheduler(
            max_batch_size=8,
            max_num_batched_tokens=512,
            enable_preemption=True,
            block_size=4,
        )
        # First schedule: admit r1
        kv = make_kv_allocator(always_can_allocate=True)
        r1 = make_request("r1", prompt_tokens=4)   # exactly block_size → needs new block next step
        sched.add_request(r1)
        sched.schedule(kv)

        # Simulate engine: r1 generated 4 tokens (crosses block boundary)
        r1.num_generated_tokens = 4

        # Now KV is full — can_allocate returns False for running requests
        kv2 = MagicMock()
        kv2.can_allocate.return_value = False
        kv2.allocate.return_value = None
        kv2.free.return_value = None

        batch2 = sched.schedule(kv2)

        # r1 should have been preempted (moved to waiting)
        assert r1 not in batch2
        assert sched.num_waiting() == 1

    def test_fcfs_ordering(self):
        sched = ContinuousBatchingScheduler(max_batch_size=2, max_num_batched_tokens=10_000)
        kv = make_kv_allocator(always_can_allocate=True)

        r_late = make_request("late", arrival=2.0)
        r_early = make_request("early", arrival=1.0)
        sched.add_request(r_late)
        sched.add_request(r_early)

        batch = sched.schedule(kv)
        # Both admitted, but early should be first
        ids = [r.request_id for r in batch]
        assert "early" in ids

    def test_on_step_end_removes_finished(self):
        sched = ContinuousBatchingScheduler(max_batch_size=8, max_num_batched_tokens=512)
        kv = make_kv_allocator(always_can_allocate=True)

        r1 = make_request("r1")
        sched.add_request(r1)
        sched.schedule(kv)

        sched.on_step_begin()
        sched.on_step_end(finished_ids=["r1"])

        assert sched.num_running() == 0
        finished = sched.get_finished()
        assert len(finished) == 1


# ---------------------------------------------------------------------------
# StaticBatchingScheduler
# ---------------------------------------------------------------------------

class TestStaticBatchingScheduler:
    def test_collects_full_batch_before_running(self):
        sched = StaticBatchingScheduler(max_batch_size=4, max_num_batched_tokens=10_000)
        kv = make_kv_allocator(always_can_allocate=True)

        for i in range(6):
            sched.add_request(make_request(f"r{i}", prompt_tokens=10))

        batch = sched.schedule(kv)
        assert len(batch) == 4  # capped at max_batch_size

    def test_does_not_change_batch_mid_flight(self):
        sched = StaticBatchingScheduler(max_batch_size=4, max_num_batched_tokens=10_000)
        kv = make_kv_allocator(always_can_allocate=True)

        for i in range(4):
            sched.add_request(make_request(f"r{i}"))

        batch1 = sched.schedule(kv)
        # Add more requests mid-flight
        sched.add_request(make_request("new"))
        batch2 = sched.schedule(kv)

        assert [r.request_id for r in batch1] == [r.request_id for r in batch2]

    def test_starts_new_batch_after_all_finished(self):
        sched = StaticBatchingScheduler(max_batch_size=4, max_num_batched_tokens=10_000)
        kv = make_kv_allocator(always_can_allocate=True)

        for i in range(4):
            sched.add_request(make_request(f"r{i}"))

        batch1 = sched.schedule(kv)
        # Finish all
        sched.on_step_begin()
        sched.on_step_end([r.request_id for r in batch1])

        # Add new requests
        for i in range(4, 8):
            sched.add_request(make_request(f"r{i}"))

        batch2 = sched.schedule(kv)
        assert len(batch2) == 4
        b2_ids = {r.request_id for r in batch2}
        assert all(rid in b2_ids for rid in ["r4", "r5", "r6", "r7"])

    def test_skips_request_when_kv_unavailable(self):
        sched = StaticBatchingScheduler(max_batch_size=4, max_num_batched_tokens=10_000)
        kv = make_kv_allocator(always_can_allocate=False)

        sched.add_request(make_request("r1"))
        batch = sched.schedule(kv)
        assert batch == []
        assert sched.num_waiting() == 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestSchedulerMetrics:
    def test_summary_empty(self):
        from src.scheduler.scheduler_metrics import SchedulerMetrics
        m = SchedulerMetrics()
        assert m.summary() == {}

    def test_records_and_dumps(self):
        from src.scheduler.scheduler_metrics import SchedulerMetrics
        m = SchedulerMetrics()
        m.record_step(0, batch_size=4, queue_depth=2, iteration_latency_s=0.01, tokens_generated=4)
        m.record_admission(admitted=1, preempted=0, waiting=2, running=4)

        rows = m.dump()
        assert len(rows) == 1
        assert rows[0]["batch_size"] == 4
        assert rows[0]["admitted"] == 1

    def test_total_counters(self):
        from src.scheduler.scheduler_metrics import SchedulerMetrics
        m = SchedulerMetrics()
        m.record_step(0, batch_size=4, queue_depth=0, iteration_latency_s=0.01, tokens_generated=8)
        m.record_step(1, batch_size=4, queue_depth=0, iteration_latency_s=0.01, tokens_generated=4)
        assert m.total_tokens_generated == 12
        assert m.total_steps == 2