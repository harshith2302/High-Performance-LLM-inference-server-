"""
policies.py — Scheduling policies (FCFS, admission guards).
Owner: Tarun (Member 3 — Scheduler Engineer)
"""

from __future__ import annotations

import time
from typing import Callable

from src.common.request import Request


# ---------------------------------------------------------------------------
# Sorting policies
# ---------------------------------------------------------------------------

def fcfs_sort(requests: list[Request]) -> list[Request]:
    """
    First-Come-First-Served: sort by arrival_time ascending.
    Requests with no arrival_time set are placed last.
    """
    return sorted(
        requests,
        key=lambda r: r.arrival_time if r.arrival_time is not None else float("inf"),
    )


def shortest_job_first_sort(requests: list[Request]) -> list[Request]:
    """
    Shortest Job First: prefer requests with shorter prompt lengths.
    Use with caution — can starve long requests.
    """
    return sorted(requests, key=lambda r: r.num_prompt_tokens)


# ---------------------------------------------------------------------------
# Admission guards
# ---------------------------------------------------------------------------

def token_budget_admits(
    candidate: Request,
    current_batch: list[Request],
    max_num_batched_tokens: int,
) -> bool:
    """
    True if admitting `candidate` keeps total batched tokens within budget.
    """
    current_tokens = sum(r.num_prompt_tokens for r in current_batch)
    return current_tokens + candidate.num_prompt_tokens <= max_num_batched_tokens


def seq_limit_admits(
    current_batch: list[Request],
    max_num_seqs: int,
    max_batch_size: int,
) -> bool:
    """
    True if there is still room for at least one more sequence.
    """
    n = len(current_batch)
    return n < max_num_seqs and n < max_batch_size


# ---------------------------------------------------------------------------
# Admission policy factory (convenience)
# ---------------------------------------------------------------------------

AdmissionPolicy = Callable[[Request, list[Request]], bool]


def make_fcfs_policy(
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_batch_size: int,
    kv_allocator,
) -> AdmissionPolicy:
    """
    Returns a single callable that combines all admission guards.

    Usage::

        admit = make_fcfs_policy(...)
        for req in fcfs_sort(waiting):
            if not admit(req, running):
                break
            running.append(req)
    """
    def _admit(candidate: Request, current_batch: list[Request]) -> bool:
        if not seq_limit_admits(current_batch, max_num_seqs, max_batch_size):
            return False
        if not token_budget_admits(candidate, current_batch, max_num_batched_tokens):
            return False
        if not kv_allocator.can_allocate(candidate):
            return False
        return True

    return _admit