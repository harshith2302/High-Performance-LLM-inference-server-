"""
common — shared contracts for the LLM inference server.

OWNED BY: Ajay (M1). This package is the single source of truth for how the
four components talk to each other. Everyone imports from here; nobody imports
another member's concrete class directly.

Convenience re-exports so teammates can write:

    from common import Request, SamplingParams, Engine, Scheduler, KVCacheManager
"""

from common.sampling_params import SamplingParams
from common.request import Request, RequestState
from common.interfaces import (
    Engine,
    Scheduler,
    KVCacheManager,
    StepOutput,
    SchedulerOutput,
)
from common.stubs import (
    DummyEngine,
    DummyScheduler,
    DummyKVCacheManager,
)

__all__ = [
    "SamplingParams",
    "Request",
    "RequestState",
    "Engine",
    "Scheduler",
    "KVCacheManager",
    "StepOutput",
    "SchedulerOutput",
    "DummyEngine",
    "DummyScheduler",
    "DummyKVCacheManager",
]
