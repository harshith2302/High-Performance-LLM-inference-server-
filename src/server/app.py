"""
app.py — the FastAPI application: all HTTP endpoints.

OWNED BY: Ajay (M1).

Endpoints:
    POST /generate         -> send a prompt, get the full generated text back
    POST /generate/stream  -> stream tokens as they are produced (SSE)
    GET  /healthz          -> liveness check
    GET  /stats            -> live system metrics

The app holds one EngineWorker. On startup it builds the worker from the
Dummy* stubs so the server runs end-to-end TODAY. When Harshith/Tarun/Pavan
finish their real components, swap them in inside `build_worker()` — nothing
else in this file changes.

Run it with:  python -m server.main   (see main.py)
"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from common.request import Request
from common.sampling_params import SamplingParams
from common.stubs import DummyEngine, DummyScheduler, DummyKVCacheManager
from server.api_models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    StatsResponse,
)
from server.lifecycle import EngineWorker


# Components are swapped here when the real ones land. Today: dummies.
def build_worker() -> EngineWorker:
    return EngineWorker(
        engine=DummyEngine(),
        scheduler=DummyScheduler(max_batch_size=8),
        kv_cache=DummyKVCacheManager(total_blocks=1000),
    )


app = FastAPI(title="LLM Inference Server")

# Module-level handle so endpoints can reach the worker and its parts.
worker: EngineWorker | None = None
_kv = None          # kept for /stats
_scheduler = None   # kept for /stats


@app.on_event("startup")
async def _startup() -> None:
    """Build and start the worker inside the running event loop."""
    global worker, _kv, _scheduler
    w = build_worker()
    # Stash the pieces /stats reads. (build_worker hides them, so rebuild refs.)
    _kv = w._kv            # noqa: SLF001 - intentional internal access for metrics
    _scheduler = w._scheduler  # noqa: SLF001
    w.start()
    worker = w


@app.on_event("shutdown")
async def _shutdown() -> None:
    if worker is not None:
        worker.stop()


def _to_sampling(body: GenerateRequest) -> SamplingParams:
    return SamplingParams(
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        stop=body.stop,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    """Submit a prompt, await the Future, return the finished result."""
    assert worker is not None
    req = Request(prompt=body.prompt, sampling=_to_sampling(body))
    future = worker.submit(req)
    result = await future
    # NOTE: real text needs the tokenizer (Harshith). For now we echo token ids.
    text = f"<{len(result.output_tokens)} tokens generated>"
    return GenerateResponse(
        request_id=result.request_id,
        text=text,
        output_tokens=result.output_tokens,
        finish_reason=result.finish_reason,
    )


@app.post("/generate/stream")
async def generate_stream(body: GenerateRequest) -> StreamingResponse:
    """Stream tokens as Server-Sent Events.

    Today the dummy engine produces all tokens then finishes, so we await the
    result and emit them. When the real engine streams per-step, this becomes a
    live per-token feed (the structure is already here)."""
    assert worker is not None
    req = Request(prompt=body.prompt, sampling=_to_sampling(body))
    future = worker.submit(req)

    async def event_stream():
        result = await future
        for tok in result.output_tokens:
            yield f"data: {json.dumps({'token': tok})}\n\n"
            await asyncio.sleep(0)  # let the loop breathe between sends
        yield f"data: {json.dumps({'done': True, 'finish_reason': result.finish_reason})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Live metrics. Reads queue depth from the scheduler and block usage
    from the KV cache."""
    sched_out = _scheduler.step() if _scheduler is not None else None
    num_waiting = sched_out.num_waiting if sched_out else 0
    num_running = sched_out.num_running if sched_out else 0
    return StatsResponse(
        num_waiting=num_waiting,
        num_running=num_running,
        free_blocks=_kv.num_free_blocks() if _kv else 0,
        total_blocks=_kv.num_total_blocks() if _kv else 0,
    )
