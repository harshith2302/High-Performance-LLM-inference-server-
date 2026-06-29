"""
lifecycle.py — the async/sync bridge. THE HEART OF M1'S COMPONENT.

OWNED BY: Ajay (M1).

THE PROBLEM THIS SOLVES:
    - The engine wants to run in ONE background thread, looping forever:
      ask scheduler what to run -> run one engine step -> repeat.
    - HTTP handlers are ASYNC and there can be hundreds waiting at once.
    - These two worlds can't touch each other's objects safely.

THE SOLUTION (what this file does):
    1. An HTTP handler calls `submit(request)`. That creates an asyncio.Future,
       hands the request to the scheduler, and returns the Future.
    2. The handler `await`s the Future -> it "falls asleep" without blocking
       the event loop, so other requests keep being served.
    3. A background worker thread runs the engine loop. When a request finishes,
       the worker resolves that request's Future using `call_soon_threadsafe`
       (the ONLY safe way for a thread to wake the async world).
    4. The sleeping handler wakes up, gets the result, and replies.

This file is built to run against the Dummy* stubs TODAY. When Harshith/Tarun/
Pavan finish their real components, swap them in at construction — nothing here
changes.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

from common.request import Request, RequestState
from common.interfaces import Engine, Scheduler, KVCacheManager


@dataclass
class GenerationResult:
    """What a finished request hands back to its HTTP handler."""
    request_id: str
    output_tokens: list[int]
    finish_reason: str | None


class EngineWorker:
    """
    Drives the scheduler + engine + KV cache on a single background thread,
    and bridges results back to async HTTP handlers via per-request Futures.
    """

    def __init__(
        self,
        engine: Engine,
        scheduler: Scheduler,
        kv_cache: KVCacheManager,
        idle_sleep: float = 0.005,
    ) -> None:
        self._engine = engine
        self._scheduler = scheduler
        self._kv = kv_cache
        self._idle_sleep = idle_sleep      # nap when there's no work, so we don't busy-spin

        # The asyncio loop the HTTP server runs on. Captured when we start.
        self._loop: asyncio.AbstractEventLoop | None = None

        # request_id -> the Future its handler is awaiting.
        self._futures: dict[str, asyncio.Future] = {}

        # request_id -> the live Request object (so we can read final tokens).
        self._requests: dict[str, Request] = {}

        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()      # guards the dicts (touched from both worlds)

    # ----- called from the ASYNC world (HTTP handlers) -----------------------

    def start(self) -> None:
        """Capture the event loop and launch the worker thread.
        Call this once at server startup, from inside the async loop."""
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="engine-worker")
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker thread to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def submit(self, request: Request) -> asyncio.Future:
        """Register a request and return a Future its handler can await.
        Called from an async handler, so we're on the loop thread here."""
        future: asyncio.Future = self._loop.create_future()  # type: ignore[union-attr]
        with self._lock:
            self._futures[request.request_id] = future
            self._requests[request.request_id] = request
        self._scheduler.add_request(request)
        return future

    # ----- the BACKGROUND THREAD ---------------------------------------------

    def _run_loop(self) -> None:
        """The engine loop. Runs forever on its own thread."""
        while self._running:
            out = self._scheduler.step()

            if not out.scheduled:
                time.sleep(self._idle_sleep)   # nothing to do; don't burn CPU
                continue

            # Make sure every scheduled request has KV blocks allocated.
            for req in out.scheduled:
                if self._kv.get_block_table(req.request_id) == []:
                    if self._kv.can_allocate(req):
                        self._kv.allocate(req)

            # Run one forward pass.
            result = self._engine.step(out.scheduled)

            # Account for the new tokens in the KV cache.
            for rid in result.new_token_ids:
                self._kv.append_token(rid)

            # Wake up any handlers whose requests just finished.
            for rid in result.finished_ids:
                self._kv.free(rid)
                self._resolve(rid)

    def _resolve(self, request_id: str) -> None:
        """Hand the result back to the waiting async handler — safely."""
        with self._lock:
            future = self._futures.pop(request_id, None)
            request = self._requests.pop(request_id, None)
        if future is None or request is None or self._loop is None:
            return

        result = GenerationResult(
            request_id=request_id,
            output_tokens=list(request.output_tokens),
            finish_reason=request.finish_reason,
        )

        # THE BRIDGE: a thread cannot set a Future directly. call_soon_threadsafe
        # schedules the set_result back on the event loop thread, which wakes the
        # sleeping handler. This single line is the seam between the two worlds.
        self._loop.call_soon_threadsafe(self._set_future_result, future, result)

    @staticmethod
    def _set_future_result(future: asyncio.Future, result: GenerationResult) -> None:
        if not future.done():
            future.set_result(result)
