"""
smoke_test_lifecycle.py — proves the async/sync bridge works end-to-end against
the dummy components. Run from repo root:
    set PYTHONPATH=src   (Windows)
    python scripts\smoke_test_lifecycle.py
"""
import asyncio

from common import (
    Request, SamplingParams,
    DummyEngine, DummyScheduler, DummyKVCacheManager,
)
from server.lifecycle import EngineWorker


async def main() -> None:
    worker = EngineWorker(
        engine=DummyEngine(),
        scheduler=DummyScheduler(max_batch_size=4),
        kv_cache=DummyKVCacheManager(),
    )
    worker.start()

    # Fire 3 concurrent requests, like 3 HTTP handlers awaiting their result.
    async def one(i: int):
        req = Request(prompt=f"hello {i}", sampling=SamplingParams(max_tokens=5))
        future = worker.submit(req)
        result = await future
        print(f"request {i} finished: {len(result.output_tokens)} tokens, reason={result.finish_reason}")

    await asyncio.gather(one(0), one(1), one(2))
    worker.stop()
    print("Bridge works: all handlers were woken with their results.")


if __name__ == "__main__":
    asyncio.run(main())
