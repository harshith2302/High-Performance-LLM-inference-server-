"""
smoke_test_contracts.py — proves the common/ contracts + dummies form a working
loop. Run from the repo root:  PYTHONPATH=src python scripts/smoke_test_contracts.py
"""
from common import (
    Request, SamplingParams,
    DummyEngine, DummyScheduler, DummyKVCacheManager,
    Engine, Scheduler, KVCacheManager,
)


def main() -> None:
    kv: KVCacheManager = DummyKVCacheManager()
    engine: Engine = DummyEngine()
    scheduler: Scheduler = DummyScheduler(max_batch_size=4)

    for i in range(3):
        scheduler.add_request(
            Request(prompt=f"hello {i}", sampling=SamplingParams(max_tokens=5))
        )

    steps = 0
    while scheduler.has_unfinished():
        out = scheduler.step()
        for req in out.scheduled:
            if kv.get_block_table(req.request_id) == []:
                kv.allocate(req)
        result = engine.step(out.scheduled)
        for rid in result.new_token_ids:
            kv.append_token(rid)
        for rid in result.finished_ids:
            kv.free(rid)
        steps += 1

    print(f"All requests finished after {steps} steps.")
    print(f"Free blocks remaining: {kv.num_free_blocks()}/{kv.num_total_blocks()}")
    # runtime_checkable protocols: confirm dummies satisfy the contracts
    assert isinstance(engine, Engine)
    assert isinstance(scheduler, Scheduler)
    assert isinstance(kv, KVCacheManager)
    print("Protocol conformance checks passed.")


if __name__ == "__main__":
    main()
