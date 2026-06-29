# Scheduler Design — Member 3 (Tarun)

## Overview

The scheduler is the brain of the inference server. It decides which requests run on each forward pass, enforcing resource constraints while maximising GPU utilisation. This document describes the design decisions, policies, and interfaces for the two schedulers delivered: `ContinuousBatchingScheduler` and `StaticBatchingScheduler`.

---

## File Structure

```
src/scheduler/
├── __init__.py               # Public exports
├── base_scheduler.py         # Abstract base; shared bookkeeping
├── continuous_scheduler.py   # Primary: continuous batching
├── static_scheduler.py       # Baseline: static batching
├── policies.py               # FCFSPolicy, PriorityPolicy, AdmissionChecker
├── queue_manager.py          # Waiting / running / finished sets
└── scheduler_metrics.py      # Per-step telemetry dataclass
```

---

## Scheduling Policy: Prefill-Prioritised Interleaving

### Decision

We implement **prefill-prioritised interleaving**. On every forward pass:

1. **Decode phase** — every already-running request gets one new token.
2. **Admit phase** — waiting requests are prefilled (all prompt tokens at once) into the _same_ forward pass as the decoding sequences.

### Why this policy?

| Policy | TTFT | GPU Utilisation | Implementation Complexity |
|---|---|---|---|
| Prefill-then-decode rounds | High | Low (GPU idles between rounds) | Low |
| **Prefill-prioritised interleaving (chosen)** | **Low** | **High** | **Moderate** |
| Chunked prefill | Very low | Highest | High |

Prefill-then-decode rounds stall decoding sequences while a new prefill occupies the whole GPU. Our interleaving policy overlaps the two phases: decoding sequences keep producing tokens while new arrivals are prefilled in the same step. This gives a good TTFT without the complexity of chunked prefill.

**Chunked prefill** (splitting large prompts across multiple steps) is implemented as a stretch goal via the `chunked_prefill_size` parameter in `ContinuousBatchingScheduler`.

---

## Constraint Enforcement

Every step, admission is gated by three hard constraints:

| Constraint | Parameter | Applied to |
|---|---|---|
| Maximum concurrent sequences | `max_num_seqs` | Sequence count |
| Maximum batch size | `max_batch_size` | Sequence count |
| Token budget per step | `max_num_batched_tokens` | Prefill cost = prompt length; decode cost = 1 |
| KV-block availability | `kv_allocator.can_allocate()` | Block count |

Decode sequences each cost **1 token** from the budget (they produce exactly one token per step). Prefilling a new request costs **`num_prompt_tokens`** tokens. This means a step can process, for example, 4 decoding sequences (4 tokens) plus one 60-token prompt (60 tokens) within a 128-token budget.

---

## Admission Policy

**Default: First-Come-First-Served (FCFS)**

The `FCFSPolicy` preserves the arrival order of the waiting queue. When a head-of-queue request cannot be admitted (e.g. budget exceeded by a large prompt), the scheduler _continues scanning_ for shorter requests that fit. This breaks strict FCFS but prevents head-of-line blocking by huge prompts.

Alternative policies available in `policies.py`:

- **`PriorityPolicy`** — higher `Request.priority` admitted first; ties broken by arrival time.
- **`ShortestJobFirst`** — shorter prompt admitted first; reduces average wait time but risks starvation.
- **`AdmissionChecker`** — stateless gate that encapsulates all constraint checks; used internally and available for external testing.

---

## Preemption (Stretch Goal)

When the KV-cache pool is exhausted and a running request needs a new block for its next decode token, the scheduler can **preempt** the lowest-priority running request:

1. Evict the victim from the running set.
2. Free all its KV blocks.
3. Discard its generated tokens (re-computation strategy — same as vLLM).
4. Push it to the **front** of the waiting queue so it is re-admitted before newer arrivals.

Preemption is enabled by default (`enable_preemption=True`). To disable, set `enable_preemption=False` at construction time.

**Trade-off:** Re-computation wastes compute on the victim's already-generated tokens. The alternative (swapping KV blocks to CPU) avoids this but adds PCIe transfer overhead. Re-computation is simpler and sufficient for the scope of this project.

---

## Static Batching Baseline

`StaticBatchingScheduler` implements classic static batching for Member 5's benchmark comparisons:

- Collects a fixed batch at the start of each round.
- Runs all sequences to completion before admitting any new requests.
- Sequences that finish early leave their GPU slots idle until the last sequence is done.

This deliberate inefficiency is what continuous batching corrects. Member 5 can observe the gap directly by comparing `avg_active_batch_size` and `total_steps` between the two schedulers.

---

## Interface Contract

Both schedulers implement `BaseScheduler`:

```python
scheduler.add_request(request)          # enqueue a new request
batch   = scheduler.schedule()          # get batch for next forward pass
metrics = scheduler.step(step_outputs)  # process engine outputs, emit metrics
```

### Integration with other members

| Member | Direction | Interface |
|---|---|---|
| **Member 2 (Harshith / Engine)** | Scheduler → Engine | `schedule()` returns `List[Request]`; engine calls `engine.step(batch)` |
| **Member 2 (Harshith / Engine)** | Engine → Scheduler | `List[StepOutput]` passed to `scheduler.step()` |
| **Member 4 (Pavan / KV Cache)** | Scheduler → KVAllocator | `can_allocate()`, `allocate(request)`, `free(request)` |
| **Member 5 (Jay / Benchmarks)** | Scheduler → Metrics consumer | `SchedulerMetrics` returned each step; `scheduler.metrics_log` for full history |

---

## Metrics Emitted

Each call to `step()` returns a `SchedulerMetrics` object:

| Field | Description |
|---|---|
| `step` | Sequential step counter |
| `timestamp` | Monotonic clock reading |
| `active_batch_size` | Sequences in the running set |
| `queue_depth` | Sequences still waiting |
| `iteration_latency_s` | Scheduler-side latency (not engine compute time) |
| `tokens_generated_this_step` | New tokens this step |
| `total_tokens_generated` | Cumulative since scheduler creation |
| `num_preemptions` | Evictions this step (CB only) |
| `num_prefills_this_step` | New prefills this step |

`SchedulerMetrics.summary(metrics_log)` produces aggregate statistics (avg batch size, p50/p99 latency, peak throughput) in a JSON-serialisable dict for Member 5's report.

---

## Testing

Unit tests: `tests/unit/test_scheduler.py` — 40 tests covering correctness, constraints, KV lifecycle, preemption, metrics, queue manager, and policies.

Integration tests: `tests/integration/test_scheduler_engine.py` — scheduler ↔ engine handshake, token accountability, metrics consistency, and CB vs static output parity.

Run all scheduler tests:
```bash
pytest tests/unit/test_scheduler.py tests/integration/test_scheduler_engine.py -v
```