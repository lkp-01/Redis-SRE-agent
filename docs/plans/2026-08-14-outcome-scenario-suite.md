# Outcome Scenario Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add eleven reproducible Redis SRE outcome scenarios covering connection readiness, persistence, memory, disk, swap, network, CPU, I/O, failover, and replication-buffer failures.

**Architecture:** Generalize the current BigKey-only replay contract so every call names a provider family (`redis_command`, `prometheus`, or `loki`) plus an operation. Keep all evidence local through provider-scoped JSON fixtures, retain the real `ChatAgent` and `ToolManager` execution path, and use semantic expectations for diagnosis quality. Preserve the BigKey public APIs as compatibility aliases while introducing generic outcome names.

**Tech Stack:** Python 3.12, dataclasses, PyYAML, LangChain/LangGraph, Redis diagnostic tools, Prometheus tools, Loki tools, pytest.

---

### Task 1: Specify the multi-provider replay contract

**Files:**
- Modify: `redis_sre_agent/evaluation/scenarios.py`
- Modify: `redis_sre_agent/evaluation/tool_runtime.py`
- Modify: `redis_sre_agent/evaluation/assertions.py`
- Modify: `redis_sre_agent/evaluation/judge.py`
- Modify: `tests/unit/evaluation/test_assertions.py`
- Create: `tests/unit/evaluation/test_outcome_scenarios.py`

**Steps:**
1. Add failing tests requiring every replay call and trace to carry `provider_family`.
2. Add a failing inventory test requiring exactly the original BigKey plus eleven named outcome scenarios.
3. Run the focused tests and confirm failures represent missing multi-provider support and missing scenario files.
4. Introduce generic `EvalScenario` and provider-scoped tool behavior mappings, retaining `BigKeyScenario` as an alias.
5. Make fixture dispatch validate both provider and operation.
6. Include provider identity in mechanical assertions and Judge expectation context.

### Task 2: Generalize runtime and CLI names

**Files:**
- Modify: `redis_sre_agent/evaluation/runtime.py`
- Modify: `redis_sre_agent/evaluation/runner.py`
- Modify: `redis_sre_agent/cli/eval/__init__.py`
- Modify: `tests/unit/evaluation/test_bigkey_eval.py`

**Steps:**
1. Add generic `run_eval_scenario` while retaining `run_bigkey_scenario` as a compatibility wrapper.
2. Select replay tools by provider prefix and operation suffix.
3. Remove BigKey-only wording from errors, docstrings, and CLI failure messages.
4. Run existing BigKey tests and keep them green.

### Task 3: Add the eleven outcome definitions

**Files:**
- Create: `evals/scenarios/outcome/<ScenarioName>/scenario.yaml`
- Create: `evals/scenarios/outcome/<ScenarioName>/fixtures/tools/*.json`

**Scenarios:**
1. `ConnectionReadyStorm`: many simultaneously ready client connections overload event-loop work.
2. `AOFEventLoopStall`: slow AOF fsync delays write processing.
3. `SlowClientOutputBuffer`: a slow receiver grows client output-buffer memory.
4. `ForkMemorySpike`: copy-on-write during fork raises RSS.
5. `DiskFullFailures`: ENOSPC breaks AOF/RDB persistence and rejects writes.
6. `SwapThrashing`: host memory shortage drives swap activity and latency.
7. `NetworkDegradation`: loss and retransmits explain request latency.
8. `CPUThreadContention`: CPU quota saturation and contention starve Redis work.
9. `SlowStorageIO`: high block-device latency stalls persistence.
10. `FailoverFlapping`: repeated role changes cause availability instability.
11. `ReplicaLagOutputBuffer`: a slow replica expands the master's replica output buffer.

**Steps:**
1. Give every YAML field a concise Chinese purpose comment.
2. Use three independent fixture calls per scenario wherever possible.
3. Write a source-constrained reference answer, required findings, and forbidden claims.
4. Keep `knowledge` and `memory` absent because these tests evaluate observed operational evidence.

### Task 4: Verify every scenario offline

**Files:**
- Modify: `tests/unit/evaluation/test_outcome_scenarios.py`

**Steps:**
1. Parameterize one end-to-end case over all eleven new scenarios.
2. Inject fake answer and Judge LLMs and forbid real Redis access.
3. Assert each expected provider/operation/argument trace and fixture result is observed.
4. Run `uv run pytest tests/unit/evaluation -q` and expect all tests to pass.
5. Run Python compilation and inspect the scenario inventory, IDs, fixture paths, and final diff.
