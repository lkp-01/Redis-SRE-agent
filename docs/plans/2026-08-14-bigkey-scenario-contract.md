# BigKey Scenario Contract Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the BigKey YAML self-contained and typo-safe by adding provenance and Redis scope metadata, honoring the declared agent, and documenting every scenario field inline.

**Architecture:** Extend the immutable scenario dataclasses with typed provenance, scope, and Redis instance records. Parse and validate those records with explicit allow-lists, construct the runtime Redis target from the parsed scope, and route the declared `execution.agent` through a small agent factory. Keep knowledge and user-memory injection out of this evidence-only scenario.

**Tech Stack:** Python 3.12, dataclasses, PyYAML, LangChain/LangGraph, pytest.

---

### Task 1: Lock down the scenario contract

**Files:**
- Modify: `tests/unit/evaluation/test_bigkey_eval.py`
- Modify: `redis_sre_agent/evaluation/scenarios.py`

**Steps:**
1. Add tests proving `provenance`, `scope`, and `execution.agent` load into typed fields.
2. Add tests proving unknown top-level and execution fields are rejected instead of silently ignored.
3. Run the new tests and verify they fail against the old loader.
4. Add immutable dataclasses and explicit key validation for the new contract.
5. Run the focused loader tests and verify they pass.

### Task 2: Make scope and agent affect execution

**Files:**
- Modify: `redis_sre_agent/evaluation/runtime.py`
- Modify: `tests/unit/evaluation/test_bigkey_eval.py`

**Steps:**
1. Add tests proving the runtime Redis instance is built from scenario scope and unsupported agents fail clearly.
2. Replace the hard-coded evaluation instance with a conversion from `scenario.scope.redis_instance`.
3. Add a small agent factory keyed by `scenario.execution_agent`.
4. Run the focused runtime tests and verify they pass.

### Task 3: Preserve metadata in Judge context

**Files:**
- Modify: `redis_sre_agent/evaluation/judge.py`
- Modify: `tests/unit/evaluation/test_judge.py`

**Steps:**
1. Assert the Judge test-case payload includes provenance, execution agent, and scope.
2. Serialize those fields into the Judge prompt without exposing them to the candidate Agent as evidence.
3. Run Judge tests and verify the metadata is present.

### Task 4: Document the YAML and verify regressions

**Files:**
- Modify: `evals/scenarios/outcome/BigKey/scenario.yaml`

**Steps:**
1. Add the validated `provenance` and `scope` blocks.
2. Add concise Chinese comments above every existing and new YAML field.
3. Deliberately omit `knowledge` and `memory`, because this scenario does not inject either capability.
4. Run `uv run pytest tests/unit/evaluation -q` and expect all tests to pass.
5. Review the final diff for accidental changes outside the evaluation slice.
