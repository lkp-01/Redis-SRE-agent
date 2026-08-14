# Judge Item Assessments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Derive semantic evaluation failures from validated per-item Judge assessments instead of trusting a contradictory missing-ID list.

**Architecture:** Add required and forbidden assessment arrays to the Judge protocol, validate catalog coverage and evidence presence, and retry once for either an invalid judgment or any blocking judgment that needs confirmation. Require verbatim evidence for forbidden violations, while allowing concise grounded summaries for composite required findings. Surface a second invalid result as evaluation infrastructure failure through the runtime and CLI.

**Tech Stack:** Python 3.12, Pydantic, LangChain message fakes, pytest.

---

### Task 1: Add failing protocol tests

**Files:**
- Modify: `tests/unit/evaluation/test_judge.py`
- Modify: `tests/unit/evaluation/test_bigkey_eval.py`

**Step 1: Add per-item fixtures**

Generate one assessment for every catalog ID with a verbatim excerpt from the candidate answer.

**Step 2: Test derived outcomes**

Cover `present`, `partial`, `missing`, and `violated` status mapping.

**Step 3: Test invalid-contract recovery**

Return an invalid first payload followed by a valid second payload and assert two Judge calls. Return two invalid payloads and assert `judge_valid=false`.

**Step 4: Run tests and confirm failure**

Run: `uv run pytest tests/unit/evaluation/test_judge.py tests/unit/evaluation/test_bigkey_eval.py -q`

Expected: FAIL because the current evaluator has no item-assessment protocol.

### Task 2: Implement item validation and retry

**Files:**
- Modify: `redis_sre_agent/evaluation/judge.py`

**Step 1: Extend result models**

Add `required_element_results`, `forbidden_claim_results`, `partial_elements`, `judge_valid`, and `validation_errors`.

**Step 2: Replace missing-list normalization**

Validate every catalog ID exactly once, validate statuses, and require normalized verbatim evidence for non-absent claims.

**Step 3: Add one contract retry**

On validation failure, call the Judge once with the invalid payload, validation errors, catalogs, and candidate answer. Validate the repaired result without a third semantic retry.

For a contract-valid result containing missing required items, forbidden violations, or factual errors, use the same single retry as an independent blocking confirmation before failing the Agent.

**Step 4: Run focused Judge tests**

Run: `uv run pytest tests/unit/evaluation/test_judge.py -q`

Expected: PASS.

### Task 3: Surface invalid evaluation separately

**Files:**
- Modify: `redis_sre_agent/evaluation/runtime.py`
- Modify: `redis_sre_agent/cli/eval/__init__.py`
- Modify: `tests/unit/evaluation/test_bigkey_eval.py`

**Step 1: Make pass state tri-valued**

Use `None` for `judge_passed` and top-level `passed` when `judge_valid=false`.

**Step 2: Update CLI output**

Print `judge_valid`, validation errors, partial elements, and `passed=invalid`; raise `Outcome evaluation invalid` for this state.

**Step 3: Run focused runtime/CLI tests**

Run: `uv run pytest tests/unit/evaluation/test_bigkey_eval.py -q`

Expected: PASS.

### Task 4: Regression verification

**Files:**
- Inspect: scoped evaluation files and tests

**Step 1: Run the evaluation suite**

Run: `uv run pytest tests/unit/evaluation -q`

Expected: PASS.

**Step 2: Compile changed Python modules**

Run: `uv run python -m compileall -q redis_sre_agent/evaluation redis_sre_agent/cli/eval`

Expected: exit code 0.

**Step 3: Inspect workspace status**

Run: `git status --short`

Expected: only scoped new evaluation changes plus pre-existing user modifications.
