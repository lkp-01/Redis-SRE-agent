# Outcome Evaluation Contract Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make semantic outcome evaluation fail only for declared required findings, declared forbidden-claim violations, material factual errors, or a score below threshold.

**Architecture:** Generate stable IDs for scenario requirements, require Judge outputs to reference those IDs, and normalize unknown Judge omissions into non-blocking advisory feedback. Keep the existing YAML string contract and CLI fields compatible while adding explicit forbidden-violation and advisory fields.

**Tech Stack:** Python 3.12, Pydantic, PyYAML, pytest, LangChain message fakes.

---

### Task 1: Lock the semantic contract with failing tests

**Files:**
- Modify: `tests/unit/evaluation/test_judge.py`
- Modify: `tests/unit/evaluation/test_bigkey_eval.py`

**Step 1: Write the failing tests**

Add tests proving that `required_finding_1` maps to the first required finding, an unknown reference-only omission becomes advisory, and `forbidden_claim_1` is blocking.

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/evaluation/test_judge.py tests/unit/evaluation/test_bigkey_eval.py -q`

Expected: failures for the new result fields, normalization, and pass-gate behavior.

### Task 2: Implement stable Judge requirement catalogs

**Files:**
- Modify: `redis_sre_agent/evaluation/judge.py`
- Modify: `redis_sre_agent/evaluation/runtime.py`
- Modify: `redis_sre_agent/cli/eval/__init__.py`

**Step 1: Add result fields and catalog helpers**

Add `violated_forbidden_claims` and `advisory_missing_elements` to `EvaluationResult`. Build catalogs shaped like:

```python
{"id": "required_finding_1", "description": finding}
```

**Step 2: Normalize Judge payloads**

Map declared IDs or exact descriptions to blocking descriptions. Move unmatched `missing_elements` to advisory feedback.

**Step 3: Tighten the Judge prompt**

Document the exact blocking semantics and omit the non-normative reference answer from the rendered prompt.

**Step 4: Update the pass gate and CLI output**

Require no forbidden violations in addition to the existing threshold/error checks; print advisory items separately.

**Step 5: Run focused tests**

Run: `uv run pytest tests/unit/evaluation/test_judge.py tests/unit/evaluation/test_bigkey_eval.py -q`

Expected: PASS.

### Task 3: Repair FailoverFlapping fixture chronology

**Files:**
- Modify: `evals/scenarios/outcome/FailoverFlapping/fixtures/tools/prometheus-role-changes.json`
- Test: `tests/unit/evaluation/test_outcome_scenarios.py`

**Step 1: Add chronology assertions**

Assert that Prometheus and Loki samples belong to the same 30-minute window and that `redis_up=0` samples align with `switch-master` timestamps.

**Step 2: Run the chronology test to verify it fails**

Run: `uv run pytest tests/unit/evaluation/test_outcome_scenarios.py -q`

Expected: FAIL on the stale 2024 Prometheus epochs.

**Step 3: Replace stale epochs with aligned 2026 values**

Keep the metric values unchanged while aligning their timestamps to the Sentinel events.

**Step 4: Run focused and full evaluation tests**

Run: `uv run pytest tests/unit/evaluation -q`

Expected: PASS.

### Task 4: Verify repository impact

**Files:**
- Inspect: all modified evaluation and fixture files

**Step 1: Run formatting/static checks available through pytest**

Run: `uv run pytest tests/unit/evaluation -q`

**Step 2: Inspect the diff**

Run: `git diff -- redis_sre_agent/evaluation redis_sre_agent/cli/eval tests/unit/evaluation evals/scenarios/outcome/FailoverFlapping docs/plans/2026-08-14-evaluation-contract-*`

Expected: only scoped evaluation changes; no unrelated user edits.
