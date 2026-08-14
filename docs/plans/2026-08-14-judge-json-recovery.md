# Judge JSON Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Recover once from malformed Judge JSON without confusing an evaluation-system failure with a genuine zero score.

**Architecture:** Keep the existing strict JSON/YAML parser as the validation boundary. If the first Judge response cannot be parsed, make one narrowly scoped repair request containing the malformed response and require the same JSON object shape; parse and validate that response normally. If repair also fails, preserve the existing safe failure result and log both parse failures.

**Tech Stack:** Python, LangChain chat model interface, Pydantic, pytest.

---

### Task 1: Specify malformed-JSON recovery

**Files:**
- Modify: `tests/unit/evaluation/test_judge.py`

**Step 1:** Add a sequential fake Judge LLM capable of returning malformed JSON followed by repaired JSON.

**Step 2:** Add a test whose first response contains unescaped quotes inside `weaknesses`.

**Step 3:** Assert that evaluation succeeds from the second response, exactly two model calls occur, and the repair prompt includes the malformed response.

**Step 4:** Run the focused test and verify it fails before implementation.

### Task 2: Implement one-shot repair

**Files:**
- Modify: `redis_sre_agent/evaluation/judge.py`

**Step 1:** Add a narrowly scoped repair prompt that forbids changing evaluation meaning and requests only valid JSON.

**Step 2:** On the first parse failure, invoke the same bound Judge model once more with the malformed payload and parse the repaired response.

**Step 3:** If the repair fails, allow the existing outer exception handler to return the safe evaluation-error result.

**Step 4:** Run focused Judge tests and verify they pass.

### Task 3: Verify regression safety

**Files:**
- Test: `tests/unit/evaluation/test_judge.py`
- Test: `tests/unit/evaluation/test_bigkey_eval.py`
- Test: `tests/unit/evaluation/test_outcome_scenarios.py`

**Step 1:** Run the evaluation unit-test directory.

**Step 2:** Confirm valid Judge JSON still uses one call and malformed JSON uses no more than two calls.

**Step 3:** Review the diff to ensure no scenario fixtures or unrelated user changes were modified.
