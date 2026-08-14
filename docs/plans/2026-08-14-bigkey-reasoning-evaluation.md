# BigKey Reasoning Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the BigKey evaluation generate its diagnosis from replayed tool evidence and reject semantic Judge results that report factual errors or missing required elements.

**Architecture:** Keep deterministic tool-call replay and fixture dispatch for reproducibility. After all replayed ToolMessages are present, delegate final synthesis to an injected candidate LLM; retain the YAML answer only as a Judge reference. Combine deterministic assertions, Judge score, factual-error absence, and required-element completeness into the final pass gate.

**Tech Stack:** Python 3.12, LangChain messages, LangGraph ChatAgent, Pydantic, pytest.

---

### Task 1: Add evidence-driven answer synthesis

**Files:**
- Modify: `redis_sre_agent/evaluation/runtime.py`
- Modify: `redis_sre_agent/evaluation/scenarios.py`
- Modify: `evals/scenarios/outcome/BigKey/scenario.yaml`
- Test: `tests/unit/evaluation/test_bigkey_eval.py`

**Steps:**
1. Add a failing test proving the candidate answer LLM is invoked after all fixture ToolMessages and sees the tool evidence.
2. Replace replay final-answer injection with delegation to an injected `answer_llm`.
3. Default `answer_llm` to the configured main reasoning model for CLI execution.
4. Treat the YAML answer as a reference answer supplied only to Judge context.
5. Run the focused BigKey tests.

### Task 2: Strengthen semantic pass criteria

**Files:**
- Modify: `redis_sre_agent/evaluation/runtime.py`
- Test: `tests/unit/evaluation/test_bigkey_eval.py`

**Steps:**
1. Add failing tests for high-score Judge responses containing `factual_errors` and `missing_elements`.
2. Require score threshold, zero factual errors, and zero missing required elements.
3. Run the focused evaluation test suite.

### Task 3: Verify CLI and regression behavior

**Files:**
- Test: `tests/unit/evaluation/`

**Steps:**
1. Inject candidate and Judge fake models in offline CLI tests.
2. Run `uv run pytest tests/unit/evaluation -q`.
3. Run the real BigKey CLI and inspect candidate answer and Judge output.
4. Report any external-model/configuration limitation separately.

