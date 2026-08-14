"""使用 LLM 对 Redis SRE Agent 回答做语义评分。

该模块复刻 original 的 LLM-as-Judge 主路径。semantic outcome runner 会自动调用它；
测试应注入 fake LLM，默认 judge 使用配置的模型进行真实语义评分。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Sequence

import yaml
from pydantic import BaseModel, Field

from redis_sre_agent.agent.helpers import guarded_ainvoke
from redis_sre_agent.core.llm_helpers import create_mini_llm
from redis_sre_agent.evaluation.scenarios import EvalScenario

logger = logging.getLogger(__name__)
_FENCED_JSON_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(?=\s*[}\]])")


class EvaluationCriteria(BaseModel):
    """一项 judge 评分标准。"""

    name: str = Field(..., description="Name of the evaluation criteria")
    description: str = Field(..., description="What this criteria evaluates")
    weight: float = Field(default=1.0, description="Weight for this criteria (0-1)")
    required_elements: List[str] = Field(default_factory=list)
    accuracy_points: List[str] = Field(default_factory=list)


class RequiredElementAssessment(BaseModel):
    """Judge 对单个必需项的结构化判定。"""

    id: str
    description: str
    status: Literal["present", "partial", "missing"]
    evidence: str = ""
    explanation: str = ""


class ForbiddenClaimAssessment(BaseModel):
    """Judge 对单个禁止结论的结构化判定。"""

    id: str
    description: str
    status: Literal["absent", "violated"]
    evidence: str = ""
    explanation: str = ""


class EvaluationResult(BaseModel):
    """LLM judge 返回的标准化评分。"""

    test_case_id: str
    overall_score: float = Field(..., description="Overall score (0-100)")
    criteria_scores: Dict[str, float] = Field(default_factory=dict)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    factual_errors: List[str] = Field(default_factory=list)
    required_element_results: List[RequiredElementAssessment] = Field(default_factory=list)
    forbidden_claim_results: List[ForbiddenClaimAssessment] = Field(default_factory=list)
    missing_elements: List[str] = Field(default_factory=list)
    partial_elements: List[str] = Field(default_factory=list)
    violated_forbidden_claims: List[str] = Field(default_factory=list)
    advisory_missing_elements: List[str] = Field(default_factory=list)
    judge_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)
    detailed_feedback: str


class JudgeContractError(ValueError):
    """Judge JSON 可解析，但逐项评测合同不完整或自相矛盾。"""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class SREAgentJudge:
    """根据 rubric 评价 Redis 诊断回答；模型可注入，便于完全离线测试。"""

    JUDGE_SYSTEM_PROMPT = """You are an expert Redis SRE evaluator. Your role is to judge the quality, accuracy, and completeness of Redis SRE agent responses.

## Evaluation Framework

You will evaluate responses across the criteria supplied in the user prompt.

Common rubric dimensions include:
- technical accuracy
- completeness and relevance
- actionability
- evidence use
- instruction-following quality
- citation quality
- communication quality

## Response Format

Provide your evaluation as JSON:

```json
{
  "overall_score": <0-100>,
  "criteria_scores": {
    "<criteria_name>": <numeric score>
  },
  "strengths": ["specific strength 1", "specific strength 2"],
  "weaknesses": ["specific weakness 1", "specific weakness 2"],
  "factual_errors": ["material error 1 with explanation"],
  "required_element_results": [
    {
      "id": "required_finding_1",
      "status": "present",
      "evidence": "exact verbatim excerpt from the agent response",
      "explanation": "why the excerpt satisfies the requirement"
    }
  ],
  "forbidden_claim_results": [
    {
      "id": "forbidden_claim_1",
      "status": "absent",
      "evidence": "",
      "explanation": "the response does not endorse this claim"
    }
  ],
  "advisory_missing_elements": ["optional improvement 1"],
  "detailed_feedback": "Comprehensive feedback explaining scores and observations"
}
```

## Item-assessment contract

- `required_element_results` must contain every exact ID from the Required Element Catalog
  exactly once. Status must be `present`, `partial`, or `missing`.
- For `present` or `partial`, `evidence` must be a non-empty direct excerpt or concise grounded
  summary. Composite requirements may cite coverage spread across multiple response sections.
  For `missing`, evidence must be empty.
- Required coverage may appear anywhere in the response. Do not require concepts to be grouped
  in one paragraph or copied from the catalog verbatim.
- `forbidden_claim_results` must contain every exact ID from the Forbidden Claim Catalog exactly
  once. Status must be `absent` or `violated`. A `violated` result requires an exact verbatim
  excerpt; an `absent` result requires empty evidence.
- Silence is not a forbidden-claim violation and the response need not state the opposite warning.
- Put non-blocking omissions in `advisory_missing_elements` and minor wording issues in
  `weaknesses`.
- Put only material technical errors that could change the diagnosis, risk assessment, or
  recommended action in `factual_errors`.

Be rigorous in your evaluation. Technical accuracy is paramount.
"""

    JUDGE_REPAIR_SYSTEM_PROMPT = """You repair malformed JSON produced by an evaluation model.

Return exactly one valid JSON object and no Markdown or commentary. Preserve every score, list item,
and piece of feedback from the supplied payload. Only repair JSON syntax such as unescaped quotes,
invalid commas, or broken string delimiters. Do not add, remove, reinterpret, or re-evaluate content.
The object must use these keys: overall_score, criteria_scores, strengths, weaknesses,
factual_errors, required_element_results, forbidden_claim_results,
advisory_missing_elements, and detailed_feedback.
"""

    JUDGE_CONTRACT_REPAIR_SYSTEM_PROMPT = """You repair a semantically invalid Redis SRE evaluation.

Re-evaluate every required and forbidden catalog item against the supplied agent response.
Return exactly one JSON object and no Markdown. Every catalog ID must appear exactly once.
For present or partial items, provide a non-empty direct excerpt or concise grounded summary;
composite requirements may reference coverage across multiple sections. Violated forbidden
claims must use an exact verbatim excerpt. Missing or absent items must use an empty evidence
string. Do not copy previous invalid statuses merely to preserve them; correct them from the response.

The object must use these keys: overall_score, criteria_scores, strengths, weaknesses,
factual_errors, required_element_results, forbidden_claim_results,
advisory_missing_elements, and detailed_feedback.
"""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = (
            llm
            if llm is not None
            else create_mini_llm(model="deepseek-v4-pro").bind(
                response_format={"type": "json_object"}
            )
        )

    @staticmethod
    def _serialize_for_prompt(value: Any) -> str:
        """稳定序列化 prompt 上下文，减少无意义的顺序差异。"""

        if value is None:
            return "None"
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        elif hasattr(value, "__dataclass_fields__"):
            value = asdict(value)
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(value)

    @staticmethod
    def _sanitize_judge_response(content: str) -> str:
        """修正常见的 JSON 邻近格式，但不猜测缺失字段。"""

        # 中文引号在 JSON 字符串内是合法字符；不能全局替换为 ASCII 双引号，
        # 否则会把诸如 “冷却窗口” 的正常 prose 变成未转义的字符串分隔符。
        response_text = str(content or "").strip()
        fenced = _FENCED_JSON_RE.match(response_text)
        if fenced:
            response_text = fenced.group(1).strip()
        response_text = _TRAILING_COMMA_RE.sub("", response_text)

        cleaned: list[str] = []
        in_string = False
        escaping = False
        for index, char in enumerate(response_text):
            if escaping:
                cleaned.append(char)
                escaping = False
                continue
            if char == "\\":
                cleaned.append(char)
                escaping = True
                continue
            if char == '"':
                if not in_string:
                    cleaned.append(char)
                    in_string = True
                    continue
                next_index = index + 1
                while next_index < len(response_text) and response_text[next_index].isspace():
                    next_index += 1
                next_char = response_text[next_index] if next_index < len(response_text) else ""
                if not next_char or next_char in {":", ",", "}", "]"}:
                    cleaned.append(char)
                    in_string = False
                else:
                    cleaned.append('\\"')
                continue
            if in_string and char in {"\n", "\r", "\t"}:
                cleaned.append(" ")
                continue
            cleaned.append(char)
        return "".join(cleaned)

    @staticmethod
    def _parse_judge_payload(content: str) -> dict[str, Any]:
        """优先按 JSON 解析，并兼容模型偶尔返回的 YAML 风格对象。"""

        response_text = SREAgentJudge._sanitize_judge_response(content)
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = yaml.safe_load(response_text)
        if not isinstance(parsed, dict):
            raise ValueError("Judge response must deserialize to an object")
        return parsed

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        """只保留 Judge 列表字段中的非空字符串。"""

        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _normalize_evidence_text(value: Any) -> str:
        return " ".join(str(value or "").split())

    @staticmethod
    def _catalog_map(catalog: Any) -> dict[str, str]:
        if not isinstance(catalog, list):
            return {}
        return {
            str(item.get("id")): str(item.get("description"))
            for item in catalog
            if isinstance(item, dict) and item.get("id") and item.get("description")
        }

    @classmethod
    def _validate_required_element_results(
        cls,
        value: Any,
        catalog: Any,
        agent_response: str,
    ) -> tuple[list[RequiredElementAssessment], list[str]]:
        expected = cls._catalog_map(catalog)
        errors: list[str] = []
        if not isinstance(value, list):
            return [], ["required_element_results must be a list"]

        answer_text = cls._normalize_evidence_text(agent_response)
        seen: set[str] = set()
        assessments: dict[str, RequiredElementAssessment] = {}
        allowed_statuses = {"present", "partial", "missing"}
        for index, raw in enumerate(value, start=1):
            if not isinstance(raw, dict):
                errors.append(f"required_element_results[{index}] must be an object")
                continue
            item_id = str(raw.get("id") or "").strip()
            if item_id not in expected:
                errors.append(f"required_element_results[{index}] has unknown id: {item_id or '<empty>'}")
                continue
            if item_id in seen:
                errors.append(f"required_element_results contains duplicate id: {item_id}")
                continue
            seen.add(item_id)
            status = str(raw.get("status") or "").strip().lower()
            if status not in allowed_statuses:
                errors.append(f"{item_id} has invalid required status: {status or '<empty>'}")
                continue
            evidence = str(raw.get("evidence") or "").strip()
            explanation = str(raw.get("explanation") or "").strip()
            normalized_evidence = cls._normalize_evidence_text(evidence)
            if status in {"present", "partial"}:
                if not normalized_evidence:
                    errors.append(f"{item_id} status {status} requires grounded evidence")
            elif normalized_evidence:
                errors.append(f"{item_id} status missing requires empty evidence")
            assessments[item_id] = RequiredElementAssessment(
                id=item_id,
                description=expected[item_id],
                status=status,
                evidence=evidence,
                explanation=explanation,
            )

        missing_ids = [item_id for item_id in expected if item_id not in seen]
        if missing_ids:
            errors.append(
                "required_element_results omitted catalog ids: " + ", ".join(missing_ids)
            )
        ordered = [assessments[item_id] for item_id in expected if item_id in assessments]
        return ordered, errors

    @classmethod
    def _validate_forbidden_claim_results(
        cls,
        value: Any,
        catalog: Any,
        agent_response: str,
    ) -> tuple[list[ForbiddenClaimAssessment], list[str]]:
        expected = cls._catalog_map(catalog)
        errors: list[str] = []
        if not isinstance(value, list):
            return [], ["forbidden_claim_results must be a list"]

        answer_text = cls._normalize_evidence_text(agent_response)
        seen: set[str] = set()
        assessments: dict[str, ForbiddenClaimAssessment] = {}
        allowed_statuses = {"absent", "violated"}
        for index, raw in enumerate(value, start=1):
            if not isinstance(raw, dict):
                errors.append(f"forbidden_claim_results[{index}] must be an object")
                continue
            item_id = str(raw.get("id") or "").strip()
            if item_id not in expected:
                errors.append(f"forbidden_claim_results[{index}] has unknown id: {item_id or '<empty>'}")
                continue
            if item_id in seen:
                errors.append(f"forbidden_claim_results contains duplicate id: {item_id}")
                continue
            seen.add(item_id)
            status = str(raw.get("status") or "").strip().lower()
            if status not in allowed_statuses:
                errors.append(f"{item_id} has invalid forbidden status: {status or '<empty>'}")
                continue
            evidence = str(raw.get("evidence") or "").strip()
            explanation = str(raw.get("explanation") or "").strip()
            normalized_evidence = cls._normalize_evidence_text(evidence)
            if status == "violated":
                if not normalized_evidence:
                    errors.append(f"{item_id} status violated requires verbatim evidence")
                elif normalized_evidence not in answer_text:
                    errors.append(f"{item_id} evidence is not a verbatim excerpt from the agent response")
            elif normalized_evidence:
                errors.append(f"{item_id} status absent requires empty evidence")
            assessments[item_id] = ForbiddenClaimAssessment(
                id=item_id,
                description=expected[item_id],
                status=status,
                evidence=evidence,
                explanation=explanation,
            )

        missing_ids = [item_id for item_id in expected if item_id not in seen]
        if missing_ids:
            errors.append(
                "forbidden_claim_results omitted catalog ids: " + ", ".join(missing_ids)
            )
        ordered = [assessments[item_id] for item_id in expected if item_id in assessments]
        return ordered, errors

    @classmethod
    def _result_from_payload(
        cls,
        *,
        payload: dict[str, Any],
        test_case: Dict[str, Any],
        agent_response: str,
    ) -> EvaluationResult:
        required_results, required_errors = cls._validate_required_element_results(
            payload.get("required_element_results"),
            test_case.get("required_element_catalog"),
            agent_response,
        )
        forbidden_results, forbidden_errors = cls._validate_forbidden_claim_results(
            payload.get("forbidden_claim_results"),
            test_case.get("forbidden_claim_catalog"),
            agent_response,
        )
        errors = [*required_errors, *forbidden_errors]
        if errors:
            raise JudgeContractError(errors)
        return EvaluationResult(
            test_case_id=test_case.get("id", "unknown"),
            overall_score=payload.get("overall_score", 0),
            criteria_scores=payload.get("criteria_scores", {}),
            strengths=cls._string_list(payload.get("strengths", [])),
            weaknesses=cls._string_list(payload.get("weaknesses", [])),
            factual_errors=cls._string_list(payload.get("factual_errors", [])),
            required_element_results=required_results,
            forbidden_claim_results=forbidden_results,
            missing_elements=[
                item.description for item in required_results if item.status == "missing"
            ],
            partial_elements=[
                item.description for item in required_results if item.status == "partial"
            ],
            violated_forbidden_claims=[
                item.description for item in forbidden_results if item.status == "violated"
            ],
            advisory_missing_elements=cls._string_list(
                payload.get("advisory_missing_elements", [])
            ),
            judge_valid=True,
            validation_errors=[],
            detailed_feedback=str(payload.get("detailed_feedback") or "No feedback provided"),
        )

    @classmethod
    def _invalid_result(
        cls,
        *,
        test_case: Dict[str, Any],
        errors: Sequence[str],
        payload: dict[str, Any] | None = None,
        detailed_feedback: str | None = None,
    ) -> EvaluationResult:
        payload = payload or {}
        try:
            overall_score = float(payload.get("overall_score", 0))
        except (TypeError, ValueError):
            overall_score = 0.0
        criteria_scores = payload.get("criteria_scores", {})
        if not isinstance(criteria_scores, dict):
            criteria_scores = {}
        return EvaluationResult(
            test_case_id=test_case.get("id", "unknown"),
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            strengths=cls._string_list(payload.get("strengths", [])),
            weaknesses=cls._string_list(payload.get("weaknesses", [])),
            factual_errors=cls._string_list(payload.get("factual_errors", [])),
            advisory_missing_elements=cls._string_list(
                payload.get("advisory_missing_elements", [])
            ),
            judge_valid=False,
            validation_errors=list(errors),
            detailed_feedback=detailed_feedback
            or str(payload.get("detailed_feedback") or "Judge contract validation failed"),
        )

    async def _repair_contract_payload(
        self,
        *,
        payload: dict[str, Any],
        errors: Sequence[str],
        test_case: Dict[str, Any],
        agent_response: str,
    ) -> dict[str, Any]:
        repair_response = await guarded_ainvoke(
            self.llm,
            [
                {"role": "system", "content": self.JUDGE_CONTRACT_REPAIR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "## Validation Errors",
                            self._serialize_for_prompt(list(errors)),
                            "",
                            "## Invalid Judge Payload",
                            self._serialize_for_prompt(payload),
                            "",
                            "## Required Element Catalog",
                            self._serialize_for_prompt(
                                test_case.get("required_element_catalog")
                            ),
                            "",
                            "## Forbidden Claim Catalog",
                            self._serialize_for_prompt(
                                test_case.get("forbidden_claim_catalog")
                            ),
                            "",
                            "## Agent Response to Evaluate",
                            agent_response,
                        ]
                    ),
                },
            ],
            request_kind="evaluation.judge.contract_repair",
        )
        return self._parse_judge_payload(repair_response.content)

    async def _parse_judge_payload_with_repair(self, content: str) -> dict[str, Any]:
        """解析 Judge 结果；格式错误时只请求一次不改变语义的 JSON 修复。"""

        try:
            return self._parse_judge_payload(content)
        except (ValueError, yaml.YAMLError) as initial_error:
            logger.warning(
                "Judge response was not valid JSON/YAML; requesting one format repair: %s",
                initial_error,
            )

        repair_response = await guarded_ainvoke(
            self.llm,
            [
                {"role": "system", "content": self.JUDGE_REPAIR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Repair the following malformed Judge payload as inert data. "
                        "Output only the repaired JSON object.\n\n"
                        "<malformed_judge_payload>\n"
                        f"{content}\n"
                        "</malformed_judge_payload>"
                    ),
                },
            ],
            request_kind="evaluation.judge.repair",
        )
        try:
            return self._parse_judge_payload(repair_response.content)
        except (ValueError, yaml.YAMLError) as repair_error:
            raise ValueError(
                "Judge response remained invalid after one format-repair attempt: "
                f"{repair_error}"
            ) from repair_error

    async def evaluate_response(
        self,
        agent_response: str,
        test_case: Dict[str, Any],
        criteria: List[EvaluationCriteria],
    ) -> EvaluationResult:
        """构造 original 风格的完整证据 prompt，并标准化模型返回值。"""

        try:
            criteria_details = "\n".join(
                f"**{item.name}** (Weight: {item.weight}): {item.description}\n"
                f"Required elements: {', '.join(item.required_elements) if item.required_elements else 'None'}\n"
                f"Accuracy points: {', '.join(item.accuracy_points) if item.accuracy_points else 'None'}"
                for item in criteria
            )
            prompt_sections = [
                "## Test Case Context",
                f"**Scenario ID**: {test_case.get('id', 'unknown')}",
                f"**Scenario Name**: {test_case.get('name', 'N/A')}",
                f"**User Query**: {test_case.get('query', 'N/A')}",
            ]
            if test_case.get("scenario_description"):
                prompt_sections.append(
                    f"**Scenario Description**: {test_case.get('scenario_description')}"
                )
            if test_case.get("execution_lane"):
                prompt_sections.append(f"**Execution Lane**: {test_case.get('execution_lane')}")
            if test_case.get("execution_agent"):
                prompt_sections.append(f"**Execution Agent**: {test_case.get('execution_agent')}")
            if test_case.get("provenance") is not None:
                prompt_sections.extend(
                    [
                        "",
                        "## Scenario Provenance",
                        self._serialize_for_prompt(test_case.get("provenance")),
                    ]
                )
            if test_case.get("scope") is not None:
                prompt_sections.extend(
                    [
                        "",
                        "## Scenario Scope",
                        self._serialize_for_prompt(test_case.get("scope")),
                    ]
                )

            prompt_sections.extend(
                [
                    "",
                    "## Startup Context",
                    self._serialize_for_prompt(test_case.get("startup_context")),
                    "",
                    "## Tool Trace",
                    self._serialize_for_prompt(test_case.get("tool_trace")),
                    "",
                    "## Retrieved Sources",
                    self._serialize_for_prompt(test_case.get("retrieved_sources")),
                    "",
                    "## Expectation Set",
                    self._serialize_for_prompt(test_case.get("expectations")),
                    "",
                    "## Structured Assertions",
                    self._serialize_for_prompt(test_case.get("structured_assertions")),
                ]
            )
            if test_case.get("actual_routing_decision") is not None:
                prompt_sections.extend(
                    [
                        "",
                        "## Actual Routing Decision",
                        self._serialize_for_prompt(test_case.get("actual_routing_decision")),
                    ]
                )
            if test_case.get("diagnostic_data") is not None:
                prompt_sections.extend(
                    [
                        "",
                        "## Diagnostic Data",
                        self._serialize_for_prompt(test_case.get("diagnostic_data")),
                    ]
                )
            if test_case.get("required_element_catalog") is not None:
                prompt_sections.extend(
                    [
                        "",
                        "## Required Element Catalog",
                        self._serialize_for_prompt(test_case.get("required_element_catalog")),
                    ]
                )
            if test_case.get("forbidden_claim_catalog") is not None:
                prompt_sections.extend(
                    [
                        "",
                        "## Forbidden Claim Catalog",
                        self._serialize_for_prompt(test_case.get("forbidden_claim_catalog")),
                    ]
                )
            prompt_sections.extend(
                [
                    "",
                    "## Evaluation Criteria",
                    criteria_details,
                    "",
                    "## Agent Response to Evaluate",
                    agent_response,
                    "",
                    "Please evaluate this Redis SRE agent response thoroughly and provide your assessment.",
                ]
            )

            response = await guarded_ainvoke(
                self.llm,
                [
                    {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(prompt_sections)},
                ],
                request_kind="evaluation.judge",
            )
            result_data = await self._parse_judge_payload_with_repair(response.content)
            try:
                initial_result = self._result_from_payload(
                    payload=result_data,
                    test_case=test_case,
                    agent_response=agent_response,
                )
            except JudgeContractError as initial_contract_error:
                contract_errors = initial_contract_error.errors
                logger.warning(
                    "Judge item contract was invalid; requesting one semantic repair: %s",
                    initial_contract_error,
                )
            else:
                blocking_findings = [
                    *(f"missing required element: {item}" for item in initial_result.missing_elements),
                    *(
                        f"violated forbidden claim: {item}"
                        for item in initial_result.violated_forbidden_claims
                    ),
                    *(f"factual error: {item}" for item in initial_result.factual_errors),
                ]
                if not blocking_findings:
                    return initial_result
                contract_errors = [
                    "Blocking findings require one independent confirmation before failing the agent.",
                    *blocking_findings,
                ]
                logger.warning(
                    "Judge produced blocking findings; requesting one confirmation: %s",
                    "; ".join(blocking_findings),
                )
            try:
                repaired_data = await self._repair_contract_payload(
                    payload=result_data,
                    errors=contract_errors,
                    test_case=test_case,
                    agent_response=agent_response,
                )
                return self._result_from_payload(
                    payload=repaired_data,
                    test_case=test_case,
                    agent_response=agent_response,
                )
            except JudgeContractError as repair_contract_error:
                logger.error(
                    "Judge item contract remained invalid after one semantic repair: %s",
                    repair_contract_error,
                )
                return self._invalid_result(
                    test_case=test_case,
                    errors=repair_contract_error.errors,
                    payload=repaired_data,
                )
            except Exception as repair_error:
                logger.error("Judge contract repair failed: %s", repair_error)
                return self._invalid_result(
                    test_case=test_case,
                    errors=[f"Judge contract repair failed: {repair_error}"],
                    payload=result_data,
                )
        except Exception as exc:
            logger.error("Error during evaluation: %s", exc)
            return self._invalid_result(
                test_case=test_case,
                errors=[f"Evaluation error: {exc}"],
                detailed_feedback=f"Evaluation failed due to error: {exc}",
            )


def build_default_eval_criteria() -> list[EvaluationCriteria]:
    """返回与 original 一致的默认 Redis SRE rubric。"""

    return [
        EvaluationCriteria(
            name="technical_accuracy",
            description="Correct Redis and SRE diagnosis with no material factual errors.",
            weight=0.25,
        ),
        EvaluationCriteria(
            name="completeness_relevance",
            description="Coverage of the user's request and the scenario's expected findings.",
            weight=0.2,
        ),
        EvaluationCriteria(
            name="actionability",
            description="Clear, prioritized next steps and operational guidance.",
            weight=0.2,
        ),
        EvaluationCriteria(
            name="evidence_use",
            description="Grounding in the provided startup context, tool results, and retrieved sources.",
            weight=0.15,
        ),
        EvaluationCriteria(
            name="instruction_following",
            description="Alignment with the scenario expectations and source-constrained instructions.",
            weight=0.1,
        ),
        EvaluationCriteria(
            name="citation_quality",
            description="Use of specific source or tool evidence when making claims.",
            weight=0.1,
        ),
    ]


def _serialize_expectation_set(scenario: EvalScenario) -> dict[str, Any]:
    return {
        "judge": scenario.judge,
        "required_tool_calls": [
            {
                "provider_family": call.provider_family,
                "operation": call.operation,
                "args": call.args,
            }
            for call in scenario.replay_calls
        ],
        "required_response_patterns": list(scenario.required_response_patterns),
        "required_findings": list(scenario.required_findings),
        "forbidden_claims": list(scenario.forbidden_claims),
    }


def _serialize_scenario_scope(scenario: EvalScenario) -> dict[str, Any]:
    """给 Judge 保留目标身份与环境，但不暴露 Redis 连接地址。"""

    scope = asdict(scenario.scope)
    scope["redis_instance"].pop("connection_url", None)
    return scope


def build_eval_judge_test_case(
    scenario: EvalScenario,
    *,
    startup_context: Any = None,
    tool_trace: Sequence[Any] | None = None,
    retrieved_sources: Sequence[Any] | None = None,
    structured_assertions: Any = None,
    actual_routing_decision: str | None = None,
    diagnostic_data: Any = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把当前 outcome dataclass 转成 original judge 使用的丰富上下文。"""

    required_element_catalog = [
        {"id": f"required_finding_{index}", "description": finding}
        for index, finding in enumerate(scenario.required_findings, start=1)
    ]
    forbidden_claim_catalog = [
        {"id": f"forbidden_claim_{index}", "description": claim}
        for index, claim in enumerate(scenario.forbidden_claims, start=1)
    ]
    payload = {
        "id": scenario.id,
        "name": scenario.name,
        "query": scenario.query,
        "scenario_description": scenario.description,
        "provenance": asdict(scenario.provenance),
        "execution_lane": scenario.execution_lane,
        "execution_agent": scenario.execution_agent,
        "scope": _serialize_scenario_scope(scenario),
        "startup_context": startup_context,
        "tool_trace": list(tool_trace or []),
        "retrieved_sources": list(retrieved_sources or []),
        "expectations": _serialize_expectation_set(scenario),
        "structured_assertions": structured_assertions.model_dump(mode="json")
        if hasattr(structured_assertions, "model_dump")
        else structured_assertions,
        "actual_routing_decision": actual_routing_decision,
        "diagnostic_data": diagnostic_data,
        "reference_answer": scenario.reference_answer,
        "required_element_catalog": required_element_catalog,
        "forbidden_claim_catalog": forbidden_claim_catalog,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


async def evaluate_eval_scenario_response(
    *,
    scenario: EvalScenario,
    agent_response: str,
    judge: SREAgentJudge | None = None,
    criteria: Iterable[EvaluationCriteria] | None = None,
    startup_context: Any = None,
    tool_trace: Sequence[Any] | None = None,
    retrieved_sources: Sequence[Any] | None = None,
    structured_assertions: Any = None,
    actual_routing_decision: str | None = None,
    diagnostic_data: Any = None,
    extra_fields: dict[str, Any] | None = None,
) -> EvaluationResult:
    """显式运行 LLM judge；调用方负责决定阈值和是否允许网络。"""

    active_judge = judge or SREAgentJudge()
    test_case = build_eval_judge_test_case(
        scenario,
        startup_context=startup_context,
        tool_trace=tool_trace,
        retrieved_sources=retrieved_sources,
        structured_assertions=structured_assertions,
        actual_routing_decision=actual_routing_decision,
        diagnostic_data=diagnostic_data,
        extra_fields=extra_fields,
    )
    return await active_judge.evaluate_response(
        agent_response,
        test_case,
        list(criteria or build_default_eval_criteria()),
    )


class EvaluationSuite:
    """保留 original 的通用批量 judge 入口。"""

    def __init__(self, judge: SREAgentJudge | None = None) -> None:
        self.judge = judge or SREAgentJudge()
        self.test_cases: list[Dict[str, Any]] = []

    def add_test_case(self, test_case: Dict[str, Any]) -> None:
        self.test_cases.append(test_case)

    async def run_evaluation(self, agent_function: Any) -> List[EvaluationResult]:
        results: list[EvaluationResult] = []
        for index, test_case in enumerate(self.test_cases):
            try:
                response = await agent_function(
                    test_case["query"],
                    f"eval_session_{index}",
                    "evaluation_user",
                )
                criteria = [EvaluationCriteria(**item) for item in test_case.get("criteria", [])]
                results.append(await self.judge.evaluate_response(response, test_case, criteria))
            except Exception as exc:
                logger.error("Failed to run test case %s: %s", index + 1, exc)
                results.append(
                    EvaluationResult(
                        test_case_id=test_case.get("id", f"test_{index}"),
                        overall_score=0,
                        criteria_scores={},
                        weaknesses=[f"Test execution failed: {exc}"],
                        detailed_feedback=f"Test case failed to execute: {exc}",
                    )
                )
        return results

    def generate_report(self, results: List[EvaluationResult]) -> str:
        """按 original 的人类可读格式汇总分数、常见问题和单项结果。"""

        if not results:
            return "No evaluation results available."
        total_score = sum(item.overall_score for item in results) / len(results)
        report = f"""# Redis SRE Agent Evaluation Report

**Date**: {datetime.now().isoformat()}
**Test Cases**: {len(results)}
**Average Score**: {total_score:.1f}/100

## Summary Statistics

"""
        score_ranges = {"90-100": 0, "80-89": 0, "70-79": 0, "60-69": 0, "<60": 0}
        for result in results:
            score = result.overall_score
            if score >= 90:
                score_ranges["90-100"] += 1
            elif score >= 80:
                score_ranges["80-89"] += 1
            elif score >= 70:
                score_ranges["70-79"] += 1
            elif score >= 60:
                score_ranges["60-69"] += 1
            else:
                score_ranges["<60"] += 1

        report += "**Score Distribution:**\n"
        for range_name, count in score_ranges.items():
            report += f"- {range_name}: {count} tests ({count / len(results) * 100:.1f}%)\n"

        weaknesses = [item for result in results for item in result.weaknesses]
        errors = [item for result in results for item in result.factual_errors]
        if weaknesses:
            report += "\n**Most Common Weaknesses:**\n"
            counts = {item: weaknesses.count(item) for item in weaknesses}
            for item, count in sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:5]:
                report += f"- {item} ({count} cases)\n"
        if errors:
            report += "\n**Factual Errors Found:**\n"
            counts = {item: errors.count(item) for item in errors}
            for item, count in sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:5]:
                report += f"- {item} ({count} cases)\n"

        report += "\n## Individual Test Results\n\n"
        for index, result in enumerate(results, 1):
            report += f"### Test Case {index}\n"
            report += f"**Score**: {result.overall_score:.1f}/100\n"
            report += "**Criteria Scores**: " + ", ".join(
                f"{key}: {value}" for key, value in result.criteria_scores.items()
            ) + "\n"
            if result.strengths:
                report += f"**Strengths**: {'; '.join(result.strengths)}\n"
            if result.weaknesses:
                report += f"**Weaknesses**: {'; '.join(result.weaknesses)}\n"
            if result.factual_errors:
                report += f"**Factual Errors**: {'; '.join(result.factual_errors)}\n"
            report += "\n"
        return report


__all__ = [
    "EvaluationCriteria",
    "EvaluationResult",
    "EvaluationSuite",
    "SREAgentJudge",
    "build_default_eval_criteria",
    "build_eval_judge_test_case",
    "evaluate_eval_scenario_response",
]
