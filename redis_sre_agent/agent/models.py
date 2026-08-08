"""Agent 数据模型。

阶段三只需要 `AgentResponse` 承载曳光弹报告，但文件和核心模型名沿用原项目。
真实 LLM 话题抽取、推荐和决策追踪属于后续阶段，这里只保留轻量数据结构。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# AI在后台使用工具就要按照这个类进行记录(就像填表一样)
class ResultEnvelope(BaseModel):
    """一次工具调用的结构化记录。"""

    tool_key: str = Field(..., description="用于路由调用的完整工具名。")
    name: Optional[str] = Field(None, description="工具短操作名。")
    description: Optional[str] = Field(None, description="工具说明。")
    args: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field(..., description="'success' 或 'error'。")
    data: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None
    timestamp: Optional[str] = None

# AI发现的问题后，填写进来该问题的具体描述
class Topic(BaseModel):
    id: str
    title: str
    category: Literal[
        "Availability",
        "Replication",
        "Configuration",
        "Performance",
        "Persistence",
        "Security",
        "Networking",
        "Observability",
        "Other",
    ] = "Other"
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    scope: str = "cluster"
    narrative: str = ""
    evidence_keys: List[str] = Field(default_factory=list)

# 诊断的合集
class TopicsList(BaseModel):
    items: List[Topic] = Field(default_factory=list)

# 只记下了是哪个工具（tool_key）发现了上述的那个问题。
class TopicEvidence(BaseModel):
    tool_key: str

# 参考资料
class Citation(BaseModel):
    source: str
    snippet: str = ""

# 问题修复的具体操作步骤
class RecommendationStep(BaseModel):
    description: str
    commands: Optional[List[str]] = None
    api_examples: Optional[List[str]] = None
    citations: List[Citation] = Field(default_factory=list)

# 针对某个问题的修正处方
class Recommendation(BaseModel):
    topic_id: str
    title: Optional[str] = None
    steps: List[RecommendationStep] = Field(default_factory=list)
    risks: Optional[List[str]] = None
    verification: Optional[List[str]] = None

# AI发现自己回答错了后记录修改前的回答
class CorrectionResult(BaseModel):
    edited_response: str = Field(..., description="修正后的回答文本。")
    edits_applied: List[str] = Field(default_factory=list)


class TargetSelectionDecision(BaseModel):
    """DeepSeek 对是否实时诊断以及诊断目标作出的结构化决定。"""

    requires_live_diagnostics: bool = Field(
        ..., description="当前请求是否必须读取实时 Redis 状态才能回答。"
    )
    selected_target: Optional[str] = Field(
        None, description="从安全目标目录中选择的完整 display_name。"
    )
    reason_code: str = Field(
        default="", description="简短决策标签，不包含思维过程。"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

# AI思考完最终打包出来的东西，包括回复、缩到的结果、工具调用的证据
class AgentResponse(BaseModel):
    """Agent 返回值，沿用原项目响应形状。"""

    response: str = Field(..., description="Agent 的回答文本。")
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    tool_envelopes: List[Dict[str, Any]] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """citation 的唯一权威来源是顶层真实 tool_envelopes。"""
        from redis_sre_agent.agent.helpers import extract_citations

        object.__setattr__(self, "search_results", extract_citations(self.tool_envelopes))

# 用来记录AI是怎么思考的记录留存
class DecisionTrace(BaseModel):
    """后续阶段用于保存决策过程的轻量插槽。"""

    message_id: str = Field(..., description="消息 ID。")
    tool_envelopes: List[Dict[str, Any]] = Field(default_factory=list)
    otel_trace_id: Optional[str] = None
    created_at: Optional[str] = None
