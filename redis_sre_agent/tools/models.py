"""
定义“什么是工具”、“工具长什么样”以及“这个工具的安全性”的标准模板。
它把工具的各种信息打包归类，方便系统的其他部分统一管理。

BaseModel 是 Python 数据校验库 Pydantic 的核心基类。
在 Pydantic 的语境下，所有这类用于规范数据、定义字段和校验输入的类都被官方称为 Model（模型）。
因此，按照 Python 开发的常见惯例，存放这些 Pydantic 类的文件就会统一命名为 models.py。

这个文件里定义的不是具体的工具怎么去干活，而是工具的结构和属性规范。
例如：规定一个工具必须有名字、有描述、有能力分类。
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolCapability(str, Enum):
    """工具能力分类。"""

    METRICS = "metrics"
    LOGS = "logs"
    TICKETS = "tickets"
    REPOS = "repos"
    TRACES = "traces"
    DIAGNOSTICS = "diagnostics"
    KNOWLEDGE = "knowledge"
    UTILITIES = "utilities"
    ADMIN = "admin"


class ToolActionKind(str, Enum):
    """工具动作类型，用于区分只读工具和可能改动外部系统的工具。"""

    READ = "read"
    WRITE = "write"
    UNKNOWN = "unknown"


_READ_PREFIXES = (
    "get_",
    "list_",
    "query_",
    "search_",
    "find_",
    "read_",
    "inspect_",
    "describe_",
    "ping",
)
_WRITE_PREFIXES = (
    "create_",
    "update_",
    "delete_",
    "remove_",
    "add_",
    "upload_",
    "move_",
    "restart_",
    "stop_",
    "start_",
    "enable_",
    "disable_",
    "approve_",
    "resume_",
    "retry_",
    "cancel_",
    "set_",
    "link_",
    "unlink_",
    "transition_",
    "attach_",
    "detach_",
    "failover_",
)
_READ_EXACT = {
    "acl_log",
    "auth_status",
    "client_list",
    "config_get",
    "info",
    "logs",
    "ping",
    "rebalance_status",
    "slowlog",
    "bigkey_scan",
}
_READ_DESCRIPTION_MARKERS = (
    "get ",
    "list ",
    "query ",
    "search ",
    "read ",
    "inspect ",
    "retrieve ",
    "returns ",
    "ping ",
)
_WRITE_DESCRIPTION_MARKERS = (
    "create ",
    "update ",
    "delete ",
    "remove ",
    "add ",
    "upload ",
    "move ",
    "restart ",
    "enable ",
    "disable ",
    "approve ",
    "resume ",
    "retry ",
    "cancel ",
    "set ",
    "link ",
    "unlink ",
    "transition ",
    "attach ",
    "detach ",
    "overwrite ",
    "modify ",
)

# 专门看工具的描述文字是不是以某些特定的词（比如“获取”、“创建”）开头
def _description_starts_with_marker(description: str, markers: tuple[str, ...]) -> bool:
    normalized = description.strip().lower()
    return any(normalized.startswith(marker) for marker in markers)

# 把工具名字里那些冗长的前缀（比如供应商名字）去掉，提取出核心的动作名称
def _extract_operation_name(tool_name: str, provider_name: str) -> str:
    prefix_pattern = rf"^{re.escape(provider_name)}_[^_]+_"
    if re.match(prefix_pattern, tool_name):
        return re.sub(prefix_pattern, "", tool_name, count=1)
    if provider_name.startswith("mcp_") or tool_name.startswith("_"):
        return tool_name
    parts = tool_name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return tool_name

# 这是自动判断的大脑。它会综合看工具的名字、描述和能力分类，结合上面的“字典”，
# 推断出这个工具是安全的 READ 还是可能引发变动的 WRITE
def infer_tool_action_kind(
    *,
    name: str,
    description: str,
    capability: ToolCapability,
    provider_name: str,
) -> ToolActionKind:
    """根据工具名和描述推断读写类型。

    这是原项目的保守推断方式。阶段三没有审批流，但提前保留这个字段，后续阶段可以
    在不改工具协议的情况下接回人审和安全策略。
    """

    operation = _extract_operation_name(name, provider_name).lower().lstrip("_")
    description_lower = description.lower()

    if operation in _READ_EXACT:
        return ToolActionKind.READ
    if operation.startswith(_WRITE_PREFIXES):
        return ToolActionKind.WRITE
    if operation.startswith(_READ_PREFIXES):
        return ToolActionKind.READ
    if _description_starts_with_marker(description_lower, _READ_DESCRIPTION_MARKERS):
        return ToolActionKind.READ
    if _description_starts_with_marker(description_lower, _WRITE_DESCRIPTION_MARKERS):
        return ToolActionKind.WRITE
    if capability in {
        ToolCapability.METRICS,
        ToolCapability.LOGS,
        ToolCapability.TRACES,
        ToolCapability.KNOWLEDGE,
        ToolCapability.UTILITIES,
    }:
        return ToolActionKind.READ
    return ToolActionKind.UNKNOWN

# 一个工具的基础信息
class ToolMetadata(BaseModel):

    name: str
    description: str
    capability: ToolCapability
    provider_name: str
    requires_instance: bool = False
    action_kind: Optional[ToolActionKind] = None

    @model_validator(mode="after")
    def populate_action_kind(self) -> "ToolMetadata":
        if self.action_kind is None:
            self.action_kind = infer_tool_action_kind(
                name=self.name,
                description=self.description,
                capability=self.capability,
                provider_name=self.provider_name,
            )
        return self

# 纯粹是为了让 AI 大模型看懂而设计的。它里面只有工具的名字、用途描述和需要传什么参数。
class ToolDefinition(BaseModel):
    """给 Agent/LLM 看的工具 schema，不包含执行逻辑。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="带 provider 和实例 hash 的唯一工具名。")
    description: str = Field(..., description="给上层 Agent 看的工具说明。")
    parameters: Dict[str, Any] = Field(..., description="OpenAI function calling 风格参数。")
    capability: ToolCapability = Field(..., description="工具能力分类。")

    def __str__(self) -> str:
        return f"ToolDefinition(name={self.name})"

    def __repr__(self) -> str:
        param_names = list(self.parameters.get("properties", {}).keys())
        return f"ToolDefinition(name={self.name}, parameters={param_names})"

# 完整的实体工具
class Tool(BaseModel):
    """ToolManager 路由时使用的真实工具对象。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: ToolMetadata
    definition: "ToolDefinition"
    invoke: Callable[[Dict[str, Any]], Awaitable[Any]]

# 一台服务器的基础信息
class SystemHost(BaseModel):

    host: str
    port: Optional[int] = None
    role: Optional[str] = None
    labels: Dict[str, str] = Field(default_factory=dict)


