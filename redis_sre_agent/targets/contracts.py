"""
在复杂的 AI 系统中，我们不能直接把数据库密码、内部服务器的真实 ID（敏感信息）直接扔给 AI。
这个文件通过定义一套“公开（Public）”和“私有（Private）”的分离机制，
确保 AI 只能看到安全的资源摘要，而服务端在底层去匹配和绑定真实的资源。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field, field_validator

MULTI_TARGET_SELECTION_LIMIT = 5
DISCOVERY_STATUS_TOO_MANY_MATCHES = "too_many_matches"

# Agent搜索到的资源填入该对象中，Public代表可以展示给用户
class PublicTargetMatch(BaseModel):
    """可以安全展示给 Agent 的 target 匹配结果。"""

    target_kind: str
    display_name: str
    environment: Optional[str] = None
    target_type: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    confidence: float
    match_reasons: List[str] = Field(default_factory=list)
    public_metadata: Dict[str, Any] = Field(default_factory=dict)
    resource_id: Optional[str] = Field(default=None, exclude=True)
    score: float = Field(default=0.0, exclude=True)

    @staticmethod
    def _strip_duplicate_public_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(payload.get("public_metadata") or {})
        for key in ("environment", "target_type"):
            if metadata.get(key) == payload.get(key):
                metadata.pop(key, None)
        payload["public_metadata"] = metadata
        return payload

    def public_dump(self) -> Dict[str, Any]:
        return self._strip_duplicate_public_metadata(self.model_dump(mode="json"))

# 绑定一个或多个目标，生成摘要，存入上下文。这样在不调用查询等动作的时候，Agent可以直接根据上下文中的这个摘要进行回答。
class PublicTargetBinding(BaseModel):
    """绑定到一次运行上下文中的安全 target 摘要。"""

    target_handle: str
    target_kind: str
    display_name: str
    capabilities: List[str] = Field(default_factory=list)
    public_metadata: Dict[str, Any] = Field(default_factory=dict)
    thread_id: Optional[str] = None
    task_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = Field(
        default_factory=lambda: (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    )
    resource_id: Optional[str] = None

    def public_dump(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

# Agent 上下文只携带 attached_target_handles: ["tgt_xxx"]
# 真正要加载工具时再根据 handle 查出 TargetHandleRecord
class TargetHandleRecord(BaseModel):
    """服务端私有 handle 记录，保存真实资源 id 和绑定策略。"""

    target_handle: str
    discovery_backend: str
    binding_strategy: str
    binding_subject: str
    private_binding_ref: Dict[str, Any] = Field(default_factory=dict)
    public_summary: PublicTargetBinding
    thread_id: Optional[str] = None
    task_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = Field(
        default_factory=lambda: (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    )

# 比如用户提问后，系统会构造一个 DiscoveryRequest，交给 discovery backend 去匹配目标目录
class DiscoveryRequest(BaseModel):
    """自然语言 target 解析请求。"""

    query: str
    allow_multiple: bool = False
    max_results: int = 5
    preferred_capabilities: List[str] = Field(default_factory=list)
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    task_id: Optional[str] = None

# 服务端的“备选档案”。
# 它同时包裹了“准备给 AI 看的公开信息（public_match）”和“服务端自己留着的私有绑定信息”。
#
# DiscoveryCandidate 是为了在搜索匹配时，方便把“公开信息”和“私有信息”暂时打包在一起传递而设计的过渡对象；
# 而TargetHandleRecord 是真正落地生效、分配了唯一资源 ID，用来支撑后续所有实际调用的核心凭证
class DiscoveryCandidate(BaseModel):

    public_match: PublicTargetMatch
    binding_strategy: str
    binding_subject: str
    private_binding_ref: Dict[str, Any] = Field(default_factory=dict)
    discovery_backend: str
    score: float
    confidence: float

    @staticmethod
    def _resolve_default_integration_names() -> tuple[str, str]:
        from .registry import get_target_integration_registry

        registry = get_target_integration_registry()
        return registry.default_discovery_backend, registry.default_binding_strategy

    @classmethod
    def from_public_match(
        cls,
        public_match: PublicTargetMatch,
        *,
        binding_strategy: Optional[str] = None,
        binding_subject: Optional[str] = None,
        private_binding_ref: Optional[Dict[str, Any]] = None,
        discovery_backend: Optional[str] = None,
    ) -> "DiscoveryCandidate":
        if binding_strategy is None or discovery_backend is None:
            default_discovery_backend, default_binding_strategy = (
                cls._resolve_default_integration_names()
            )
        else:
            default_discovery_backend = discovery_backend
            default_binding_strategy = binding_strategy
        return cls(
            public_match=public_match,
            binding_strategy=binding_strategy or default_binding_strategy,
            binding_subject=binding_subject or public_match.resource_id or "",
            private_binding_ref=private_binding_ref or {"target_kind": public_match.target_kind},
            discovery_backend=discovery_backend or default_discovery_backend,
            score=public_match.score,
            confidence=public_match.confidence,
        )

    @property
    def target_kind(self) -> str:
        return self.public_match.target_kind

    @property
    def display_name(self) -> str:
        return self.public_match.display_name

    @property
    def capabilities(self) -> List[str]:
        return list(self.public_match.capabilities or [])

    @property
    def resource_id(self) -> Optional[str]:
        return self.public_match.resource_id or self.binding_subject

    @property
    def match_reasons(self) -> List[str]:
        return list(self.public_match.match_reasons or [])

# 搜索的最终结果报告
class DiscoveryResponse(BaseModel):

    status: str
    clarification_required: bool = False
    matches: List[PublicTargetMatch] = Field(default_factory=list)
    attached_target_handles: List[str] = Field(default_factory=list)
    toolset_generation: int = 0
    message: Optional[str] = None
    max_selectable: Optional[int] = None
    match_count: int = 0
    truncated: bool = False
    selected_matches: List[DiscoveryCandidate] = Field(default_factory=list, exclude=True)

    def public_dump(self) -> Dict[str, Any]:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload["matches"] = [match.public_dump() for match in self.matches]
        return payload

    @field_validator("selected_matches", mode="before")
    @classmethod
    def _coerce_selected_matches(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        coerced: List[Any] = []
        for item in value:
            if isinstance(item, DiscoveryCandidate):
                coerced.append(item)
            elif isinstance(item, PublicTargetMatch):
                coerced.append(DiscoveryCandidate.from_public_match(item))
            else:
                coerced.append(item)
        return coerced

#正式连接资源的申请，里面塞入了服务端的私有档案 handle_record
class BindingRequest(BaseModel):
    handle_record: TargetHandleRecord
    thread_id: Optional[str] = None
    task_id: Optional[str] = None

# 告诉ToolManager通过那和路径provider_path拿哪个provider_key去加载这个资源
class ProviderLoadRequest(BaseModel):
    provider_path: str
    provider_key: str
    target_handle: str
    provider_context: Dict[str, Any] = Field(default_factory=dict)

# 绑定成功的最终产物。里面包含给 AI 看的通行证（public_summary）以及给底层系统去执行加载的驱动列表（provider_loads）
class BindingResult(BaseModel):
    public_summary: PublicTargetBinding
    provider_loads: List[ProviderLoadRequest] = Field(default_factory=list)
    client_refs: Dict[str, Any] = Field(default_factory=dict)


# =====================================================================
# 💡 语法说明：Protocol 与 鸭子类型 (Duck Typing)
#
# 核心理念：“如果一个动物走起来像鸭子，叫起来像鸭子，那它就是一只鸭子。”
#
# 1. 这是在干嘛？
#    这里是在定“规矩”（接口契约），而不是写具体的业务逻辑。
#
# 2. 为什么不用传统的父类继承？(解耦)
#    如果是传统继承，其他开发者写具体功能时，必须显式地 import 这个类并继承它。
#    用了 Protocol 后，其他人在地球另一边写一个类，只要他写的类里：
#    - 恰好有一个属性叫 `backend_name`
#    - 恰好有一个异步方法叫 `resolve` 且参数和返回值对应上了
#    哪怕他根本没引入当前这个文件，IDE 和 Python 也会承认：“嗯，你符合规矩”。
#
# 3. 方法后面的 `...` 是什么意思？
#    在 Python 里 `...` (Ellipsis) 的作用和 `pass` 一样。
#    代表：“我只规定这个方法叫什么名字、接收什么参数、返回什么结果。
#    至于里面具体怎么查数据，我不关心，留给真正干活的类去写。”
# =====================================================================
class TargetDiscoveryBackend(Protocol):
    backend_name: str
    async def resolve(self, request: DiscoveryRequest) -> DiscoveryResponse: ...

class TargetBindingStrategy(Protocol):
    strategy_name: str
    async def bind(self, request: BindingRequest) -> BindingResult: ...

class AuthenticatedClientFactory(Protocol):
    client_family: str
    async def build(self, handle_record: TargetHandleRecord) -> Any: ...
