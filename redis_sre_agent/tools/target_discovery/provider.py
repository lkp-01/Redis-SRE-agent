"""
AI 把人类语言翻译成真实的服务器编号，并顺手把这台服务器需要的专用工具全部准备好
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from redis_sre_agent.core.targets import (
    bind_target_matches,
    list_known_targets,
    resolve_target_query,
)
from redis_sre_agent.targets.contracts import DiscoveryResponse
from redis_sre_agent.tools.models import ToolCapability, ToolDefinition
from redis_sre_agent.tools.protocols import ToolProvider


class TargetDiscoveryToolProvider(ToolProvider):
    """把自然语言 target 描述解析成安全 Redis target handle。"""

    def __init__(self, redis_instance=None, config=None):
        super().__init__(redis_instance=redis_instance, config=config)
        self._instance_hash = hashlib.sha256(self.provider_name.encode()).hexdigest()[:6]

    @property
    def provider_name(self) -> str:
        return "target_discovery"

    @property
    def requires_redis_instance(self) -> bool:
        return False

    def create_tool_schemas(self) -> List[ToolDefinition]:
        """生成并返回给大模型（LLM）看得懂的工具描述 Schema 列表，定义了工具的名字、用途和参数结构。"""

        return [
            # 工具 1：获取已知的 Redis 目标列表
            ToolDefinition(
                # 动态生成工具名称，通常会加上前缀防止命名冲突
                name=self._make_tool_name("list_known_redis_targets"),
                # 描述工具的用途，大模型会根据这段文字来判断什么时候调用该工具
                description=(
                    "List safe Redis targets currently known in the target catalog. "
                    "This returns public metadata only and does not attach live tools."
                ),
                # 标记工具能力类型为通用工具/实用工具
                capability=ToolCapability.UTILITIES,
                # 定义工具入参的 JSON Schema 标准结构
                parameters={
                    "type": "object",
                    "properties": {
                        # 目标类型：只能是单实例(instance)或集群(cluster)
                        "target_kind": {"type": "string", "enum": ["instance", "cluster"]},
                        # 环境过滤：如 production, staging
                        "environment": {"type": "string"},
                        # 归属能力过滤
                        "capability": {"type": "string"},
                        # 分页限制：默认返回20条，最小1条，最大100条
                        "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                        # 分页偏移量：默认从0开始
                        "offset": {"type": "integer", "default": 0, "minimum": 0},
                        # 是否包含别名，默认不包含
                        "include_aliases": {"type": "boolean", "default": False},
                    },
                },
            ),
            # 工具 2：解析自然语言并绑定 Redis 目标
            ToolDefinition(
                name=self._make_tool_name("resolve_redis_targets"),
                # 描述：将人类的自然语言描述（如“帮我查一下大仓的Redis”）解析为安全的匹配项和不透明的目标句柄
                description=(
                    "Resolve natural-language Redis target descriptions into secret-safe matches "
                    "and opaque target handles."
                ),
                capability=ToolCapability.UTILITIES,
                parameters={
                    "type": "object",
                    "properties": {
                        # 用户输入的查询文本，大模型会把用户的需求提取到这里
                        "query": {"type": "string"},
                        # 是否允许同时匹配并绑定多个 Redis 目标
                        "allow_multiple": {"type": "boolean", "default": False},
                        # 最大结果返回数限制（1~10，默认5）
                        "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                        # 是否在解析成功后自动把针对该 Redis 实例的动态诊断工具（如 info, slowlog）挂载到当前的 Agent 会话中
                        "attach_tools": {"type": "boolean", "default": True},
                        # 偏好的功能列表
                        "preferred_capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    # 声明 query 是必填项，大模型不传此参数则报错
                    "required": ["query"],
                },
            ),
        ]

    async def list_known_redis_targets(
            self,
            target_kind: Optional[str] = None,
            environment: Optional[str] = None,
            capability: Optional[str] = None,
            limit: int = 20,
            offset: int = 0,
            include_aliases: bool = False,
    ) -> Dict[str, Any]:
        """异步执行：获取已知 Redis 目标的具体业务逻辑。"""

        # 安全获取当前的全局上下文管理器 manager
        manager = getattr(self, "_manager", None)
        # 从 manager 中获取当前操作的用户 ID
        user_id = getattr(manager, "user_id", None)
        # 获取当前的工具集版本代数（用于控制动态工具的刷新状态），获取不到则默认为 0
        toolset_generation = manager.get_toolset_generation() if manager else 0

        # 调用底层核心服务，传入过滤和分页参数，异步查询目标列表
        payload = await list_known_targets(
            user_id=user_id,
            target_kind=target_kind,
            environment=environment,
            capability=capability,
            limit=limit,
            offset=offset,
            include_aliases=include_aliases,
        )
        # 将当前的工具集版本塞入返回结果中，供前端或状态机校验
        payload["toolset_generation"] = toolset_generation
        # 返回最终的结构化字典结果
        return payload

    async def resolve_redis_targets(
            self,
            query: str,
            allow_multiple: bool = False,
            max_results: int = 5,
            attach_tools: bool = True,
            preferred_capabilities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """异步执行：将用户的自然语言描述解析并动态绑定到 Redis 实例的业务逻辑。"""

        # 从 manager 提取图流转所需的会话标识：线程 ID、任务 ID 和用户 ID
        manager = getattr(self, "_manager", None)
        thread_id = getattr(manager, "thread_id", None)
        task_id = getattr(manager, "task_id", None)
        user_id = getattr(manager, "user_id", None)

        # 1. 异步调用发现服务，通过 NLP 或特征匹配来查找对应的 Redis 目标实例
        result = await resolve_target_query(
            query=query,
            user_id=user_id,
            allow_multiple=allow_multiple,
            max_results=max_results,
            preferred_capabilities=preferred_capabilities,
        )

        # 初始化已挂载的目标句柄列表
        attached_handles: List[str] = []
        # 初始化作用域变量，用于存储绑定后的动态上下文
        scope = None
        # 默认保留当前的工具集代数
        toolset_generation = manager.get_toolset_generation() if manager else 0

        # 2. 动态绑定逻辑：如果要求挂载工具、manager 正常、且服务成功锁定了目标实例
        if attach_tools and manager and result.status == "resolved" and result.selected_matches:
            # 调用绑定服务，将筛选出的 Redis 目标与当前的线程、任务会话做上下文绑定
            scope = await bind_target_matches(
                matches=result.selected_matches,
                thread_id=thread_id,
                task_id=task_id,
                # 如果不允许同时绑定多个，则会覆盖掉当前会话之前绑定的旧 Redis 实例
                replace_existing=not allow_multiple,
                manager=manager,
            )
            # 从绑定成功的作用域上下文更新中，提取出成功挂载的 Redis 目标唯一句柄（Opaque Handles）
            attached_handles = scope.context_updates.get("attached_target_handles", [])
            # 更新工具集代数（因为成功绑定了新实例，大模型接下来会获得操作这个实例的新工具，如 info/slowlog，即工具集发生了进化）
            toolset_generation = scope.toolset_generation

        # 3. 序列化输出：如果是特定的 DiscoveryResponse 对象则使用 public_dump 过滤掉敏感凭据，否则使用标准 Pydantic 的 model_dump
        payload = (
            result.public_dump() if isinstance(result, DiscoveryResponse) else result.model_dump()
        )
        # 将刚刚动态生成的句柄和最新的工具集版本号注入到输出 Payload 中
        payload["attached_target_handles"] = attached_handles
        payload["toolset_generation"] = toolset_generation

        # 4. 上下文合并：如果生成了新作用域，将除了具体实例 ID 以外的上下文更新合并到返回的 payload 中（对外隐藏敏感物理 ID）
        if scope is not None:
            payload.update(
                {
                    key: value
                    for key, value in scope.context_updates.items()
                    if key not in {"instance_id", "cluster_id"}
                }
            )
        # 返回供大模型或 ToolMessage 消费的最终字典数据
        return payload
