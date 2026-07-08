"""
拿着档案去查redis数据库的连接信息查出来
为AI组出来一套针对这个数据库的工具
"""

from __future__ import annotations

from typing import Any, List

from redis_sre_agent.core.config import settings
from redis_sre_agent.core.instances import RedisInstance, get_instance_by_id

from .contracts import BindingRequest, BindingResult, ProviderLoadRequest, TargetHandleRecord

"""下面这俩测试函数倒是很有意思，可以留意一下"""
# 测试用的找找有没有测试种子
def _eval_target_seed(handle_record: TargetHandleRecord) -> dict[str, Any] | None:
    seed = (handle_record.private_binding_ref or {}).get("eval_target_seed")
    return seed if isinstance(seed, dict) else None

# 如果找到了测试种子，就会用这些假数据，造一个假的RedisInstance虚拟库对象
def _build_seeded_instance(handle_record: TargetHandleRecord) -> RedisInstance | None:
    seed = _eval_target_seed(handle_record)
    if not seed or seed.get("seed_kind") != "instance":
        return None
    payload = dict(seed)
    payload["id"] = handle_record.target_handle
    payload.setdefault("created_by", "agent")
    payload.setdefault("user_id", "eval")
    return RedisInstance.model_validate(payload)

#把一个 TargetHandleRecord 变成工具可用的 Redis 实例对象。
class RedisDataClientFactory:

    client_family = "redis.data"

    async def build(self, handle_record: TargetHandleRecord) -> Any:
        if handle_record.public_summary.target_kind != "instance":
            return None
        instance = await get_instance_by_id(handle_record.binding_subject)
        if instance is None:
            return _build_seeded_instance(handle_record)
        return instance.model_copy(update={"id": handle_record.target_handle})

#把“已经选中的目标”绑定成当前 ToolManager 可以加载的工具。
#具体代码没看懂
class RedisTargetBindingStrategy:

    strategy_name = "redis_default"

    async def bind(self, request: BindingRequest) -> BindingResult:
        from .registry import get_target_integration_registry

        registry = get_target_integration_registry()
        handle_record = request.handle_record
        public_summary = handle_record.public_summary
        provider_loads: List[ProviderLoadRequest] = []
        client_refs: dict[str, Any] = {}

        if public_summary.target_kind == "instance":
            data_instance = await registry.get_client_factory("redis.data").build(handle_record)
            if data_instance is not None:
                client_refs["redis.data"] = data_instance
                for provider_path in settings.tool_providers:
                    provider_loads.append(
                        ProviderLoadRequest(
                            provider_path=provider_path,
                            provider_key=f"target:{public_summary.target_handle}:{provider_path}",
                            target_handle=public_summary.target_handle,
                            provider_context={"redis_instance_override": data_instance},
                        )
                    )

        return BindingResult(
            public_summary=public_summary,
            provider_loads=provider_loads,
            client_refs=client_refs,
        )
