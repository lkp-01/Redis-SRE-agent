"""
拿着档案去查redis数据库的连接信息查出来
为AI组出来一套针对这个数据库的工具
"""

from __future__ import annotations

from typing import Any, List

from redis_sre_agent.core.config import settings
from redis_sre_agent.core.instances import RedisInstance, get_instance_by_id

from .contracts import BindingRequest, BindingResult, ProviderLoadRequest, TargetHandleRecord

#把一个 TargetHandleRecord 变成工具可用的 Redis 实例对象。
class RedisDataClientFactory:

    client_family = "redis.data"

    async def build(self, handle_record: TargetHandleRecord) -> Any:
        if handle_record.public_summary.target_kind != "instance":
            return None
        instance = await get_instance_by_id(handle_record.binding_subject)
        if instance is None:
            return None
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
