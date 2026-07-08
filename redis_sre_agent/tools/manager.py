"""工具生命周期和路由管理。

这是原项目 ToolManager 的阶段三裁剪版。保留 provider 加载、路由表、动态 target
绑定和工具调用缓存；审批、MCP、support package、observability 等平台能力留作后续插槽。
"""

from __future__ import annotations

import importlib
import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from redis_sre_agent.core.clusters import RedisCluster
from redis_sre_agent.core.config import settings
from redis_sre_agent.core.instances import RedisInstance, get_instance_by_id
from redis_sre_agent.targets import get_target_handle_store, get_target_integration_registry
from redis_sre_agent.targets.contracts import BindingRequest, ProviderLoadRequest
from redis_sre_agent.tools.models import Tool, ToolActionKind, ToolCapability, ToolDefinition
from redis_sre_agent.tools.protocols import ToolProvider

logger = logging.getLogger(__name__)

_DEFAULT_LLM_TOOL_LIMIT = 64
ToolExecutionDecision = Any


class ToolManager:
    """管理 provider 生命周期，并把工具名路由到真实 provider 方法。"""

    _provider_class_cache: Dict[str, type] = {}
    _always_on_providers = [
        "redis_sre_agent.tools.target_discovery.provider.TargetDiscoveryToolProvider",
    ]

    def __init__(
        self,
        redis_instance: Optional[RedisInstance] = None,
        redis_cluster: Optional["RedisCluster"] = None,
        initial_target_bindings: Optional[List[Any]] = None,
        initial_toolset_generation: Optional[int] = None,
        exclude_mcp_categories: Optional[List[ToolCapability]] = None,
        support_package_path: Optional[Path] = None,
        cache_client: Optional[Any] = None,
        cache_ttl_overrides: Optional[Dict[str, int]] = None,
        thread_id: Optional[str] = None,
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        graph_type: str = "agent_turn",
        graph_version: str = "v1",
    ):
        self.redis_instance = redis_instance
        self.redis_cluster = redis_cluster
        self._initial_target_bindings = list(initial_target_bindings or [])
        self._initial_toolset_generation = initial_toolset_generation
        self.exclude_mcp_categories = exclude_mcp_categories
        self.support_package_path = support_package_path
        self.thread_id = thread_id
        self.task_id = task_id
        self.user_id = user_id
        self.graph_type = graph_type
        self.graph_version = graph_version
        self._loaded_provider_keys: set[str] = set()
        self._routing_table: Dict[str, ToolProvider] = {}
        self._tools: List[Tool] = []
        self._tool_by_name: Dict[str, Tool] = {}
        self._providers: List[ToolProvider] = []
        self._stack: Optional[AsyncExitStack] = None
        self._call_cache: Dict[str, Any] = {}
        self._attached_target_bindings: Dict[str, Any] = {}
        self._toolset_generation = 0
        self._shared_cache = None
        if cache_client is not None and redis_instance is not None:
            from redis_sre_agent.tools.cache import ToolCache

            self._shared_cache = ToolCache(
                redis_client=cache_client,
                instance_id=redis_instance.id,
                ttl_overrides=cache_ttl_overrides,
            )

    @staticmethod
    async def resolve_redis_enterprise_admin_instance(
        redis_instance: RedisInstance,
    ) -> tuple[RedisInstance, str]:
        """Redis Enterprise admin credential resolver 的阶段三插槽。"""
        return redis_instance, "stage3_slot"

    async def __aenter__(self) -> "ToolManager":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for provider_path in self._always_on_providers:
            await self._load_provider(provider_path, always_on=True)
        await self._load_mcp_providers()
        await self._load_support_package_provider()

        if self.redis_instance:
            await self._load_instance_scoped_providers(self.redis_instance)
        elif self.redis_cluster:
            await self._load_cluster_scoped_providers(self.redis_cluster)
        elif self._initial_target_bindings:
            await self.attach_bound_targets(
                self._initial_target_bindings,
                generation=self._initial_toolset_generation,
            )
        else:
            await self._load_thread_attached_targets()

        self._toolset_generation = max(1, self._toolset_generation)
        return self

    async def __aexit__(self, *args) -> None:
        try:
            if self._stack is not None:
                await self._stack.__aexit__(*args)
        finally:
            for provider in list(self._providers):
                try:
                    setattr(provider, "_manager", None)
                except Exception:
                    pass
            self._routing_table.clear()
            self._tools.clear()
            self._tool_by_name.clear()
            self._providers.clear()
            self._loaded_provider_keys.clear()
            self._call_cache.clear()
            self._attached_target_bindings.clear()
            self._stack = None

    async def _load_thread_attached_targets(self) -> None:
        """线程已绑定 target 恢复的阶段三 no-op 插槽。"""
        return None

    async def _load_instance_scoped_providers(
        self,
        redis_instance: RedisInstance,
        *,
        load_key_prefix: Optional[str] = None,
    ) -> RedisInstance:
        for provider_path in settings.tool_providers:
            await self._load_provider(
                provider_path,
                redis_instance_override=redis_instance,
                load_key=f"{load_key_prefix or redis_instance.id}:{provider_path}",
            )
        return redis_instance

    async def _load_cluster_scoped_providers(
        self, redis_cluster: "RedisCluster"
    ) -> None:
        """cluster-only provider 加载的阶段三 no-op 插槽。"""
        logger.info(
            "Cluster-scoped providers are stage-three slots only: %s",
            redis_cluster.name,
        )
        return None

    @staticmethod
    def build_redis_enterprise_admin_instance_from_cluster(
        redis_cluster: "RedisCluster",
        *,
        target_id_override: Optional[str] = None,
    ) -> Optional[RedisInstance]:
        """Redis Enterprise cluster admin instance 构造的阶段三插槽。"""
        return None

    async def _load_provider(
        self,
        provider_path: str,
        always_on: bool = False,
        redis_instance_override: Optional[RedisInstance] = None,
        load_key: Optional[str] = None,
    ) -> None:
        if self._stack is None:
            raise RuntimeError("ToolManager must be entered before loading providers.")
        provider_key = load_key or provider_path
        if provider_key in self._loaded_provider_keys:
            return
        try:
            provider_cls = self._get_provider_class(provider_path)
            instance = None if always_on else (redis_instance_override or self.redis_instance)
            provider = await self._stack.enter_async_context(provider_cls(redis_instance=instance))
            try:
                setattr(provider, "_manager", self)
            except Exception:
                pass
            tools = provider.tools()
            for tool in tools:
                name = tool.metadata.name
                if not name:
                    continue
                self._routing_table[name] = provider
                self._tools.append(tool)
                self._tool_by_name[name] = tool
            self._providers.append(provider)
            self._loaded_provider_keys.add(provider_key)
        except Exception:
            logger.exception("Failed to load provider %s", provider_path)

    async def _load_provider_request(self, request: ProviderLoadRequest) -> None:
        provider_context = request.provider_context or {}
        await self._load_provider(
            request.provider_path,
            always_on=bool(provider_context.get("always_on", False)),
            redis_instance_override=provider_context.get("redis_instance_override"),
            load_key=request.provider_key,
        )

    async def _load_mcp_providers(self) -> None:
        """MCP provider 加载的阶段三 no-op 插槽。"""
        return None

    async def _load_support_package_provider(self) -> None:
        """support package provider 加载的阶段三 no-op 插槽。"""
        return None

    @staticmethod
    def _build_target_scoped_instance(
        redis_instance: RedisInstance, target_handle: str
    ) -> RedisInstance:
        return redis_instance.model_copy(update={"id": target_handle})

    def get_toolset_generation(self) -> int:
        return self._toolset_generation

    def get_attached_target_bindings(self) -> List[Any]:
        return list(self._attached_target_bindings.values())

    async def attach_bound_targets(
        self,
        bindings: List[Any],
        *,
        generation: Optional[int] = None,
    ) -> List[Any]:
        new_attachment = False
        attached: List[Any] = []
        handle_store = get_target_handle_store()
        registry = get_target_integration_registry()
        requested_handles = [
            getattr(binding, "target_handle", None)
            for binding in bindings or []
            if getattr(binding, "target_handle", None)
        ]
        handle_records = await handle_store.get_records(requested_handles)

        for binding in bindings or []:
            target_handle = getattr(binding, "target_handle", None)
            target_kind = getattr(binding, "target_kind", None)
            resource_id = getattr(binding, "resource_id", None)
            if not target_handle or not target_kind:
                continue
            existing = self._attached_target_bindings.get(target_handle)
            if existing is not None:
                attached.append(existing)
                continue

            handle_record = handle_records.get(target_handle)
            if handle_record is not None:
                binding_result = await registry.get_binding_strategy(
                    handle_record.binding_strategy
                ).bind(
                    BindingRequest(
                        handle_record=handle_record,
                        thread_id=self.thread_id,
                        task_id=self.task_id,
                    )
                )
                for provider_load in binding_result.provider_loads:
                    await self._load_provider_request(provider_load)
                if binding_result.provider_loads:
                    self._attached_target_bindings[target_handle] = binding_result.public_summary
                    attached.append(binding_result.public_summary)
                    new_attachment = True
                continue

            if target_kind == "instance" and resource_id:
                instance = await get_instance_by_id(resource_id)
                if instance is None:
                    continue
                scoped_instance = self._build_target_scoped_instance(instance, target_handle)
                await self._load_instance_scoped_providers(
                    scoped_instance,
                    load_key_prefix=f"target:{target_handle}",
                )
                self._attached_target_bindings[target_handle] = binding
                attached.append(binding)
                new_attachment = True

        if generation is not None:
            self._toolset_generation = max(self._toolset_generation, generation)
        elif new_attachment:
            self._toolset_generation += 1
        return attached

    @classmethod
    def _get_provider_class(cls, provider_path: str) -> type:
        if provider_path not in cls._provider_class_cache:
            module_path, class_name = provider_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
            cls._provider_class_cache[provider_path] = provider_class
        return cls._provider_class_cache[provider_path]

    def get_tools(self) -> List[ToolDefinition]:
        return [tool.definition for tool in self._tools]

    @staticmethod
    def _llm_tool_priority(tool: Tool) -> tuple[int, int, str]:
        provider_name = str(tool.metadata.provider_name or "")
        capability = tool.metadata.capability
        if provider_name == "target_discovery":
            priority = 0
        elif provider_name == "redis_command":
            priority = 1
        elif capability is ToolCapability.UTILITIES:
            priority = 3
        elif capability is ToolCapability.KNOWLEDGE:
            priority = 4
        elif capability is ToolCapability.DIAGNOSTICS:
            priority = 5
        elif capability is ToolCapability.METRICS:
            priority = 6
        elif capability is ToolCapability.LOGS:
            priority = 7
        else:
            priority = 10
        return (priority, 0, tool.metadata.name)

    def get_tools_for_llm(self, *, max_tools: int = _DEFAULT_LLM_TOOL_LIMIT) -> List[ToolDefinition]:
        if max_tools <= 0:
            return []
        if len(self._tools) <= max_tools:
            return self.get_tools()
        return [tool.definition for tool in sorted(self._tools, key=self._llm_tool_priority)[:max_tools]]

    def get_tools_by_provider_names(self, provider_names: List[str]) -> List[ToolDefinition]:
        wanted = {str(name).lower() for name in (provider_names or [])}
        if not wanted:
            return []
        return [
            tool.definition
            for tool in self._tools
            if self._routing_table.get(tool.metadata.name)
            and self._routing_table[tool.metadata.name].provider_name.lower() in wanted
        ]

    def get_status_update(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        provider = self._routing_table.get(tool_name)
        if not provider:
            return None
        return provider.get_status_update(tool_name, args)

    def get_providers_for_capability(self, capability: Any) -> List[ToolProvider]:
        if not isinstance(capability, ToolCapability):
            raise ValueError(f"Invalid capability: {capability}")
        seen: set[int] = set()
        providers: List[ToolProvider] = []
        for tool in self._tools:
            if tool.metadata.capability is not capability:
                continue
            provider = self._routing_table.get(tool.metadata.name)
            if not provider or id(provider) in seen:
                continue
            seen.add(id(provider))
            providers.append(provider)
        return providers

    def get_providers_for_protocol(self, protocol_cls: Any) -> List[ToolProvider]:
        """按 Protocol 获取 provider 的阶段三兼容入口。"""
        seen: set[int] = set()
        providers: List[ToolProvider] = []
        for provider in self._routing_table.values():
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            try:
                if isinstance(provider, protocol_cls):
                    providers.append(provider)
            except Exception:
                continue
        return providers

    def _filter_tools_by_providers(self, providers: List[ToolProvider]) -> List[ToolDefinition]:
        provider_ids = {id(provider) for provider in providers or []}
        return [
            tool.definition
            for tool in self._tools
            if id(self._routing_table.get(tool.metadata.name)) in provider_ids
        ]

    def get_tools_for_capability(self, capability: Any) -> List[ToolDefinition]:
        providers = self.get_providers_for_capability(capability)
        return self._filter_tools_by_providers(providers)

    def get_tools_for_protocol(self, protocol_cls: Any) -> List[ToolDefinition]:
        providers = self.get_providers_for_protocol(protocol_cls)
        return self._filter_tools_by_providers(providers)

    def get_provider_for_capability(self, capability: Any) -> Optional[ToolProvider]:
        providers = self.get_providers_for_capability(capability)
        return providers[0] if providers else None

    def _target_handles_for_policy(self) -> List[str]:
        """审批策略 target scope 的阶段三兼容 helper。"""
        handles = [
            getattr(binding, "target_handle", None)
            for binding in self.get_attached_target_bindings()
            if getattr(binding, "target_handle", None)
        ]
        if handles:
            return [str(handle) for handle in handles]
        if self.redis_instance is not None and getattr(self.redis_instance, "id", None):
            return [str(self.redis_instance.id)]
        if self.redis_cluster is not None and getattr(self.redis_cluster, "id", None):
            return [str(self.redis_cluster.id)]
        return []

    async def evaluate_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> ToolExecutionDecision:
        """审批策略的阶段三 no-op 插槽。"""
        return None

    async def resolve_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        decision: Optional[ToolExecutionDecision] = None,
    ) -> Any:
        normalized_args = dict(args or {})
        provider = self._routing_table.get(tool_name)
        tool = self._tool_by_name.get(tool_name)
        if not provider or not tool:
            available_tools = list(self._routing_table.keys())
            raise ValueError(
                f"Unknown tool: {tool_name}. Available tools ({len(available_tools)}): "
                f"{available_tools[:10]}..."
            )

        try:
            args_key = json.dumps(normalized_args, sort_keys=True, separators=(",", ":"))
        except Exception:
            args_key = str(normalized_args)
        cache_key = f"{tool_name}|{args_key}"
        cacheable = tool.metadata.action_kind is ToolActionKind.READ
        if cacheable and cache_key in self._call_cache:
            return self._call_cache[cache_key]
        if cacheable and self._shared_cache:
            cached_result = await self._shared_cache.get(tool_name, normalized_args)
            if cached_result is not None:
                self._call_cache[cache_key] = cached_result
                return cached_result

        result = await tool.invoke(normalized_args)
        if cacheable:
            self._call_cache[cache_key] = result
            if self._shared_cache:
                await self._shared_cache.set(tool_name, normalized_args, result)
        return result

    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        results: List[Any] = []
        for tool_call in tool_calls or []:
            try:
                name = tool_call.get("name")
                if not name and isinstance(tool_call.get("function"), dict):
                    name = tool_call["function"].get("name")
                args = tool_call.get("args")
                if args is None and isinstance(tool_call.get("function"), dict):
                    arguments = tool_call["function"].get("arguments")
                    if isinstance(arguments, str):
                        try:
                            args = json.loads(arguments or "{}")
                        except Exception:
                            args = {}
                    elif isinstance(arguments, dict):
                        args = arguments
                if not isinstance(args, dict):
                    args = {}
                if not name:
                    results.append({"status": "failed", "error": "missing tool name"})
                    continue
                results.append(await self.resolve_tool_call(name, args))
            except Exception as exc:
                logger.exception("Tool call execution failed for %s", tool_call)
                results.append({"status": "failed", "error": str(exc)})
        return results
