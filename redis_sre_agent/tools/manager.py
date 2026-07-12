"""工具生命周期和路由管理。

这是原项目 ToolManager 的阶段三裁剪版。保留 provider 加载、路由表、动态 target
绑定和工具调用缓存；审批、MCP、support package、observability 等平台能力留作后续插槽。
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import shutil
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
_REDACTED = "[REDACTED]"


def _safe_error_message(exc: Exception) -> str:
    """清洗工具异常文本，避免 manager 兜底路径泄漏连接串或凭据。"""

    message = str(exc)
    message = re.sub(
        r"(?i)\b(rediss?|unix)://[^\s'\"<>@]+@[^\s'\"<>]+",
        lambda match: re.sub(r"://([^:@/]+):[^@/]+@", r"://\1:[REDACTED]@", match.group(0)),
        message,
    )
    message = re.sub(
        r"(?i)\b(password|secret|token|requirepass|masterauth|pass)(\s*[=:]\s*)([^\s,;}]+)",
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        message,
    )
    return message


def _command_is_available(command: Optional[str]) -> bool:
    """按 original 形状在启动子进程前检查命令是否存在。"""

    if command is None:
        return True
    candidate = command.strip()
    if not candidate:
        return False
    if Path(candidate).is_absolute() or "/" in candidate or "\\" in candidate:
        return Path(candidate).is_file()
    return shutil.which(candidate) is not None


def _missing_local_mcp_arg_path(args: Optional[List[str]]) -> Optional[str]:
    """返回第一个明确指向本地、但并不存在的 MCP 入口文件。"""

    for raw_arg in args or []:
        arg = str(raw_arg or "").strip()
        if not arg or arg.startswith("-") or any(character.isspace() for character in arg):
            continue
        if arg.startswith("file://"):
            candidate = Path(arg.removeprefix("file://")).expanduser()
        elif "://" in arg:
            continue
        else:
            path = Path(arg).expanduser()
            looks_like_script = path.suffix.lower() in {
                ".js",
                ".mjs",
                ".cjs",
                ".ts",
                ".py",
                ".sh",
            }
            if not (path.is_absolute() or arg.startswith(("./", "../", "~")) or looks_like_script):
                continue
            candidate = path
        if not candidate.exists():
            return str(candidate)
    return None


class ToolManager:
    """管理 provider 生命周期，并把工具名路由到真实 provider 方法。"""

    _provider_class_cache: Dict[str, type] = {}
    _always_on_providers = [
        "redis_sre_agent.tools.target_discovery.provider.TargetDiscoveryToolProvider",
    ]
    _knowledge_provider = (
        "redis_sre_agent.tools.knowledge.knowledge_base.KnowledgeBaseToolProvider"
    )

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
        from redis_sre_agent.core.redis import RAGReadiness

        self.rag_readiness = RAGReadiness(
            state="disabled",
            reason_code="disabled",
            message="RAG 未启用。",
        )
        if cache_client is not None and redis_instance is not None:
            from redis_sre_agent.tools.cache import ToolCache

            self._shared_cache = ToolCache(
                redis_client=cache_client,
                instance_id=redis_instance.id,
                ttl_overrides=cache_ttl_overrides,
            )

    #触发初始工具集和提供者（Provider）的加载
    async def __aenter__(self) -> "ToolManager":
        # 初始化一个异步退出栈，用于集中管理后续实例化的各个 Provider 的生命周期，确保退出时能安全释放
        self._stack = AsyncExitStack()

        # 启动并进入这个异步退出栈
        await self._stack.__aenter__()

        # 遍历预设的“常驻工具提供者”（如目标发现工具、知识库工具）
        for provider_path in self._always_on_providers:
            # 异步加载这些常驻 Provider，并标记 always_on=True，表示它们不依赖特定目标
            await self._load_provider(provider_path, always_on=True)

        # knowledge provider 不再常驻。关闭时完全不做 readiness/Redis/embedding 检查；
        # 开启后也只有 ready 才把 search 工具暴露给 LLM。
        if settings.rag_enabled:
            from redis_sre_agent.core.redis import get_rag_readiness

            self.rag_readiness = await get_rag_readiness(settings)
            if self.rag_readiness.ready:
                await self._load_provider(self._knowledge_provider, always_on=True)
            else:
                logger.warning(
                    "RAG knowledge provider 未加载：%s",
                    self.rag_readiness.reason_code,
                )

        # 触发 MCP (Model Context Protocol) 相关的 Provider 加载（当前为预留的插槽方法）
        await self._load_mcp_providers()

        # 触发支持包 (Support Package) 相关的 Provider 加载（当前为预留的插槽方法）
        await self._load_support_package_provider()

        # [上下文感知路由]：根据当前管理器绑定的不同上下文，加载专属工具
        if self.redis_instance:
            # 场景A：如果明确指定了一个 Redis 实例，则加载针对该实例的工具
            await self._load_instance_scoped_providers(self.redis_instance)

        elif self.redis_cluster:
            # 场景B：如果没有实例但指定了 Redis 集群，则加载针对该集群的工具（当前为预留插槽）
            await self._load_cluster_scoped_providers(self.redis_cluster)

        elif self._initial_target_bindings:
            # 场景C：如果通过参数传入了初始的目标绑定列表，则解析并加载这些目标的工具
            await self.attach_bound_targets(
                self._initial_target_bindings,
                generation=self._initial_toolset_generation,  # 传入工具集的代数/版本
            )

        else:
            # 场景D：如果上述都没指定，尝试从当前线程（Thread Context）恢复并加载之前绑定的工具状态
            await self._load_thread_attached_targets()

        # 确保工具集代数（版本号）最小值为 1，用于追踪工具列表的变更
        self._toolset_generation = max(1, self._toolset_generation)

        # 返回 ToolManager 实例自身，供 `async with ToolManager(...) as manager:` 中的 manager 变量使用
        return self

    #过路径把工具箱变成对象缓存起来，还把里面所有工具的“名字与对象的映射关系”登记到了路由表里，方便后续直接按名字调用。
    async def _load_provider(
            self,
            provider_path: str,  # Provider 类的 Python import 路径
            always_on: bool = False,  # 是否是全局常驻（不依赖特定实例）的 Provider
            redis_instance_override: Optional[RedisInstance] = None,  # 覆盖传入的实例对象
            load_key: Optional[str] = None,  # 用于去重的唯一键
    ) -> None:
        # 1. 【防呆检查】大管家自己必须先通过 `async with` 启动，有了垃圾桶（_stack）才能装插件
        if self._stack is None:
            raise RuntimeError("ToolManager must be entered before loading providers.")

        # 2. 【生成身份证】如果没有传唯一的 key，就默认用它的类路径当做它的“身份证号”
        provider_key = load_key or provider_path

        # 3. 【去重检查】如果这个身份证号在“已安装列表”里，说明装过了，直接打道回府
        if provider_key in self._loaded_provider_keys:
            return

        try:
            # 4. 【顺藤摸瓜】利用 Python 的反射机制，把字符串路径变成真正的“插件工场类”
            provider_cls = self._get_provider_class(provider_path)

            # 5. 【分配连接】如果是通用工具（always_on）就不给它连 Redis；
            #    否则看看有没有单独指定的 Redis，没有就用大管家默认的 Redis 实例
            instance = None if always_on else (redis_instance_override or self.redis_instance)

            # 6. 【买回并接电】实例化这个插件，并把它丢进异步垃圾桶（_stack）里托管。
            #    这样大管家死的时候，这个插件占用的 Redis 连接也能自动断开，不会内存泄漏。
            provider = await self._stack.enter_async_context(provider_cls(redis_instance=instance))

            # 7. 【认个门】尝试把大管家（self）反向塞给插件，让插件以后有需要能找大管家帮忙
            try:
                setattr(provider, "_manager", self)
            except Exception:
                pass

            # 8. 【开箱检阅】把这个插件箱里自带的所有小工具全部掏出来
            tools = provider.tools()

            # 9. 【登记名册】遍历每一个小工具，把它们登记到大管家的各个核心名册里
            for tool in tools:
                name = tool.metadata.name
                if not name:
                    continue
                # 核心步骤 A：登记到“路由表”（以后大模型喊工具名时，靠它一秒找到对应的插件箱）
                self._routing_table[name] = provider
                # 核心步骤 B：加进全局工具大列表
                self._tools.append(tool)
                # 核心步骤 C：做成“工具名 -> 工具对象”的快速查阅字典
                self._tool_by_name[name] = tool

            # 10. 【大功告成】把这个装好的插件箱放进已装插件列表
            self._providers.append(provider)
            # 11. 【打上已装标记】把它的身份证号存进集合，防止下次重复安装
            self._loaded_provider_keys.add(provider_key)

        except Exception:
            # 12. 【安全兜底】如果安装过程中任何一步崩了，只悄悄记个日志，绝对不抛异常让整个 Agent 系统死机
            logger.exception("Failed to load provider %s", provider_path)

    @classmethod
    def _get_provider_class(cls, provider_path: str) -> type:
        # 检查缓存中是否已经加载过这个类的引用，避免重复反射带来的性能开销
        if provider_path not in cls._provider_class_cache:
            # 从右侧按 "." 拆分一次，将路径分为模块路径和类名
            # 例如 "my_pkg.my_module.MyClass" 会被拆分为 "my_pkg.my_module" 和 "MyClass"
            module_path, class_name = provider_path.rsplit(".", 1)

            # 动态导入该模块
            module = importlib.import_module(module_path)

            # 从模块中获取对应的类对象
            provider_class = getattr(module, class_name)

            # 将获取到的类对象放入类级别的缓存字典中
            cls._provider_class_cache[provider_path] = provider_class

        # 返回类对象
        return cls._provider_class_cache[provider_path]

    async def _load_mcp_providers(self) -> None:
        """加载当前 turn 独占的只读 MCP provider，失败不阻断内建诊断。"""

        mcp_servers = settings.mcp_servers
        if not mcp_servers:
            return None
        if self._stack is None:
            raise RuntimeError("ToolManager must be entered before loading MCP providers.")

        from redis_sre_agent.core.config import MCPServerConfig
        from redis_sre_agent.tools.mcp.provider import MCPToolProvider

        excluded_capabilities = set(self.exclude_mcp_categories or [])
        for server_name, raw_config in mcp_servers.items():
            provider_key = f"mcp:{server_name}"
            if provider_key in self._loaded_provider_keys:
                continue
            try:
                server_config = (
                    raw_config
                    if isinstance(raw_config, MCPServerConfig)
                    else MCPServerConfig.model_validate(raw_config)
                )
            except Exception:
                logger.warning("External MCP provider skipped: mcp_config_invalid")
                continue

            if server_config.command and not _command_is_available(server_config.command):
                logger.warning("External MCP provider skipped: mcp_command_unavailable")
                continue
            if _missing_local_mcp_arg_path(server_config.args) is not None:
                logger.warning("External MCP provider skipped: mcp_entrypoint_unavailable")
                continue

            try:
                provider = MCPToolProvider(
                    server_name=str(server_name),
                    server_config=server_config,
                    redis_instance=None,
                )
                provider = await self._stack.enter_async_context(provider)
                discovered_tools = provider.tools()
                candidates = [
                    tool
                    for tool in discovered_tools
                    if tool.metadata.action_kind is ToolActionKind.READ
                    and tool.metadata.capability not in excluded_capabilities
                ]

                candidate_names = [tool.metadata.name for tool in candidates]
                has_invalid_name = any(not name for name in candidate_names)
                has_batch_conflict = len(candidate_names) != len(set(candidate_names))
                has_existing_conflict = any(
                    name in self._routing_table or name in self._tool_by_name
                    for name in candidate_names
                )
                if has_invalid_name or has_batch_conflict or has_existing_conflict:
                    logger.warning("External MCP provider skipped: mcp_name_conflict")
                    continue

                try:
                    setattr(provider, "_manager", self)
                except Exception:
                    pass
                for tool in candidates:
                    name = tool.metadata.name
                    self._routing_table[name] = provider
                    self._tools.append(tool)
                    self._tool_by_name[name] = tool
                self._providers.append(provider)
                self._loaded_provider_keys.add(provider_key)
            except Exception:
                logger.warning("External MCP provider skipped: mcp_provider_unavailable")
                continue
        return None

    async def _load_support_package_provider(self) -> None:
        """support package provider 加载的阶段三 no-op 插槽。"""
        return None

    #遍历系统全局配置（settings）中注册的所有工具提供者路径（如 metrics_provider, command_provider 等）
    async def _load_instance_scoped_providers(
            self,
            redis_instance: RedisInstance,  # 当前需要绑定工具的 Redis 实例对象
            *,
            load_key_prefix: Optional[str] = None,  # 可选的前缀，用于生成全局唯一的加载键，防止重复加载
    ) -> RedisInstance:
        for provider_path in settings.tool_providers:
            # 调用底层的 _load_provider 执行实际的加载逻辑
            await self._load_provider(
                provider_path,
                # 强制覆盖 Provider 绑定的实例为当前传入的实例
                redis_instance_override=redis_instance,
                # 构造唯一的去重 Key，格式通常为 "instance_id:provider_path"
                load_key=f"{load_key_prefix or redis_instance.id}:{provider_path}",
            )
        # 返回处理完成的实例对象
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

    async def _load_provider_request(self, request: ProviderLoadRequest) -> None:
        provider_context = request.provider_context or {}
        await self._load_provider(
            request.provider_path,
            always_on=bool(provider_context.get("always_on", False)),
            redis_instance_override=provider_context.get("redis_instance_override"),
            load_key=request.provider_key,
        )

    @staticmethod
    def _build_target_scoped_instance(
        redis_instance: RedisInstance, target_handle: str
    ) -> RedisInstance:
        return redis_instance.model_copy(update={"id": target_handle})

    # 从 Thread context 恢复已绑定 target，并加载对应 provider。
    async def _load_thread_attached_targets(self) -> None:

        # 如果当前 Manager 实例化时没有传入 thread_id，说明不在特定的对话上下文中，直接返回
        if not self.thread_id:
            return None

        try:
            # 延迟导入获取线程目标状态的方法，避免循环依赖或过早加载
            from redis_sre_agent.core.targets import get_thread_target_state

            # 根据 thread_id 从外部存储（如 Redis 或数据库）读取当前线程绑定的目标状态
            state = await get_thread_target_state(self.thread_id)
        except Exception:
            # 如果读取状态失败，记录日志并静默失败，避免阻断核心流程
            logger.exception("Failed to load attached targets for thread %s", self.thread_id)
            return None

        # 如果该线程上下文中没有任何绑定的目标，直接返回
        if not state.target_bindings:
            return None

        # 核心步骤：将恢复出的目标绑定列表传入 attach_bound_targets 方法
        # generation 参数用于同步工具集的代数（版本），让大模型感知到工具列表发生了变化
        await self.attach_bound_targets(
            state.target_bindings,
            generation=state.target_toolset_generation or None,
        )
        return None

    #清空路由表、缓存和已加载的 Provider
    async def __aexit__(self, *args) -> None:
        try:
            # 检查异步退出栈是否已经初始化
            if self._stack is not None:
                # 触发栈内所有受管对象（如各个 Provider）的退出清理逻辑，按入栈的相反顺序执行
                await self._stack.__aexit__(*args)

        finally:
            # 无论上面的退出栈清理是否成功或抛出异常，finally 块确保本地缓存和状态被强制清空

            # 将 _providers 转换为列表进行遍历，避免在迭代过程中修改原集合
            for provider in list(self._providers):
                try:
                    # 解除 Provider 对当前 ToolManager 实例的引用，防止循环引用导致内存泄漏
                    setattr(provider, "_manager", None)
                except Exception:
                    # 如果解绑失败（比如属性不存在），直接忽略，继续下一个
                    pass

            # 清空“工具名称 -> Provider”的路由映射表
            self._routing_table.clear()

            # 清空所有的工具定义列表
            self._tools.clear()

            # 清空“工具名称 -> 工具实例”的索引字典
            self._tool_by_name.clear()

            # 清空已加载的 Provider 实例列表
            self._providers.clear()

            # 清空已加载的 Provider 路径标识集合（用于防止重复加载）
            self._loaded_provider_keys.clear()

            # 清空当前生命周期内产生的工具调用结果缓存
            self._call_cache.clear()

            # 清空已绑定的外部目标记录
            self._attached_target_bindings.clear()

            # 销毁对异步退出栈的引用，彻底释放资源
            self._stack = None

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
                results.append({"status": "failed", "error": _safe_error_message(exc)})
        return results

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

    def get_tools_for_llm(self, *, max_tools: int = _DEFAULT_LLM_TOOL_LIMIT) -> List[ToolDefinition]:
        if max_tools <= 0:
            return []
        if len(self._tools) <= max_tools:
            return self.get_tools()
        return [tool.definition for tool in sorted(self._tools, key=self._llm_tool_priority)[:max_tools]]

    def get_tools(self) -> List[ToolDefinition]:
        return [tool.definition for tool in self._tools]

    @staticmethod
    def _llm_tool_priority(tool: Tool) -> tuple[int, int, str]:
        provider_name = str(tool.metadata.provider_name or "")
        capability = tool.metadata.capability
        if provider_name.startswith("mcp_"):
            priority = 100
        elif provider_name == "target_discovery":
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

    def get_tools_for_capability(self, capability: Any) -> List[ToolDefinition]:
        providers = self.get_providers_for_capability(capability)
        return self._filter_tools_by_providers(providers)

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

    def _filter_tools_by_providers(self, providers: List[ToolProvider]) -> List[ToolDefinition]:
        provider_ids = {id(provider) for provider in providers or []}
        return [
            tool.definition
            for tool in self._tools
            if id(self._routing_table.get(tool.metadata.name)) in provider_ids
        ]

    def get_tools_for_protocol(self, protocol_cls: Any) -> List[ToolDefinition]:
        providers = self.get_providers_for_protocol(protocol_cls)
        return self._filter_tools_by_providers(providers)

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

    def get_provider_for_capability(self, capability: Any) -> Optional[ToolProvider]:
        providers = self.get_providers_for_capability(capability)
        return providers[0] if providers else None

    def get_toolset_generation(self) -> int:
        return self._toolset_generation

    def get_attached_target_bindings(self) -> List[Any]:
        return list(self._attached_target_bindings.values())

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

    @staticmethod
    async def resolve_redis_enterprise_admin_instance(
        redis_instance: RedisInstance,
    ) -> tuple[RedisInstance, str]:
        """Redis Enterprise admin credential resolver 的阶段三插槽。"""
        return redis_instance, "stage3_slot"

    @staticmethod
    def build_redis_enterprise_admin_instance_from_cluster(
        redis_cluster: "RedisCluster",
        *,
        target_id_override: Optional[str] = None,
    ) -> Optional[RedisInstance]:
        """Redis Enterprise cluster admin instance 构造的阶段三插槽。"""
        return None
