"""
不管你是查数据库的工具，还是查日志的工具，你都必须按照统一的规矩来注册、组装，并把最终的工具交给系统。
第一部分是各种protocol
第二部分就是ToolProvider 类
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from .models import SystemHost, Tool, ToolCapability, ToolDefinition, ToolMetadata

if TYPE_CHECKING:
    from redis_sre_agent.core.instances import RedisInstance

    from .manager import ToolManager

logger = logging.getLogger(__name__)


class MetricsProviderProtocol(Protocol):
    async def query(self, query: str) -> Dict[str, Any]: ...

    async def query_range(
        self, query: str, start_time: str, end_time: str, step: Optional[str] = None
    ) -> Dict[str, Any]: ...


class LogsProviderProtocol(Protocol):
    async def query_range(
        self,
        query: str,
        start: str,
        end: str,
        step: Optional[str] = None,
        limit: Optional[int] = None,
        interval: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Any]: ...


@runtime_checkable
class DiagnosticsProviderProtocol(Protocol):
    async def info(self, section: Optional[str] = None) -> Dict[str, Any]: ...

    async def replication_info(self) -> Dict[str, Any]: ...

    async def client_list(self, client_type: Optional[str] = None) -> Dict[str, Any]: ...

    async def system_hosts(self) -> List[SystemHost]: ...


@runtime_checkable
class KnowledgeProviderProtocol(Protocol):
    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]: ...


@runtime_checkable
class UtilitiesProviderProtocol(Protocol):
    async def calculator(self, expression: str) -> Dict[str, Any]: ...

# 如果要写一个真实去连 Redis 的工具，或者连外部 API 的工具
# 这些“工具提供者”都必须继承这个模具，在这个模具的基础上填空。
# 继承自 ABC，表明这是一个抽象基类，所有具体的工具提供者都必须继承它并实现对应接口
class ToolProvider(ABC):
    """所有工具 provider 的基类。"""

    # 类属性：声明这个 Provider 具备哪些能力（如诊断、查询指标等），默认是空集合
    capabilities: set[ToolCapability] = set()
    # 类属性：用于校验扩展配置的 Pydantic 模型（如果有特殊配置的话）
    instance_config_model: Optional[type[BaseModel]] = None
    # 类属性：在配置字典中寻找特定配置时的命名空间前缀
    extension_namespace: Optional[str] = None
    # 类属性：保存对 ToolManager 的引用，方便 Provider 内部调用管理器的功能
    _manager: Optional["ToolManager"] = None

    # 初始化方法
    def __init__(
            self, redis_instance: Optional["RedisInstance"] = None, config: Optional[Any] = None
    ):
        # 保存传入的 Redis 实例上下文
        self.redis_instance = redis_instance
        # 保存全局或通用的配置
        self.config = config
        # 初始化实例级别的特定配置对象
        self.instance_config: Optional[BaseModel] = None

        # 为了防止不同实例的同名工具冲突，这里生成一个简短的哈希值作为工具名的一部分
        if redis_instance is not None:
            import hashlib
            # 如果有实例，用实例 ID 的 SHA256 前6位作为哈希
            self._instance_hash = hashlib.sha256(redis_instance.id.encode()).hexdigest()[:6]
        else:
            # 如果没有实例（全局工具），用当前对象在内存中的 ID 的十六进制前6位作为哈希
            self._instance_hash = hex(id(self))[2:8]

        try:
            # 尝试从 redis_instance 身上加载专属于这个 Provider 的扩展配置
            self.instance_config = self._load_instance_extension_config()
        except Exception:
            # 如果加载失败，优雅降级，配置置为空
            self.instance_config = None

    # 根据特殊名字 extension_namespace 寻找其特定扩展配置
    def _load_instance_extension_config(self) -> Optional[BaseModel]:
        """解析实例上的 provider 扩展配置。
        这里只做结构解析，不打印 secret，也不主动连接外部系统。
        """
        # 如果当前类没有定义配置模型，或者没有绑定实例，直接跳过
        if not self.instance_config_model or not self.redis_instance:
            return None

        try:
            # 获取当前 Provider 的命名空间
            ns = self._get_extension_namespace()
            data: Dict[str, Any] = {}

            # 获取实例上的普通扩展数据
            ext = self.redis_instance.extension_data or {}

            # 尝试获取嵌套形式的配置 (如 ext={"my_provider": {"host": "..."}})
            if isinstance(ext.get(ns), dict):
                data.update(ext.get(ns) or {})
            else:
                # 尝试获取打平形式的配置 (如 ext={"my_provider.host": "..."})
                prefix = f"{ns}."
                for key, value in ext.items():
                    if isinstance(key, str) and key.startswith(prefix):
                        data[key[len(prefix):]] = value

            # 以同样的逻辑（嵌套或打平）合并敏感数据（secrets）
            secrets = self.redis_instance.extension_secrets or {}
            secrets_ns = secrets.get(ns)
            if isinstance(secrets_ns, dict):
                data.update(secrets_ns)
            else:
                prefix = f"{ns}."
                for key, value in secrets.items():
                    if isinstance(key, str) and key.startswith(prefix):
                        data[key[len(prefix):]] = value

            # 将收集到的字典数据，通过 Pydantic 模型进行校验和转换
            return self.instance_config_model.model_validate(data or {})
        except Exception:
            return None

    # 找某个工具的特殊名字 extension_namespace
    def _get_extension_namespace(self) -> str:
        try:
            # 优先使用 extension_namespace，其次使用 provider_name，并去除首尾空格
            ns = (self.extension_namespace or self.provider_name or "").strip()
            return ns or ""
        except Exception:
            # 发生异常时返回空字符串，保证鲁棒性
            return ""

    # 这是一个抽象属性，强制所有子类必须提供自己的 provider_name
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """provider 类型名，也是工具名前缀。"""
        ...

    # 支持异步上下文管理 (async with)，方便进入时初始化资源
    async def __aenter__(self) -> "ToolProvider":
        return self

    # 支持异步上下文管理，退出时清理资源（默认无操作）
    async def __aexit__(self, *args) -> None:
        return None

    # 核心方法：将 schemas 定义转化为可以直接被系统调用的 Tool 对象列表
    def tools(self) -> List[Tool]:
        # 1. 拿着子类重写的“说明书草稿”，获取当前工具箱里所有工具的结构定义（Schema）
        #    里面包含了工具叫什么名字、需要什么参数等描述文本
        schemas = self.create_tool_schemas()

        # 2. 初始化一个空列表，用来装待会打包好的标准化 Tool 对象
        tools: List[Tool] = []

        # 3. 开始遍历说明书里的每一个工具描述
        for schema in schemas:
            # 4. 根据说明书，组装出这个工具的“元数据（身份证）”
            #    记录它的名字、描述、属于什么分类（能力）、是哪个工具箱出的、是否需要连 Redis 实例
            meta = ToolMetadata(
                name=schema.name,
                description=schema.description,
                capability=schema.capability,
                provider_name=self.provider_name,
                requires_instance=self.requires_redis_instance,
            )

            # 5. 【核心翻译】大模型喊的名字是带防伪前缀的（如 redis_command_a1b2c3_get）
            #    这里调用解码器，把前缀剥掉，翻译出在当前类里真正的 Python 方法名叫什么（如 'get'）
            op_name = self.resolve_operation(schema.name, {}) or ""

            # 6. 【核心反射】根据翻译出来的字符串方法名（如 'get'），去自己身上（self）把这个真实的函数引用拿出来
            #    如果 op_name 有值，就拿函数；否则拿不到，置为 None
            method = getattr(self, op_name, None) if op_name else None

            # 7. 【安全质检】防呆检查：如果在类里根本没写这个方法，或者写了但不是一个可以运行的函数（不可调用）
            #    说明开发人员“光写了说明书，没写实现代码”，直接抛出严重错误，不让不合格产品出厂
            if not callable(method):
                raise RuntimeError(
                    f"Provider {self.__class__.__name__} has no method {op_name!r} "
                    f"for tool {schema.name!r}."
                )

            # 8. 【核心闭包】这是一个神奇的“外包小函数”（闭包）。
            #    为什么要它？因为大管家未来调用工具时，只会死板地传一个字典参数：`await tool.invoke(args)`。
            #    但类里面的真实方法（比如 `def get(self, key, db)`）是需要把字典拆开作为命名参数传进去的。
            #    所以这里做了一个外包层：大管家调 `_invoke(args)`，内部自动把 args 字典打散成 `**args` 喂给真实函数。
            async def _invoke(args: Dict[str, Any], _method=method) -> Any:
                # 把大管家传过来的参数字典（如果是 None 就变空字典），打散传给真实的方法并异步等待执行结果
                return await _method(**(args or {}))

            # 9. 【完美组装】把前面做好的“身份证（meta）”、“说明书（definition）”以及刚才包装好的“执行外包（invoke）”
            #    三合一，组装成一个标准化的 Tool 对象，并塞进刚才的流水线列表里
            tools.append(Tool(metadata=meta, definition=schema, invoke=_invoke))

        # 10. 全部的工具都打包完毕，整整齐齐地返回给大管家去登记名册
        return tools

    # 提供者必须提供工具的描述结构，默认抛出异常，强制子类重写此方法或直接重写 tools()
    def create_tool_schemas(self) -> List[ToolDefinition]:
        raise NotImplementedError(
            f"{self.__class__.__name__}.create_tool_schemas() is not implemented; "
            "override create_tool_schemas() or tools()."
        )

    # 组装函数：将提供者的名字、实例哈希和操作名拼接，生成全局唯一的工具注册名
    def _make_tool_name(self, operation: str) -> str:
        return f"{self.provider_name}_{self._instance_hash}_{operation}"

    # 检查当前 Provider 是否绑定了 Redis 实例上下文
    @property
    def requires_redis_instance(self) -> bool:
        return self.redis_instance is not None

    # 为正在执行的工具获取人类可读的状态更新信息
    def get_status_update(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        try:
            # 找到对应的操作名
            op = self.resolve_operation(tool_name, args)
            if not op:
                return None

            # 找到对应的方法
            method = self.__dict__.get(op) or type(self).__dict__.get(op)
            # 尝试获取方法上可能绑定的状态模版 (通常通过装饰器注入)
            template = getattr(method, "_status_update_template", None) if method else None

            if not template:
                return None
            try:
                # 用工具执行的参数填充模版（例如 "正在查询 {key} 的值..."）
                return template.format(**args)
            except Exception:
                # 填充失败则直接返回原始模版字符串
                return template
        except Exception:
            return None

    # 反向解析：从包含哈希和前缀的完整工具名中，提取出真正的“方法名” (operation)
    def resolve_operation(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        try:
            # 拼装出前缀，例如 "redis_1a2b3c_"
            prefix = f"{self.provider_name}_{self._instance_hash}_"
            # 如果是标准前缀，直接去掉前缀返回后面的真实操作名
            if tool_name.startswith(prefix):
                return tool_name[len(prefix):]

            # 如果前缀没匹配上，尝试按下划线拆分，并取第三部分及以后的内容作为兜底
            parts = tool_name.split("_")
            if len(parts) >= 3:
                return "_".join(parts[2:])

            # 如果都失败了，记录警告，并原样返回完整名字
            logger.warning(
                "resolve_operation falling back to full tool name %r as operation.",
                tool_name,
            )
            return tool_name
        except Exception as exc:
            logger.warning("resolve_operation failed for %r: %s", tool_name, exc)
            return None
