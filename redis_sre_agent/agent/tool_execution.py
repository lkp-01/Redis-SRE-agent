"""Agent 工具执行封装。

original 项目里 tool node 不直接拼报告，而是把 LLM 请求的 tool_calls 交给
ToolManager，再把结果作为 ToolMessage 放回图状态。这里保留同样的生命周期，
审批、人审恢复等更复杂 gate 仍是后续插槽。
"""

# 从未来版本导入注解特性，以便在低版本 Python 中使用延迟类型注解（例如直接用 List[Dict]）
from __future__ import annotations

# 导入必要的内置库和类型提示模块
import json
import re
from typing import Any, Dict, List, Optional

# 导入 LangChain 核心库中的工具消息类，用于将工具执行结果返回给 LLM
from langchain_core.messages import ToolMessage

# 定义敏感信息脱敏后的替代文本
_REDACTED = "[REDACTED]"


def _safe_error_message(exc: Exception) -> str:
    """对异常信息进行脱敏处理，防止凭据（如密码、Token等）泄露到日志或前端。"""
    # 将异常对象转换为字符串形式
    message = str(exc)

    # 匹配并脱敏 Redis 或 Unix 套接字等连接字符串中的密码信息（格式如 redis://user:password@host）
    message = re.sub(
        r"(?i)\b(rediss?|unix)://[^\s'\"<>@]+@[^\s'\"<>]+",
        lambda match: re.sub(r"://([^:@/]+):[^@/]+@", r"://\1:[REDACTED]@", match.group(0)),
        message,
    )

    # 匹配并脱敏形如 password=xxx, secret: xxx, token = xxx 等常见的敏感配置项
    message = re.sub(
        r"(?i)\b(password|secret|token|requirepass|masterauth|pass)(\s*[=:]\s*)([^\s,;}]+)",
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        message,
    )
    # 返回脱敏后的安全错误信息
    return message


async def execute_tool_call_with_gate(
    *,
    tool_manager: Any,
    tool_name: str,
    tool_args: Dict[str, Any],
    local_tools: Optional[Dict[str, Any]] = None,
) -> Any:
    """执行单个工具调用，保留后续审批 gate（如人工审批锁）的接缝。"""

    # 优先检查是否存在本地覆盖或 mock 的工具方法
    local_tool = (local_tools or {}).get(tool_name)
    if local_tool is not None:
        # 如果存在本地工具，直接传入参数进行调用
        result = local_tool(**dict(tool_args or {}))
        # 如果该本地工具是一个异步函数（协程），则使用 await 等待其执行完毕
        if hasattr(result, "__await__"):
            return await result
        # 如果是同步函数，直接返回结果
        return result

    # 如果没有本地覆盖，则交由全局的 tool_manager 去解析并执行该工具
    return await tool_manager.resolve_tool_call(tool_name, dict(tool_args or {}))


def _tool_call_id(tool_call: Dict[str, Any]) -> str:
    """兼容不同 LLM 厂商或版本的字段定义，提取工具调用的唯一 ID。"""
    return str(
        tool_call.get("id")
        or tool_call.get("tool_call_id")
        or tool_call.get("call_id")
        or ""
    )


def _tool_call_name(tool_call: Dict[str, Any]) -> str:
    """解析并提取工具的名称（兼容标准的 name 字段以及 OpenAI 的 function.name 嵌套格式）。"""
    name = tool_call.get("name")
    # 如果外层没有 name，且存在符合 OpenAI 格式的 function 字典，则从内部提取
    if not name and isinstance(tool_call.get("function"), dict):
        name = tool_call["function"].get("name")
    return str(name or "")


def _tool_call_args(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """解析并提取工具的入参（兼容字典格式以及未反序列化的 JSON 字符串）。"""
    args = tool_call.get("args")
    # 如果外层没有 args，且存在 function 字典，则尝试从 function 内部提取 arguments
    if args is None and isinstance(tool_call.get("function"), dict):
        raw_arguments = tool_call["function"].get("arguments")
        # 如果入参是未解析的 JSON 字符串，尝试反序列化为字典
        if isinstance(raw_arguments, str):
            try:
                args = json.loads(raw_arguments or "{}")
            except Exception:
                args = {}  # 解析失败则降级为空字典
        # 如果已经是字典格式，直接赋值
        elif isinstance(raw_arguments, dict):
            args = raw_arguments

    # 确保最终返回的一定是一个合法的字典类型
    return dict(args or {}) if isinstance(args, dict) else {}


def _message_content(result: Any) -> str:
    """将工具的执行结果标准化转化为可存储/可传输的 JSON 字符串。"""
    try:
        # 尝试序列化结果，禁用 ASCII 转义以保留中文，并对 key 进行排序保证一致性
        return json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        # 如果序列化失败（例如包含不可序列化的对象），则采用降级方案，将其强转为字符串返回
        return json.dumps({"status": "success", "raw": str(result)}, ensure_ascii=False)


async def execute_tool_calls_with_gate(
    *,
    tool_manager: Any,
    tool_calls: List[Dict[str, Any]],
    local_tools: Optional[Dict[str, Any]] = None,
) -> List[ToolMessage]:
    """批量执行工具调用，并返回可追加到 LangGraph 状态的 ToolMessage 列表。"""

    # 初始化用于存放返回消息的列表
    messages: List[ToolMessage] = []

    # 遍历 LLM 生成的所有工具调用请求
    for index, tool_call in enumerate(tool_calls or []):
        # 提取当前工具的名称、参数和唯一 ID
        name = _tool_call_name(tool_call)
        args = _tool_call_args(tool_call)
        # 如果 LLM 没有返回 ID，则根据当前索引自动生成一个（例如 tool_call_1）防止 LangGraph 报错
        call_id = _tool_call_id(tool_call) or f"tool_call_{index + 1}"

        # 边界校验：如果没有获取到有效的工具名称，直接返回失败消息
        if not name:
            payload = {"status": "failed", "error": "missing tool name"}
            messages.append(ToolMessage(_message_content(payload), tool_call_id=call_id))
            continue

        try:
            # 调用单个工具执行函数（包含本地覆盖逻辑）并等待结果
            result = await execute_tool_call_with_gate(
                tool_manager=tool_manager,
                tool_name=name,
                tool_args=args,
                local_tools=local_tools,
            )
            # 执行成功，将结果封装为成功的 ToolMessage 放入列表
            messages.append(
                ToolMessage(_message_content(result), tool_call_id=call_id, name=name)
            )
        except Exception as exc:
            # 执行期间发生任何异常，捕获并进行脱敏处理
            payload = {"status": "failed", "error": _safe_error_message(exc)}
            # 将错误信息封装为失败的 ToolMessage 放入列表，确保图流程不中断
            messages.append(ToolMessage(_message_content(payload), tool_call_id=call_id, name=name))

    # 返回构建好的所有 ToolMessage 消息列表
    return messages