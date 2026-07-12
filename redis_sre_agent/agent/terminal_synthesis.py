"""终端回答合成 helper。

original 会在 Agent 已经收集到中间状态但没有自然结束时，用 LLM 读取消息和
evidence 生成最终回答。当前裁剪版没有真实 LLM 依赖，所以保留同名合成插槽，
并提供一个只用于 fake/test fallback 的确定性合成函数。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .helpers import coerce_response_text

GuardedInvoke = Callable[..., Awaitable[Any]]
FailureResponseFactory = Callable[[], str]
ExceptionFormatter = Callable[[Exception], str]

_REDACTED = "[REDACTED]"


@dataclass(frozen=True)
class TerminalSynthesisConfig:
    """终端合成的提示和截断配置。"""

    request_kind: str
    system_prompt: str
    messages_heading: str = "Captured messages"
    evidence_heading: str = "Captured evidence"
    no_messages_text: str = "No messages captured."
    no_evidence_text: str = "No tool evidence captured."
    failure_log_message: str = "Terminal synthesis failed: %s"
    empty_log_message: str = "Terminal synthesis returned empty text."
    context_limit: int = 16000
    item_limit: int = 2000
    message_item_limit: int = 2000
    message_tail_limit: int = 12
    evidence_tail_limit: int = 8
    include_system_messages: bool = True
    detailed_message_headers: bool = False
    empty_message_text: str | None = None
    message_omitted_unit: str = "conversation message(s)"
    evidence_omitted_unit: str = "tool result envelope(s)"


def _safe_text(value: Any, max_chars: int = 500) -> str:
    text = str(value if value is not None else "")
    text = re.sub(
        r"(?i)\b(rediss?|unix)://[^\s'\"<>@]+@[^\s'\"<>]+",
        lambda match: re.sub(r"://([^:@/]+):[^@/]+@", r"://\1:[REDACTED]@", match.group(0)),
        text,
    )
    text = re.sub(
        r"(?i)\b(password|secret|token|requirepass|masterauth|pass)(\s*[=:]\s*)([^\s,;}]+)",
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        text,
    )
    if len(text) > max_chars:
        return f"{text[:max_chars].rstrip()}..."
    return text


def _json_preview(value: Any, max_chars: int = 360) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    return _safe_text(text, max_chars=max_chars)


def describe_captured_state(
    *,
    messages: Sequence[Any],
    tool_envelopes: Sequence[dict[str, Any]],
) -> str:
    gathered: list[str] = []
    if messages:
        gathered.append(f"{len(messages)} conversation message(s)")
    if tool_envelopes:
        gathered.append(f"{len(tool_envelopes)} tool result envelope(s)")
    return ", ".join(gathered) if gathered else "no usable intermediate state"


def truncate_terminal_synthesis_text(value: Any, max_chars: int) -> str:
    text = coerce_response_text(value) or str(value)
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars].rstrip()}\n... [truncated {omitted} chars]"


def format_terminal_synthesis_messages(
    messages: Sequence[Any],
    config: TerminalSynthesisConfig,
) -> str:
    visible = list(messages or [])
    if not config.include_system_messages:
        visible = [msg for msg in visible if getattr(msg, "type", "") != "system"]
    if not visible:
        return config.no_messages_text
    selected = visible[-config.message_tail_limit :]
    lines = []
    omitted = len(visible) - len(selected)
    if omitted > 0:
        lines.append(f"... [{omitted} earlier {config.message_omitted_unit} omitted]")
    for message in selected:
        role = getattr(message, "type", message.__class__.__name__)
        content = coerce_response_text(getattr(message, "content", ""))
        if not content and isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            content = json.dumps(message.tool_calls, ensure_ascii=False, default=str)
        if not content and config.empty_message_text is not None:
            content = config.empty_message_text
        if not config.detailed_message_headers:
            lines.append(
                f"{role}: {truncate_terminal_synthesis_text(content, config.message_item_limit)}"
            )
            continue
        header = str(role)
        tool_calls = getattr(message, "tool_calls", None) or []
        tool_names = [
            str(call.get("name"))
            for call in tool_calls
            if isinstance(call, dict) and call.get("name")
        ]
        if tool_names:
            header = f"{header} requested tools: {', '.join(tool_names)}"
        if isinstance(message, ToolMessage) and getattr(message, "name", None):
            header = f"{header} ({message.name})"
        lines.append(
            f"{header}:\n{truncate_terminal_synthesis_text(content, config.message_item_limit)}"
        )
    return truncate_terminal_synthesis_text("\n\n".join(lines), config.context_limit)


def format_terminal_synthesis_tool_evidence(
        tool_envelopes: Sequence[dict[str, Any]],  # 输入参数：工具执行结果的“信封”列表（包含状态和数据）
        config: TerminalSynthesisConfig,  # 输入参数：终端合成的配置对象（包含截断限制和文本模板）
) -> str:
    """将所有工具执行的结果（证据）格式化为适合喂给 LLM 的排版文本。"""

    # 边界校验：如果没有收集到任何工具执行结果，直接返回配置中预设的“无证据”提示文本
    if not tool_envelopes:
        return config.no_evidence_text

    # 根据配置中的截断限制（evidence_tail_limit），只切片截取最后几条工具结果（防止上下文过长爆 Token）
    selected = list(tool_envelopes)[-config.evidence_tail_limit:]
    # 初始化用于存放格式化文本行的列表
    lines = []

    # 计算有多少条较早的工具执行结果被忽略了
    omitted = len(tool_envelopes) - len(selected)
    # 如果有被忽略的结果，在开头添加一行提示，表明前面有多少条记录被省略了
    if omitted > 0:
        lines.append(f"... [{omitted} earlier {config.evidence_omitted_unit} omitted]")

    # 遍历筛选出来的每一个工具结果信封
    for envelope in selected:
        # 获取工具的标识名，优先取 tool_key，其次取 name，都没有则降级显示 "unknown_tool"
        tool_key = envelope.get("tool_key") or envelope.get("name") or "unknown_tool"
        # 尝试直接获取已经提炼好的摘要信息（summary）
        payload = envelope.get("summary")
        # 如果没有现成的摘要，则将原始的 data 字典序列化为 JSON 字符串（保留中文，未知对象强转 string）
        if not payload:
            payload = json.dumps(envelope.get("data", {}), ensure_ascii=False, default=str)
        # 将工具名与截断后的数据拼接成固定格式，追加到列表中（item_limit 控制单个工具结果的最大长度）
        lines.append(f"{tool_key}:\n{truncate_terminal_synthesis_text(payload, config.item_limit)}")

    # 将所有格式化好的工具文本用双换行拼接，并在最外层做最后一次整体长度截断（由 context_limit 控制总长度）
    return truncate_terminal_synthesis_text("\n\n".join(lines), config.context_limit)

# 把之前收集到的所有人话消息和工具回执打包，喂给一个特定的“总结大模型”，让它出一份结合了所有证据的专业最终回答。
# 同时保持一个预设的失败方案
async def synthesize_terminal_response(
        llm: Any,  # 输入参数：负责最终总结的 LLM 对象
        *,
        config: TerminalSynthesisConfig,  # 命名空间参数：终端合成配置项
        messages: Sequence[Any],  # 命名空间参数：历史聊天消息列表
        tool_envelopes: Sequence[dict[str, Any]],  # 命名空间参数：工具执行结果信封列表
        guarded_invoke: GuardedInvoke,  # 命名空间参数：带有熔断或安全防御的 LLM 调用器
        failure_response_factory: FailureResponseFactory,  # 命名空间参数：当合成失败时的降级兜底报告生成工厂
        logger: logging.Logger,  # 命名空间参数：日志记录器
        human_prelude: str | None = None,  # 可选参数：人类提示词的前导前缀（前言）
        format_exception: ExceptionFormatter = str,  # 可选参数：异常格式化工具，默认直接转字符串
) -> str:
    """保留 original 终端 LLM 合成入口，用于最终报告的异步组装与生成。"""

    # 初始化用于存放发送给大模型（HumanMessage）的不同文本区块
    human_sections = []

    # 如果传入了前导前缀（例如特定的诊断引导语），优先放入区块列表的第一位
    if human_prelude:
        human_sections.append(human_prelude)

    # 依次追加两段核心内容：1. 格式化后的对话历史；2. 格式化后的工具执行证据
    human_sections.extend(
        [
            # 拼接历史消息标题及格式化后的消息内容
            f"{config.messages_heading}:\n{format_terminal_synthesis_messages(messages, config)}",
            # 拼接工具证据标题及上面函数格式化出来的工具执行结果
            (
                f"{config.evidence_heading}:\n"
                f"{format_terminal_synthesis_tool_evidence(tool_envelopes, config)}"
            ),
        ]
    )

    try:
        # 使用防爆/带防御机制的 guarded_invoke 异步请求大模型
        synthesized = await guarded_invoke(
            llm,
            [
                # 注入系统全局提示词（SystemMessage），通常规定了报告的格式、语气和诊断规范
                SystemMessage(content=config.system_prompt),
                # 将前面拼接好的所有历史上下文作为 HumanMessage 发送给模型
                HumanMessage(content="\n\n".join(human_sections)),
            ],
            # 传入请求类型标识，便于底层的防护网进行针对性的速率限制或审计
            request_kind=config.request_kind,
        )
    except Exception as exc:
        # 如果调用大模型期间发生任何异常（如超时、限流、敏感词拦截等），记录警告日志并输出堆栈信息
        logger.warning(config.failure_log_message, format_exception(exc), exc_info=True)
        # 触发降级机制，返回工厂函数生成的兜底安全报告，确保系统不崩溃
        return failure_response_factory()

    # 将大模型返回的 response 对象安全转换为纯文本格式
    response_text = coerce_response_text(getattr(synthesized, "content", ""))

    # 如果成功拿到了非空的诊断报告文本，直接将其返回给终端用户
    if response_text:
        return response_text

    # 如果模型返回的内容为空（极端情况），记录空结果警告日志
    logger.warning(config.empty_log_message)
    # 同样触发降级机制，返回兜底报告
    return failure_response_factory()


def summarize_evidence_for_report(tool_envelopes: list[dict[str, Any]]) -> list[str]:
    """把 Redis evidence（工具执行证据/结果）压缩并翻译成报告中稳定、格式统一的摘要行。"""

    # 初始化用于存放每一行摘要的列表
    lines: list[str] = []

    # 循环遍历所有工具执行完返回的“信封”数据结构
    for envelope in tool_envelopes or []:
        # 获取工具名称，兼容 name 字段或 tool_key 字段
        name = str(envelope.get("name") or envelope.get("tool_key") or "")
        # 获取工具返回的数据，若不是字典格式则降级为空字典
        data = envelope.get("data") if isinstance(envelope.get("data"), dict) else {}
        # 获取工具执行的状态，并强转为小写（如 "success" 或 "error"）
        status = str(envelope.get("status") or "").lower()

        # 边界处理：如果工具执行失败，直接生成一条错误摘要行，并截取前180个字符防止撑爆报告
        if status == "error":
            lines.append(f"- {name}: error={_json_preview(data, 180)}")
            continue

        # 分支1：处理 Redis `INFO` 命令的结果
        if name == "info" and isinstance(data, dict) and data.get("status") == "success":
            # 提取具体的 info section（如 memory, stats 等），默认是 all
            section = data.get("section") or "all"
            # 提取核心的 payload 数据
            payload = data.get("data") if isinstance(data.get("data"), dict) else {}
            # 初始化当前行的字段片段列表，首项标识这是哪个 section
            bits = [f"INFO section={section}"]
            # 遍历定义好的 Redis 核心监控指标 Key 和报告中显示的标签 Label
            for key, label in (
                    ("redis_version", "redis_version"),
                    ("used_memory_human", "used_memory"),
                    ("used_memory", "used_memory"),
                    ("maxmemory", "maxmemory"),
                    ("connected_clients", "connected_clients"),
                    ("instantaneous_ops_per_sec", "ops_per_sec"),
                    ("evicted_keys", "evicted_keys"),
            ):
                # 如果返回的数据里包含这个核心指标，将其安全的转为文本并追加到片段列表里
                if key in payload:
                    bits.append(f"{label}={_safe_text(payload.get(key))}")
            # 用逗号将所有解析出的指标片段连接起来，拼成一条完整的 markdown 列表行
            lines.append("- " + ", ".join(bits))

        # 分支2：处理 `CLIENT LIST`（客户端连接列表）结果，摘要打印总连接数
        elif name == "client_list" and isinstance(data, dict) and data.get("status") == "success":
            lines.append(f"- CLIENT LIST count={data.get('count', 0)}")

        # 分支3：处理 `SLOWLOG`（慢日志）结果，摘要打印收集到的慢日志条数
        elif name == "slowlog" and isinstance(data, dict) and data.get("status") == "success":
            lines.append(f"- SLOWLOG entries={data.get('count', 0)}")

        # 分支4：处理 `MEMORY STATS`（内存细分统计）结果
        elif name == "memory_stats" and isinstance(data, dict) and data.get("status") == "success":
            stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
            # 提取其中最核心的指标：历史峰值分配内存
            peak = stats.get("peak.allocated")
            # 如果峰值存在则拼接显示，不存在则留空
            suffix = f", peak.allocated={_safe_text(peak)}" if peak is not None else ""
            lines.append(f"- MEMORY STATS collected{suffix}")

        # 分支5：处理 `CONFIG GET`（获取 Redis 配置）结果，打印匹配模式和获取到的配置项数量
        elif name == "config_get" and isinstance(data, dict) and data.get("status") == "success":
            lines.append(
                f"- CONFIG GET pattern={_safe_text(data.get('pattern'))}, count={data.get('count', 0)}"
            )

        # 分支6：处理 `replication_info`（主从复制信息）结果，提取角色类型（master/slave）
        elif name == "replication_info" and isinstance(data, dict) and data.get("status") == "success":
            role = data.get("role")
            role_type = role.get("type") if isinstance(role, dict) else None
            lines.append(f"- Replication role={_safe_text(role_type or 'unknown')}")

        # 分支7：处理 `cluster_info`（集群信息）结果，打印集群健康状态
        elif name == "cluster_info" and isinstance(data, dict):
            lines.append(f"- CLUSTER INFO status={_safe_text(data.get('status', 'unknown'))}")

        # 分支8：处理 `search_indexes`（RediSearch 索引列表）结果，打印索引总数
        elif name == "search_indexes" and isinstance(data, dict):
            lines.append(f"- RediSearch indexes={data.get('count', 0)}")

        # 分支9：处理 `search_index_info`（具体某个搜索索引的详情）结果，打印索引名称
        elif name == "search_index_info" and isinstance(data, dict):
            lines.append(f"- RediSearch index info={_safe_text(data.get('index_name'))}")

        # 分支10：处理 `resolve_redis_targets`（Redis 目标实例识别/发现）结果，打印关联的目标句柄数
        elif name == "resolve_redis_targets" and isinstance(data, dict):
            lines.append(
                f"- Target resolution status={_safe_text(data.get('status', 'unknown'))}, "
                f"attached={len(data.get('attached_target_handles') or [])}"
            )

    # 返回拼装好的摘要列表；如果整个列表为空，返回一条默认兜底提示信息
    return lines or ["- 没有收集到可用 Redis evidence。"]

# 无llm情况下纯硬编码出一份报告
def build_deterministic_diagnostic_response(
        query: str,
        tool_envelopes: list[dict[str, Any]],
        *,
        agent_kind: str = "chat",
) -> str:
    """fake/test fallback 核心输出：在不依赖真实 LLM 的情况下，将 evidence 汇总合成确定性的 Markdown 报告。"""

    # 统计成功执行的工具数量
    successes = sum(1 for envelope in tool_envelopes if envelope.get("status") == "success")
    # 统计执行失败的工具数量
    errors = sum(1 for envelope in tool_envelopes if envelope.get("status") == "error")

    # 根据当前 Agent 的类型（如 triage 分流），决定报告的一级大标题
    title = "Redis 深度诊断报告" if agent_kind == "triage" else "Redis 诊断摘要"

    # 初始化“## 观察”模块的内容列表
    observations = [
        f"- 本次通过 `{agent_kind}` Agent workflow 收集到 {successes} 条成功 evidence，{errors} 条错误 evidence。",
        "- 结论只基于本次 ToolManager 返回的结构化 evidence；没有把凭据、连接串或 token 写入报告。",
    ]
    # 如果存在报错的工具，在“观察”部分追加一条安全权限排查建议
    if errors:
        observations.append(
            "- 有部分工具返回错误，优先确认 Redis 版本、模块、权限或部署模式是否支持对应只读命令。"
        )

    # 写死一套标准的“## 下一步建议”模块内容列表
    recommendations = [
        "- 如果 memory 或 client 指标异常，继续核对 `INFO memory`、`MEMORY STATS` 与 `CLIENT LIST` 的时间点一致性。",
        "- 如果 slowlog 非空，按持续时间和命令模式定位热点访问，再结合业务流量窗口复核。",
        "- 对配置变更保持只读审计；当前 Agent 主链只收集和解释 evidence，不执行修复命令。",
    ]

    # 将标题、用户原始查询（限长240字）、摘要、观察、建议等模块用换行符拼装成一篇标准的 Markdown 文档
    return "\n".join(
        [
            f"# {title}",
            "",
            f"查询：{_safe_text(query, max_chars=240)}",
            "",
            "## Evidence 摘要",
            *summarize_evidence_for_report(tool_envelopes),  # 解包展开工具摘要列表
            "",
            "## 观察",
            *observations,  # 解包展开观察列表
            "",
            "## 下一步建议",
            *recommendations,  # 解包展开建议列表
        ]
    )
