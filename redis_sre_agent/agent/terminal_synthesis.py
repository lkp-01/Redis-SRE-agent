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

from langchain_core.messages import HumanMessage, SystemMessage

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
        lines.append(f"... [{omitted} earlier conversation message(s) omitted]")
    for message in selected:
        role = getattr(message, "type", message.__class__.__name__)
        content = coerce_response_text(getattr(message, "content", ""))
        lines.append(
            f"{role}: {truncate_terminal_synthesis_text(content, config.message_item_limit)}"
        )
    return truncate_terminal_synthesis_text("\n\n".join(lines), config.context_limit)


def format_terminal_synthesis_tool_evidence(
    tool_envelopes: Sequence[dict[str, Any]],
    config: TerminalSynthesisConfig,
) -> str:
    if not tool_envelopes:
        return config.no_evidence_text
    selected = list(tool_envelopes)[-config.evidence_tail_limit :]
    lines = []
    omitted = len(tool_envelopes) - len(selected)
    if omitted > 0:
        lines.append(f"... [{omitted} earlier tool result envelope(s) omitted]")
    for envelope in selected:
        tool_key = envelope.get("tool_key") or envelope.get("name") or "unknown_tool"
        payload = envelope.get("summary")
        if not payload:
            payload = json.dumps(envelope.get("data", {}), ensure_ascii=False, default=str)
        lines.append(f"{tool_key}:\n{truncate_terminal_synthesis_text(payload, config.item_limit)}")
    return truncate_terminal_synthesis_text("\n\n".join(lines), config.context_limit)


async def synthesize_terminal_response(
    llm: Any,
    *,
    config: TerminalSynthesisConfig,
    messages: Sequence[Any],
    tool_envelopes: Sequence[dict[str, Any]],
    guarded_invoke: GuardedInvoke,
    failure_response_factory: FailureResponseFactory,
    logger: logging.Logger,
    human_prelude: str | None = None,
    format_exception: ExceptionFormatter = str,
) -> str:
    """保留 original 终端 LLM 合成入口。"""

    human_sections = []
    if human_prelude:
        human_sections.append(human_prelude)
    human_sections.extend(
        [
            f"{config.messages_heading}:\n{format_terminal_synthesis_messages(messages, config)}",
            (
                f"{config.evidence_heading}:\n"
                f"{format_terminal_synthesis_tool_evidence(tool_envelopes, config)}"
            ),
        ]
    )
    try:
        synthesized = await guarded_invoke(
            llm,
            [
                SystemMessage(content=config.system_prompt),
                HumanMessage(content="\n\n".join(human_sections)),
            ],
            request_kind=config.request_kind,
        )
    except Exception as exc:
        logger.warning(config.failure_log_message, format_exception(exc), exc_info=True)
        return failure_response_factory()

    response_text = coerce_response_text(getattr(synthesized, "content", ""))
    if response_text:
        return response_text
    logger.warning(config.empty_log_message)
    return failure_response_factory()


def summarize_evidence_for_report(tool_envelopes: list[dict[str, Any]]) -> list[str]:
    """把 Redis evidence 压成报告里的稳定摘要行。"""

    lines: list[str] = []
    for envelope in tool_envelopes or []:
        name = str(envelope.get("name") or envelope.get("tool_key") or "")
        data = envelope.get("data") if isinstance(envelope.get("data"), dict) else {}
        status = str(envelope.get("status") or "").lower()
        if status == "error":
            lines.append(f"- {name}: error={_json_preview(data, 180)}")
            continue

        if name == "info" and isinstance(data, dict) and data.get("status") == "success":
            section = data.get("section") or "all"
            payload = data.get("data") if isinstance(data.get("data"), dict) else {}
            bits = [f"INFO section={section}"]
            for key, label in (
                ("redis_version", "redis_version"),
                ("used_memory_human", "used_memory"),
                ("used_memory", "used_memory"),
                ("maxmemory", "maxmemory"),
                ("connected_clients", "connected_clients"),
                ("instantaneous_ops_per_sec", "ops_per_sec"),
                ("evicted_keys", "evicted_keys"),
            ):
                if key in payload:
                    bits.append(f"{label}={_safe_text(payload.get(key))}")
            lines.append("- " + ", ".join(bits))
        elif name == "client_list" and isinstance(data, dict) and data.get("status") == "success":
            lines.append(f"- CLIENT LIST count={data.get('count', 0)}")
        elif name == "slowlog" and isinstance(data, dict) and data.get("status") == "success":
            lines.append(f"- SLOWLOG entries={data.get('count', 0)}")
        elif name == "memory_stats" and isinstance(data, dict) and data.get("status") == "success":
            stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
            peak = stats.get("peak.allocated")
            suffix = f", peak.allocated={_safe_text(peak)}" if peak is not None else ""
            lines.append(f"- MEMORY STATS collected{suffix}")
        elif name == "config_get" and isinstance(data, dict) and data.get("status") == "success":
            lines.append(
                f"- CONFIG GET pattern={_safe_text(data.get('pattern'))}, count={data.get('count', 0)}"
            )
        elif name == "replication_info" and isinstance(data, dict) and data.get("status") == "success":
            role = data.get("role")
            role_type = role.get("type") if isinstance(role, dict) else None
            lines.append(f"- Replication role={_safe_text(role_type or 'unknown')}")
        elif name == "cluster_info" and isinstance(data, dict):
            lines.append(f"- CLUSTER INFO status={_safe_text(data.get('status', 'unknown'))}")
        elif name == "search_indexes" and isinstance(data, dict):
            lines.append(f"- RediSearch indexes={data.get('count', 0)}")
        elif name == "search_index_info" and isinstance(data, dict):
            lines.append(f"- RediSearch index info={_safe_text(data.get('index_name'))}")
        elif name == "resolve_redis_targets" and isinstance(data, dict):
            lines.append(
                f"- Target resolution status={_safe_text(data.get('status', 'unknown'))}, "
                f"attached={len(data.get('attached_target_handles') or [])}"
            )

    return lines or ["- 没有收集到可用 Redis evidence。"]


def build_deterministic_diagnostic_response(
    query: str,
    tool_envelopes: list[dict[str, Any]],
    *,
    agent_kind: str = "chat",
) -> str:
    """fake/test fallback：不依赖真实 LLM 的 evidence 汇总。"""

    successes = sum(1 for envelope in tool_envelopes if envelope.get("status") == "success")
    errors = sum(1 for envelope in tool_envelopes if envelope.get("status") == "error")
    title = "Redis 深度诊断报告" if agent_kind == "triage" else "Redis 诊断摘要"
    observations = [
        f"- 本次通过 `{agent_kind}` Agent workflow 收集到 {successes} 条成功 evidence，{errors} 条错误 evidence。",
        "- 结论只基于本次 ToolManager 返回的结构化 evidence；没有把凭据、连接串或 token 写入报告。",
    ]
    if errors:
        observations.append(
            "- 有部分工具返回错误，优先确认 Redis 版本、模块、权限或部署模式是否支持对应只读命令。"
        )
    recommendations = [
        "- 如果 memory 或 client 指标异常，继续核对 `INFO memory`、`MEMORY STATS` 与 `CLIENT LIST` 的时间点一致性。",
        "- 如果 slowlog 非空，按持续时间和命令模式定位热点访问，再结合业务流量窗口复核。",
        "- 对配置变更保持只读审计；当前 Agent 主链只收集和解释 evidence，不执行修复命令。",
    ]
    return "\n".join(
        [
            f"# {title}",
            "",
            f"查询：{_safe_text(query, max_chars=240)}",
            "",
            "## Evidence 摘要",
            *summarize_evidence_for_report(tool_envelopes),
            "",
            "## 观察",
            *observations,
            "",
            "## 下一步建议",
            *recommendations,
        ]
    )
