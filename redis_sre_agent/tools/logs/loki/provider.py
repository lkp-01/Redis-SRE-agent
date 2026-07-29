"""通过 Loki HTTP API 收集只读日志证据。"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.tools.decorators import status_update
from redis_sre_agent.tools.models import ToolCapability, ToolDefinition
from redis_sre_agent.tools.protocols import ToolProvider

logger = logging.getLogger(__name__)

_REDACTED = "[REDACTED]"
_HTTP_URL_PATTERN = re.compile(r"(?i)\bhttps?://[^\s'\"<>]+")
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)\b(password|secret|token|api[_-]?key|authorization)(\s*[=:]\s*)([^\s,;]+)"
)


def _safe_url(value: str) -> str:
    """保留服务位置，但移除 URL 中的用户信息、查询参数和片段。"""
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except Exception:
        return "<redacted-loki-url>"


def _safe_error_message(value: Any) -> str:
    """清洗外部异常文本，避免连接信息和凭据进入工具结果或日志。"""
    message = str(value)
    message = _HTTP_URL_PATTERN.sub(lambda match: _safe_url(match.group(0)), message)
    return _SENSITIVE_VALUE_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        message,
    )


def _safe_error_payload(value: Any) -> Any:
    """递归清洗 Loki 返回的错误对象，同时保留可诊断结构。"""
    if isinstance(value, dict):
        return {str(key): _safe_error_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_error_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_error_payload(item) for item in value)
    return _safe_error_message(value)


class LokiConfig(BaseSettings):
    """从 TOOLS_LOKI_ 环境变量读取 Loki 连接配置。"""

    model_config = SettingsConfigDict(
        env_prefix="tools_loki_",
        env_ignore_empty=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    url: str = Field(
        default="http://localhost:3100",
        repr=False,
        description="Loki HTTP API 地址。",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        repr=False,
        description="可选的 X-Scope-OrgID 租户标识。",
    )
    timeout: float = Field(default=30.0, gt=0, description="HTTP 请求超时秒数。")
    default_selector: Optional[str] = Field(
        default=None,
        description="空 selector 使用的默认日志流选择器。",
    )


class LokiInstanceConfig(BaseModel):
    """单个 Redis target 可选的 Loki 日志流提示。"""

    model_config = ConfigDict(hide_input_in_errors=True)

    prefer_streams: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="空 selector 优先使用的日志标签集合。",
    )
    keywords: Optional[List[str]] = Field(
        default=None,
        description="为未来日志关键词推荐保留的扩展槽。",
    )
    default_selector: Optional[str] = Field(
        default=None,
        description="该 target 专用的默认日志流选择器。",
    )


class LokiToolProvider(ToolProvider):
    """提供查询、标签、日志流、容量和模式分析七个只读工具。"""

    capabilities = {ToolCapability.LOGS}
    instance_config_model = LokiInstanceConfig
    extension_namespace = "loki"

    def __init__(
        self,
        redis_instance: Optional[RedisInstance] = None,
        config: Optional[LokiConfig] = None,
    ):
        super().__init__(redis_instance)
        self.config = config or LokiConfig()

    @property
    def provider_name(self) -> str:
        return "loki"

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.config.tenant_id:
            headers["X-Scope-OrgID"] = self.config.tenant_id
        return headers

    def _now_epoch_ns(self) -> str:
        return str(int(datetime.now(timezone.utc).timestamp() * 1_000_000_000))

    def _parse_time_to_epoch_ns(self, value: Optional[str]) -> Optional[str]:
        """把相对时间、RFC3339 和不同精度的 Unix 时间统一转成纳秒。"""
        if not value:
            return None

        normalized = value.strip().lower()
        if normalized == "now":
            return self._now_epoch_ns()

        relative = re.match(r"^-?(\d+)([smhdw])$", normalized)
        if relative:
            amount = int(relative.group(1))
            unit = relative.group(2)
            multipliers = {
                "s": 1,
                "m": 60,
                "h": 60 * 60,
                "d": 60 * 60 * 24,
                "w": 60 * 60 * 24 * 7,
            }
            timestamp = datetime.now(timezone.utc) - timedelta(
                seconds=amount * multipliers[unit]
            )
            return str(int(timestamp.timestamp() * 1_000_000_000))

        if re.match(r"^\d+$", normalized):
            epoch = int(normalized)
            if epoch < 1_000_000_000_000:
                return str(epoch * 1_000_000_000)
            if epoch < 1_000_000_000_000_000:
                return str(epoch * 1_000_000)
            if epoch < 1_000_000_000_000_000_000:
                return str(epoch * 1_000)
            return str(epoch)

        try:
            parsed = datetime.fromisoformat(normalized.replace("z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return str(int(parsed.timestamp() * 1_000_000_000))
        except ValueError:
            return normalized

    def create_tool_schemas(self) -> List[ToolDefinition]:
        """创建七个 Loki 只读工具的 schema。"""
        return [
            ToolDefinition(
                name=self._make_tool_name("query"),
                description="使用 LogQL 查询某一时刻的日志或日志指标。",
                capability=ToolCapability.LOGS,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "LogQL 表达式。"},
                        "time": {
                            "type": "string",
                            "description": "查询时刻，可使用 RFC3339 或 Unix 时间。",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "最大日志条数。",
                            "minimum": 1,
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["forward", "backward"],
                            "description": "日志排序方向。",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("query_range"),
                description="使用 LogQL 查询一段时间内的日志或日志指标。",
                capability=ToolCapability.LOGS,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "LogQL 表达式。"},
                        "start": {"type": "string", "description": "开始时间。"},
                        "end": {"type": "string", "description": "结束时间。"},
                        "step": {
                            "type": "string",
                            "description": "日志指标查询的采样步长。",
                        },
                        "limit": {"type": "integer", "description": "最大日志条数。"},
                        "interval": {
                            "type": "string",
                            "description": "日志流返回间隔。",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["forward", "backward"],
                            "description": "日志排序方向。",
                        },
                    },
                    "required": ["query", "start", "end"],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("labels"),
                description="列出指定时间范围内可用的 Loki 标签。",
                capability=ToolCapability.LOGS,
                parameters={
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "description": "开始时间。"},
                        "end": {"type": "string", "description": "结束时间。"},
                        "since": {"type": "string", "description": "相对时间范围。"},
                        "query": {
                            "type": "string",
                            "description": "可选的日志流选择器。",
                        },
                    },
                    "required": [],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("label_values"),
                description="列出指定 Loki 标签在时间范围内的值。",
                capability=ToolCapability.LOGS,
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "标签名称。",
                            "minLength": 1,
                        },
                        "start": {"type": "string", "description": "开始时间。"},
                        "end": {"type": "string", "description": "结束时间。"},
                        "since": {"type": "string", "description": "相对时间范围。"},
                        "query": {
                            "type": "string",
                            "description": "可选的日志流选择器。",
                        },
                    },
                    "required": ["name"],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("series"),
                description="列出匹配选择器的唯一日志流标签集合。",
                capability=ToolCapability.LOGS,
                parameters={
                    "type": "object",
                    "properties": {
                        "match": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "一个或多个日志流选择器。",
                        },
                        "start": {"type": "string", "description": "开始时间。"},
                        "end": {"type": "string", "description": "结束时间。"},
                    },
                    "required": ["match"],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("volume"),
                description="按标签或日志流查询日志容量，需要 Loki 开启 volume API。",
                capability=ToolCapability.LOGS,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "日志流选择器。",
                            "minLength": 1,
                        },
                        "start": {"type": "string", "description": "开始时间。"},
                        "end": {"type": "string", "description": "结束时间。"},
                        "limit": {"type": "integer", "description": "最大结果数。"},
                        "targetLabels": {
                            "type": "string",
                            "description": "逗号分隔的目标标签。",
                        },
                        "aggregateBy": {
                            "type": "string",
                            "enum": ["series", "labels"],
                            "description": "聚合方式。",
                        },
                    },
                    "required": ["query", "start", "end"],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("patterns"),
                description="分析时间范围内的日志模式，需要 Loki 开启 pattern ingester。",
                capability=ToolCapability.LOGS,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "日志流选择器。",
                        },
                        "start": {"type": "string", "description": "开始时间。"},
                        "end": {"type": "string", "description": "结束时间。"},
                        "step": {"type": "string", "description": "采样步长。"},
                    },
                    "required": ["query", "start", "end"],
                },
            ),
        ]

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Any = None,
    ) -> Dict[str, Any]:
        """调用 Loki，并把连接、协议和 HTTP 错误统一为结构化结果。"""
        url = self.config.url.rstrip("/") + path
        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout,
                headers=self._headers(),
            ) as client:
                response = await client.request(
                    method,
                    url,
                    params=params,
                    data=data,
                )

            content_type = response.headers.get("content-type", "").lower()
            if content_type.startswith("application/json"):
                payload = response.json()
            else:
                payload = {"raw": response.text}

            if response.status_code >= 400:
                safe_payload = _safe_error_payload(payload)
                logger.error(
                    "Loki API error %s %s: %s",
                    method,
                    path,
                    safe_payload,
                )
                return {
                    "status": "error",
                    "code": response.status_code,
                    "error": safe_payload,
                }

            return {
                "status": "success",
                "code": response.status_code,
                "data": payload,
            }
        except Exception as exc:
            safe_error = _safe_error_message(exc)
            logger.error("Loki request failed %s %s: %s", method, path, safe_error)
            return {"status": "error", "error": safe_error}

    @staticmethod
    def _selector_from_labels(labels: Dict[str, str]) -> str:
        parts = []
        for key, value in (labels or {}).items():
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'{key}="{escaped}"')
        return "{" + ",".join(parts) + "}"

    def _fix_empty_stream_selector(self, query: str) -> str:
        """为空 selector 选择稳定日志流，避免 Loki 拒绝空兼容匹配器。"""
        matched = re.match(r"^\s*\{\s*\}(.*)$", query or "")
        if not matched:
            return query

        suffix = matched.group(1)
        selectors: List[str] = []
        instance_config = self.instance_config
        if isinstance(instance_config, LokiInstanceConfig):
            for labels in instance_config.prefer_streams or []:
                selectors.append(self._selector_from_labels(labels))
            if instance_config.default_selector:
                selectors.append(instance_config.default_selector.strip())

        if self.config.default_selector:
            selectors.append(self.config.default_selector.strip())

        selectors = [selector for selector in selectors if selector]
        if len(selectors) == 1:
            return f"{selectors[0]}{suffix}"
        if selectors:
            return " or ".join(f"({selector}{suffix})" for selector in selectors)
        return f'({{job=~".+"}}{suffix}) or ({{service=~".+"}}{suffix})'

    @status_update("正在查询 Loki 当前日志。")
    async def query(
        self,
        query: str,
        time: Optional[str] = None,
        limit: Optional[int] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"query": self._fix_empty_stream_selector(query)}
        if time:
            params["time"] = time
        if limit is not None:
            params["limit"] = int(limit)
        if direction:
            params["direction"] = direction
        return await self._request("GET", "/loki/api/v1/query", params=params)

    @status_update("正在查询 {start} 到 {end} 的 Loki 日志。")
    async def query_range(
        self,
        query: str,
        start: str,
        end: str,
        step: Optional[str] = None,
        limit: Optional[int] = None,
        interval: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "query": self._fix_empty_stream_selector(query),
            "start": self._parse_time_to_epoch_ns(start),
            "end": self._parse_time_to_epoch_ns(end) or self._now_epoch_ns(),
        }
        if step:
            params["step"] = step
        if limit is not None:
            params["limit"] = int(limit)
        if interval:
            params["interval"] = interval
        if direction:
            params["direction"] = direction
        return await self._request("GET", "/loki/api/v1/query_range", params=params)

    @status_update("正在查询 Loki 可用标签。")
    async def labels(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        since: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if start:
            params["start"] = self._parse_time_to_epoch_ns(start)
        if end:
            params["end"] = self._parse_time_to_epoch_ns(end) or self._now_epoch_ns()
        if since:
            params["since"] = since
        if query:
            params["query"] = query
        return await self._request("GET", "/loki/api/v1/labels", params=params)

    @status_update("正在查询 Loki 标签值。")
    async def label_values(
        self,
        name: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        since: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if start:
            params["start"] = self._parse_time_to_epoch_ns(start)
        if end:
            params["end"] = self._parse_time_to_epoch_ns(end) or self._now_epoch_ns()
        if since:
            params["since"] = since
        if query:
            params["query"] = query
        return await self._request(
            "GET",
            f"/loki/api/v1/label/{name}/values",
            params=params,
        )

    @status_update("正在查询 Loki 日志流。")
    async def series(
        self,
        match: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if start:
            params["start"] = self._parse_time_to_epoch_ns(start)
        if end:
            params["end"] = self._parse_time_to_epoch_ns(end) or self._now_epoch_ns()
        form_data = {"match[]": match}
        return await self._request(
            "POST",
            "/loki/api/v1/series",
            params=params,
            data=form_data,
        )

    @status_update("正在查询 Loki 日志容量。")
    async def volume(
        self,
        query: str,
        start: str,
        end: str,
        limit: Optional[int] = None,
        target_labels: Optional[str] = None,
        aggregate_by: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # schema 使用 Loki API 的 camelCase；Python 调用仍兼容 snake_case。
        if target_labels is None and "targetLabels" in kwargs:
            target_labels = kwargs.get("targetLabels")
        if aggregate_by is None and "aggregateBy" in kwargs:
            aggregate_by = kwargs.get("aggregateBy")

        params: Dict[str, Any] = {
            "query": self._fix_empty_stream_selector(query),
            "start": self._parse_time_to_epoch_ns(start),
            "end": self._parse_time_to_epoch_ns(end) or self._now_epoch_ns(),
        }
        if limit is not None:
            params["limit"] = int(limit)
        if target_labels:
            params["targetLabels"] = target_labels
        if aggregate_by:
            params["aggregateBy"] = aggregate_by
        return await self._request(
            "GET",
            "/loki/api/v1/index/volume",
            params=params,
        )

    @status_update("正在分析 Loki 日志模式。")
    async def patterns(
        self,
        query: str,
        start: str,
        end: str,
        step: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "query": self._fix_empty_stream_selector(query),
            "start": self._parse_time_to_epoch_ns(start),
            "end": self._parse_time_to_epoch_ns(end) or self._now_epoch_ns(),
        }
        if step:
            params["step"] = step
        return await self._request("GET", "/loki/api/v1/patterns", params=params)
