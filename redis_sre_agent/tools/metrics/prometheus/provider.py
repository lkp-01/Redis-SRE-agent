"""通过 Prometheus HTTP API 收集只读指标证据。"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import requests
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime
from pydantic import Field
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
    """保留服务位置但移除 URL 中的用户信息、查询参数和片段。"""
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
        return "<redacted-prometheus-url>"


def _safe_error_message(value: Any) -> str:
    """清洗外部异常文本，避免连接凭据进入工具结果或日志。"""
    message = str(value)
    message = _HTTP_URL_PATTERN.sub(lambda match: _safe_url(match.group(0)), message)
    return _SENSITIVE_VALUE_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        message,
    )


class PrometheusConfig(BaseSettings):
    """从 `TOOLS_PROMETHEUS_` 环境变量读取 Prometheus 连接配置。"""

    model_config = SettingsConfigDict(env_prefix="tools_prometheus_")

    url: str = Field(
        default="http://localhost:9090",
        description="Prometheus HTTP API 地址。",
    )
    disable_ssl: bool = Field(
        default=False,
        description="是否关闭 HTTPS 证书校验。",
    )


class PrometheusToolProvider(ToolProvider):
    """提供即时查询、范围查询和指标发现三个只读工具。"""

    capabilities = {ToolCapability.METRICS}

    def __init__(
        self,
        redis_instance: Optional[RedisInstance] = None,
        config: Optional[PrometheusConfig] = None,
    ):
        super().__init__(redis_instance)
        self.config = config or PrometheusConfig()
        self._client: Optional[PrometheusConnect] = None

    @property
    def provider_name(self) -> str:
        return "prometheus"

    def create_tool_schemas(self) -> List[ToolDefinition]:
        """创建三个 Prometheus 只读工具的 schema。"""
        return [
            ToolDefinition(
                name=self._make_tool_name("query"),
                description=(
                    "使用 PromQL 查询某一时刻的指标值。适合查看 Redis 可用性、内存、"
                    "连接数和命令速率等当前状态。"
                ),
                capability=ToolCapability.METRICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "PromQL 查询表达式，例如 redis_up。",
                        }
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("query_range"),
                description=(
                    "使用 PromQL 查询一段时间内的指标序列，用于观察趋势和定位异常时段。"
                ),
                capability=ToolCapability.METRICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "PromQL 查询表达式。",
                        },
                        "start_time": {
                            "type": "string",
                            "description": "开始时间，可使用 1h 等相对时间或绝对时间。",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "结束时间，默认 now。",
                            "default": "now",
                        },
                        "step": {
                            "type": "string",
                            "description": "采样步长，默认 15s。",
                            "default": "15s",
                        },
                    },
                    "required": ["query", "start_time"],
                },
            ),
            ToolDefinition(
                name=self._make_tool_name("search_metrics"),
                description=(
                    "按名称片段和可选标签查找指标。pattern 为空时列出全部指标名称。"
                ),
                capability=ToolCapability.METRICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "不区分大小写的指标名称片段。",
                            "default": "",
                        },
                        "label_filters": {
                            "type": "object",
                            "description": "可选标签过滤，例如 job=redis。",
                        },
                    },
                    "required": [],
                },
            ),
        ]

    def get_client(self) -> PrometheusConnect:
        """首次需要 fallback 时才创建 prometheus-api-client。"""
        if self._client is None:
            self._client = PrometheusConnect(
                url=self.config.url,
                disable_ssl=self.config.disable_ssl,
            )
            logger.info("Prometheus client initialized for %s", _safe_url(self.config.url))
        return self._client

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        return None

    async def _wait_for_targets(self, timeout_seconds: float = 10.0) -> None:
        """尽力等待首次抓取；没有 active target 时仍允许后续查询。"""
        base = self.config.url.rstrip("/")
        deadline = time.time() + timeout_seconds
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    f"{base}/api/v1/targets",
                    timeout=2,
                    verify=not self.config.disable_ssl,
                )
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("status") == "success":
                        active = payload.get("data", {}).get("activeTargets", [])
                        if active:
                            return
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(0.5)

        if last_error is not None:
            logger.debug(
                "Prometheus target readiness check failed: %s",
                _safe_error_message(last_error),
            )

    async def _http_get_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """在线程中执行阻塞 HTTP GET，并统一返回结构化错误。"""
        url = f"{self.config.url.rstrip('/')}{path}"
        try:
            response = await asyncio.to_thread(
                requests.get,
                url,
                params=params,
                timeout=5,
                verify=not self.config.disable_ssl,
            )
        except Exception as exc:
            return {"status": "error", "error": _safe_error_message(exc)}

        try:
            return response.json()
        except Exception:
            body = _safe_error_message(str(getattr(response, "text", ""))[:200])
            return {"status": "error", "error": f"Non-JSON response: {body}"}

    @status_update("正在查询 Prometheus 当前指标。")
    async def query(self, query: str) -> Dict[str, Any]:
        """执行 PromQL 即时查询。"""
        logger.info("Prometheus instant query requested.")
        try:
            await self._wait_for_targets(timeout_seconds=10.0)

            delays = [0.5, 1.0, 1.5, 2.0, 2.0]
            payload = await self._http_get_json(
                "/api/v1/query",
                params={"query": query},
            )
            if payload.get("status") != "success":
                return {
                    "status": "error",
                    "error": _safe_error_message(
                        payload.get("error") or "Prometheus query error"
                    ),
                    "query": query,
                }

            result = payload.get("data", {}).get("result", [])
            if not result:
                for delay in delays:
                    await asyncio.sleep(delay)
                    payload = await self._http_get_json(
                        "/api/v1/query",
                        params={"query": query},
                    )
                    if payload.get("status") != "success":
                        return {
                            "status": "error",
                            "error": _safe_error_message(
                                payload.get("error") or "Prometheus query error"
                            ),
                            "query": query,
                        }
                    result = payload.get("data", {}).get("result", [])
                    if result:
                        break

            if not result and query.strip() == "up":
                targets = await self._http_get_json("/api/v1/targets")
                active = (
                    targets.get("data", {}).get("activeTargets", [])
                    if targets.get("status") == "success"
                    else []
                )
                if active:
                    now_timestamp = int(datetime.now(timezone.utc).timestamp())
                    result = []
                    for target in active:
                        labels = target.get("labels", {})
                        metric = {
                            "__name__": "up",
                            **{
                                key: value
                                for key, value in labels.items()
                                if key in ("job", "instance", "service")
                            },
                        }
                        result.append(
                            {"metric": metric, "value": [now_timestamp, "1"]}
                        )

            return {
                "status": "success",
                "query": query,
                "data": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Prometheus instant query failed: %s", error)
            return {"status": "error", "error": error, "query": query}

    @status_update("正在查询 {start_time} 到 {end_time} 的 Prometheus 指标。")
    async def query_range(
        self,
        query: str,
        start_time: str,
        end_time: str = "now",
        step: str = "15s",
    ) -> Dict[str, Any]:
        """执行 PromQL 范围查询。"""
        logger.info("Prometheus range query requested.")
        try:
            await self._wait_for_targets(timeout_seconds=10.0)

            parsed_start = parse_datetime(start_time)
            parsed_end = parse_datetime(end_time)
            if parsed_start is None:
                return {
                    "status": "error",
                    "error": f"Invalid start_time format: {start_time}",
                    "query": query,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            if parsed_end is None:
                return {
                    "status": "error",
                    "error": f"Invalid end_time format: {end_time}",
                    "query": query,
                    "start_time": start_time,
                    "end_time": end_time,
                }

            start_param = int(parsed_start.timestamp())
            end_param = int(parsed_end.timestamp())
            params = {
                "query": query,
                "start": start_param,
                "end": end_param,
                "step": step,
            }
            payload = await self._http_get_json(
                "/api/v1/query_range",
                params=params,
            )
            if payload.get("status") != "success":
                return {
                    "status": "error",
                    "error": _safe_error_message(
                        payload.get("error") or "Prometheus range query error"
                    ),
                    "query": query,
                    "start_time": start_time,
                    "end_time": end_time,
                }

            result = payload.get("data", {}).get("result", [])
            if not result:
                for delay in [1.0, 1.5, 2.0]:
                    await asyncio.sleep(delay)
                    payload = await self._http_get_json(
                        "/api/v1/query_range",
                        params=params,
                    )
                    if payload.get("status") != "success":
                        return {
                            "status": "error",
                            "error": _safe_error_message(
                                payload.get("error") or "Prometheus range query error"
                            ),
                            "query": query,
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    result = payload.get("data", {}).get("result", [])
                    if result:
                        break

            if not result and query.strip() == "up":
                targets = await self._http_get_json("/api/v1/targets")
                active = (
                    targets.get("data", {}).get("activeTargets", [])
                    if targets.get("status") == "success"
                    else []
                )
                if active:
                    labels = active[0].get("labels", {})
                    metric = {
                        "__name__": "up",
                        **{
                            key: value
                            for key, value in labels.items()
                            if key in ("job", "instance", "service")
                        },
                    }
                    result = [
                        {
                            "metric": metric,
                            "values": [
                                [start_param, "1"],
                                [end_param, "1"],
                            ],
                        }
                    ]

            return {
                "status": "success",
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
                "step": step,
                "data": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Prometheus range query failed: %s", error)
            return {
                "status": "error",
                "error": error,
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
            }

    @status_update("正在按名称查找 Prometheus 指标。")
    async def search_metrics(
        self,
        pattern: str = "",
        label_filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """按名称片段查找指标，并可继续验证标签是否存在。"""
        logger.info("Prometheus metric discovery requested.")
        try:
            payload = await self._http_get_json(
                "/api/v1/label/__name__/values"
            )
            all_metrics = (
                payload.get("data", [])
                if payload.get("status") == "success"
                else []
            )

            if pattern:
                pattern_lower = pattern.lower()
                filtered_metrics = [
                    metric
                    for metric in all_metrics
                    if pattern_lower in metric.lower()
                ]
            else:
                pattern_lower = ""
                filtered_metrics = all_metrics

            if pattern and not filtered_metrics:
                try:
                    await asyncio.sleep(1)
                    payload = await self._http_get_json(
                        "/api/v1/label/__name__/values"
                    )
                    all_metrics = (
                        payload.get("data", [])
                        if payload.get("status") == "success"
                        else []
                    )
                    filtered_metrics = [
                        metric
                        for metric in all_metrics
                        if pattern_lower in metric.lower()
                    ]
                except Exception:
                    pass

            if pattern and not filtered_metrics:
                try:
                    client = self.get_client()
                    fetched = await asyncio.to_thread(client.all_metrics)
                    filtered_metrics = [
                        metric
                        for metric in fetched
                        if pattern_lower in metric.lower()
                    ]
                    if not filtered_metrics:
                        await asyncio.sleep(1)
                        fetched = await asyncio.to_thread(client.all_metrics)
                        filtered_metrics = [
                            metric
                            for metric in fetched
                            if pattern_lower in metric.lower()
                        ]
                except Exception as exc:
                    logger.debug(
                        "Prometheus client metric fallback failed: %s",
                        _safe_error_message(exc),
                    )

            if label_filters:
                label_parts = [
                    f'{key}="{value}"'
                    for key, value in label_filters.items()
                ]
                label_selector = "{" + ",".join(label_parts) + "}"
                try:
                    metrics_with_labels = []
                    for metric in filtered_metrics:
                        payload = await self._http_get_json(
                            "/api/v1/query",
                            params={"query": f"{metric}{label_selector}"},
                        )
                        result = (
                            payload.get("data", {}).get("result", [])
                            if payload.get("status") == "success"
                            else []
                        )
                        if result:
                            metrics_with_labels.append(metric)
                    filtered_metrics = metrics_with_labels
                except Exception as exc:
                    logger.warning(
                        "Prometheus label filtering failed: %s",
                        _safe_error_message(exc),
                    )

            return {
                "status": "success",
                "pattern": pattern,
                "label_filters": label_filters or {},
                "metrics": filtered_metrics,
                "count": len(filtered_metrics),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.error("Prometheus metric discovery failed: %s", error)
            return {
                "status": "error",
                "error": error,
                "pattern": pattern,
                "label_filters": label_filters or {},
            }
