"""统一创建 OpenAI-compatible 聊天模型，并提供单次调用级主副切换。

该模块沿用 original 的 `create_llm` / `create_mini_llm` / `create_nano_llm`
边界。当前默认端点是 DeepSeek；模块只创建客户端，不在导入时访问外部网络。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from redis_sre_agent.core.config import settings

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """真实模型配置不完整。"""


class LLMFailoverError(RuntimeError):
    """主副模型均无法完成同一次 LLM 调用。"""


class FailoverChatModel:
    """把同一次模型调用从主模型切换到副模型。

    wrapper 不重启 StateGraph，也不执行工具；它只在主模型调用抛出异常时，把完全相同的
    消息交给副模型。这样已经产生的 ToolMessage 会保留在图状态里，不会被这里主动重放。
    """

    def __init__(self, *, primary: Any, fallback: Any) -> None:
        self.primary = primary
        self.fallback = fallback

    def __repr__(self) -> str:
        return (
            "FailoverChatModel("
            f"primary={_model_name(self.primary)!r}, fallback={_model_name(self.fallback)!r})"
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FailoverChatModel":
        """给主副模型绑定同一份工具 schema。"""

        return FailoverChatModel(
            primary=self.primary.bind_tools(tools, **kwargs),
            fallback=self.fallback.bind_tools(tools, **kwargs),
        )

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        try:
            return await self.primary.ainvoke(input, **kwargs)
        except Exception as primary_exc:
            primary_error_type = type(primary_exc).__name__
            logger.warning(
                "Primary LLM call failed; switching to fallback (primary_error=%s).",
                primary_error_type,
            )

        try:
            return await self.fallback.ainvoke(input, **kwargs)
        except Exception as fallback_exc:
            logger.error(
                "Fallback LLM call failed (fallback_error=%s).",
                type(fallback_exc).__name__,
            )
            raise LLMFailoverError(
                "主模型和副模型调用均失败："
                f"primary={primary_error_type}, "
                f"fallback={type(fallback_exc).__name__}"
            ) from None

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        try:
            return self.primary.invoke(input, **kwargs)
        except Exception as primary_exc:
            primary_error_type = type(primary_exc).__name__
            logger.warning(
                "Primary LLM call failed; switching to fallback (primary_error=%s).",
                primary_error_type,
            )

        try:
            return self.fallback.invoke(input, **kwargs)
        except Exception as fallback_exc:
            logger.error(
                "Fallback LLM call failed (fallback_error=%s).",
                type(fallback_exc).__name__,
            )
            raise LLMFailoverError(
                "主模型和副模型调用均失败："
                f"primary={primary_error_type}, "
                f"fallback={type(fallback_exc).__name__}"
            ) from None


def _model_name(model: Any) -> str:
    return str(getattr(model, "model_name", type(model).__name__))


def has_llm_credentials() -> bool:
    """只判断是否配置了非空 key，不读取或输出 key 内容。"""

    key = settings.openai_api_key
    if key is None:
        return False
    if isinstance(key, SecretStr):
        return bool(key.get_secret_value().strip())
    return bool(str(key).strip())


def _api_key_value(api_key: Optional[str | SecretStr] = None) -> str:
    key = api_key if api_key is not None else settings.openai_api_key
    if key is None:
        raise LLMConfigurationError("未配置 OPENAI_API_KEY，无法创建 DeepSeek 客户端。")
    value = key.get_secret_value() if isinstance(key, SecretStr) else str(key)
    if not value.strip():
        raise LLMConfigurationError("OPENAI_API_KEY 为空，无法创建 DeepSeek 客户端。")
    return value


def _deepseek_extra_body(base_url: Optional[str]) -> Optional[dict[str, Any]]:
    if not base_url or "deepseek" not in base_url.lower():
        return None
    return {"thinking": {"type": settings.deepseek_thinking_mode}}


def _create_chat_model(
    *,
    model: str,
    api_key: Optional[str | SecretStr] = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    base_url = settings.openai_base_url
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": _api_key_value(api_key),
        "timeout": timeout if timeout is not None else settings.llm_timeout,
        **kwargs,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
    if "extra_body" not in llm_kwargs:
        extra_body = _deepseek_extra_body(base_url)
        if extra_body is not None:
            llm_kwargs["extra_body"] = extra_body
    return ChatOpenAI(**llm_kwargs)


def create_llm(
    model: Optional[str] = None,
    api_key: Optional[str | SecretStr] = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """创建主推理模型；启用 failover 时用 mini 模型作为副模型。"""

    primary = _create_chat_model(
        model=model or settings.openai_model,
        api_key=api_key,
        timeout=timeout,
        **kwargs,
    )
    fallback_model = settings.openai_model_mini
    if not settings.llm_failover_enabled or fallback_model == (model or settings.openai_model):
        return primary
    fallback = _create_chat_model(
        model=fallback_model,
        api_key=api_key,
        timeout=timeout,
        **kwargs,
    )
    return FailoverChatModel(primary=primary, fallback=fallback)


def create_mini_llm(
    model: Optional[str] = None,
    api_key: Optional[str | SecretStr] = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """创建轻量任务模型。"""

    return _create_chat_model(
        model=model or settings.openai_model_mini,
        api_key=api_key,
        timeout=timeout,
        **kwargs,
    )


def create_nano_llm(
    model: Optional[str] = None,
    api_key: Optional[str | SecretStr] = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """创建路由和简单分类模型。"""

    return _create_chat_model(
        model=model or settings.openai_model_nano,
        api_key=api_key,
        timeout=timeout,
        **kwargs,
    )
