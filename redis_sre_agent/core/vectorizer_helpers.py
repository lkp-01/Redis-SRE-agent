"""创建向量器并隔离 embedding 与 Agent chat 配置。"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from pydantic import SecretStr
from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache

from redis_sre_agent.core.config import Settings, settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingConfig:
    """传给向量器工厂的最小配置视图，不包含任何 chat 或 Redis 凭据。"""

    embedding_api_key: Optional[SecretStr]
    embedding_base_url: Optional[str]
    embedding_provider: str
    embedding_model: str
    vector_dim: int
    vectorizer_factory: Optional[str]

    @classmethod
    def from_settings(cls, config: Settings) -> "EmbeddingConfig":
        return cls(
            embedding_api_key=config.embedding_api_key,
            embedding_base_url=config.embedding_base_url,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            vector_dim=config.vector_dim,
            vectorizer_factory=config.vectorizer_factory,
        )


class VectorizerFactory(Protocol):
    """自定义向量器工厂的 original 兼容调用形状。"""

    def __call__(
        self,
        *,
        provider: str,
        model: Optional[str] = None,
        config: EmbeddingConfig,
        cache: EmbeddingsCache,
        **kwargs: Any,
    ) -> Any: ...


class Vectorizer(Protocol):
    """摄取和检索实际依赖的最小异步向量器协议。"""

    async def aembed(self, *args: Any, **kwargs: Any) -> Any: ...

    async def aembed_many(self, *args: Any, **kwargs: Any) -> Any: ...


_vectorizer_factory: Optional[VectorizerFactory] = None
_settings_vectorizer_factory: Optional[VectorizerFactory] = None
_settings_factory_path: Optional[str] = None


def _resolve_factory_from_path(factory_path: str) -> VectorizerFactory:
    module_path, _, func_name = factory_path.rpartition(".")
    if not module_path or not func_name:
        raise ValueError(
            "vectorizer_factory 必须是 'package.module.callable' 形式的导入路径。"
        )
    module = importlib.import_module(module_path)
    factory = getattr(module, func_name)
    if not callable(factory):
        raise ValueError("vectorizer_factory 指向的对象不可调用。")
    return factory


def set_vectorizer_factory(factory: Optional[VectorizerFactory]) -> None:
    """注册测试或部署方提供的向量器工厂。"""

    global _vectorizer_factory
    _vectorizer_factory = factory


def get_vectorizer_factory() -> Optional[VectorizerFactory]:
    return _vectorizer_factory or _settings_vectorizer_factory


def _load_settings_factory(factory_path: str) -> VectorizerFactory:
    global _settings_vectorizer_factory, _settings_factory_path
    if _settings_vectorizer_factory is not None and _settings_factory_path == factory_path:
        return _settings_vectorizer_factory
    factory = _resolve_factory_from_path(factory_path)
    _settings_vectorizer_factory = factory
    _settings_factory_path = factory_path
    logger.info("已加载自定义 embedding vectorizer factory。")
    return factory


def _resolve_factory(config: EmbeddingConfig) -> Optional[VectorizerFactory]:
    # set_vectorizer_factory 是显式运行时注入，优先于配置文件路径，便于离线测试。
    if _vectorizer_factory is not None:
        return _vectorizer_factory
    if config.vectorizer_factory:
        return _load_settings_factory(config.vectorizer_factory)
    return None


def validate_embedding_config(config: Settings | EmbeddingConfig) -> EmbeddingConfig:
    """只检查 embedding 边界；错误文本不包含任何 secret 或 URL。"""

    embedding = (
        config if isinstance(config, EmbeddingConfig) else EmbeddingConfig.from_settings(config)
    )
    if embedding.vector_dim <= 0:
        raise ValueError("vector_dim 必须是正整数。")
    if not embedding.embedding_model.strip():
        raise ValueError("embedding_model 不能为空。")

    if _vectorizer_factory is not None:
        return embedding
    if embedding.vectorizer_factory:
        _load_settings_factory(embedding.vectorizer_factory)
        return embedding

    provider = embedding.embedding_provider.strip().lower()
    if provider == "openai":
        if embedding.embedding_api_key is None:
            raise ValueError("embedding_provider=openai 时必须配置 embedding_api_key。")
        return embedding
    if provider == "local":
        raise ValueError("当前阶段未安装本地 sentence-transformers 向量器依赖。")
    raise ValueError("未知 embedding_provider；请配置 openai 或自定义 vectorizer_factory。")


def _build_embeddings_cache(config: Settings) -> EmbeddingsCache:
    """沿用 original 的 RedisVL cache；连接信息不会传给向量器工厂配置视图。"""

    return EmbeddingsCache(
        name="sre_embeddings_cache",
        redis_url=config.redis_url.get_secret_value(),
        ttl=config.embeddings_cache_ttl,
    )


def _default_vectorizer_factory(
    *,
    provider: str,
    model: Optional[str],
    config: EmbeddingConfig,
    cache: EmbeddingsCache,
    **kwargs: Any,
) -> Any:
    """使用 RedisVL OpenAI-compatible vectorizer，但只注入独立 embedding 配置。"""

    if provider.strip().lower() != "openai":
        raise ValueError("默认向量器只支持 embedding_provider=openai。")
    if config.embedding_api_key is None:
        raise ValueError("embedding_provider=openai 时必须配置 embedding_api_key。")

    from redis_sre_agent.core import redis as redis_core

    api_config: dict[str, str] = {
        "api_key": config.embedding_api_key.get_secret_value(),
    }
    if config.embedding_base_url:
        api_config["base_url"] = config.embedding_base_url
    return redis_core.OpenAITextVectorizer(
        model=model or config.embedding_model,
        cache=cache,
        api_config=api_config,
        **kwargs,
    )


def _validate_vectorizer_instance(vectorizer: Any) -> Vectorizer:
    missing = [
        method
        for method in ("aembed", "aembed_many")
        if not callable(getattr(vectorizer, method, None))
    ]
    if missing:
        raise TypeError(
            "Vectorizer factory 必须返回实现异步方法的对象：" + ", ".join(missing)
        )
    return vectorizer


def create_vectorizer(
    config: Optional[Settings] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Vectorizer:
    cfg = config or settings
    embedding_config = validate_embedding_config(cfg)
    factory = _resolve_factory(embedding_config) or _default_vectorizer_factory
    cache = _build_embeddings_cache(cfg)
    vectorizer = factory(
        provider=embedding_config.embedding_provider,
        model=model or embedding_config.embedding_model,
        config=embedding_config,
        cache=cache,
        **kwargs,
    )
    return _validate_vectorizer_instance(vectorizer)
