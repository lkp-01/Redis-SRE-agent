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

    # 强制隔离，只提取与 Embedding 相关的配置项
    embedding_api_key: Optional[SecretStr]  # API Key，使用 pydantic.SecretStr 保护敏感信息防泄漏
    embedding_base_url: Optional[str]  # 代理地址或私有化部署的 Base URL
    embedding_provider: str  # 供应商名称，如 "openai", "local", "azure" 等
    embedding_model: str  # 具体的模型名称，如 "text-embedding-3-small"
    vector_dim: int  # 向量的输出维度，用于校验和索引创建
    vectorizer_factory: Optional[str]  # 自定义向量生成器工厂的 Python 导入路径

    @classmethod
    def from_settings(cls, config: Settings) -> "EmbeddingConfig":
        # 工厂方法：从全局的 Settings 对象中提取出这个“最小配置视图”
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

    # 这是一个 Typing Protocol（接口定义），规定了自定义工厂函数必须接收的参数格式
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

    # 规定了真正的“向量化工具”实例必须实现的两个异步方法 (鸭子类型约束)
    async def aembed(self, *args: Any, **kwargs: Any) -> Any: ...  # 异步单文本向量化

    async def aembed_many(self, *args: Any, **kwargs: Any) -> Any: ...  # 异步多文本/批量向量化

# 缓存当前使用的工厂实例和路径，避免重复解析和加载
_vectorizer_factory: Optional[VectorizerFactory] = None
_settings_vectorizer_factory: Optional[VectorizerFactory] = None
_settings_factory_path: Optional[str] = None


def _resolve_factory_from_path(factory_path: str) -> VectorizerFactory:
    # 将字符串路径 (如 "my_pkg.my_module.my_factory") 从右往左拆分成模块和函数名
    module_path, _, func_name = factory_path.rpartition(".")
    if not module_path or not func_name:
        raise ValueError(
            "vectorizer_factory 必须是 'package.module.callable' 形式的导入路径。"
        )
    # 动态导入该模块 (类似 import my_pkg.my_module)
    module = importlib.import_module(module_path)
    # 从模块中获取对应的工厂函数或类
    factory = getattr(module, func_name)
    # 校验获取到的对象是否可以被调用
    if not callable(factory):
        raise ValueError("vectorizer_factory 指向的对象不可调用。")
    return factory


def set_vectorizer_factory(factory: Optional[VectorizerFactory]) -> None:
    """注册测试或部署方提供的向量器工厂。"""
    # 显式依赖注入的入口：允许在运行时（如单元测试或应用启动时）手动覆盖工厂，而不依赖配置文件
    global _vectorizer_factory
    _vectorizer_factory = factory


def get_vectorizer_factory() -> Optional[VectorizerFactory]:
    # 优先返回手动注入的工厂；其次返回从配置中动态加载的工厂
    return _vectorizer_factory or _settings_vectorizer_factory


def _load_settings_factory(factory_path: str) -> VectorizerFactory:
    global _settings_vectorizer_factory, _settings_factory_path
    # 性能优化：如果路径没变，直接返回缓存的工厂对象，防止频繁执行 importlib 反射
    if _settings_vectorizer_factory is not None and _settings_factory_path == factory_path:
        return _settings_vectorizer_factory

    # 解析并加载工厂
    factory = _resolve_factory_from_path(factory_path)
    # 更新本地缓存
    _settings_vectorizer_factory = factory
    _settings_factory_path = factory_path
    # (logger info 被忽略)
    return factory


def _resolve_factory(config: EmbeddingConfig) -> Optional[VectorizerFactory]:
    """按显式注入、配置路径的顺序解析向量器工厂。"""
    if _vectorizer_factory is not None:
        return _vectorizer_factory
    if config.vectorizer_factory:
        return _load_settings_factory(config.vectorizer_factory)
    return None


def validate_embedding_config(config: Settings | EmbeddingConfig) -> EmbeddingConfig:
    """只检查 embedding 边界；错误文本不包含任何 secret 或 URL。"""
    # 兼容处理：支持传入完整的系统 Settings 或是已提取的 EmbeddingConfig
    embedding = (
        config if isinstance(config, EmbeddingConfig) else EmbeddingConfig.from_settings(config)
    )
    # 基础校验：维度必须是正数
    if embedding.vector_dim <= 0:
        raise ValueError("vector_dim 必须是正整数。")
    # 基础校验：模型名不能为空白字符
    if not embedding.embedding_model.strip():
        raise ValueError("embedding_model 不能为空。")

    # 如果系统已经有了有效工厂（手动注入的或之前加载过的），就不必往下校验严格的 provider 了
    if _vectorizer_factory is not None:
        return embedding
    if embedding.vectorizer_factory:
        _load_settings_factory(embedding.vectorizer_factory)
        return embedding

    # 兜底校验内置支持的 Provider
    provider = embedding.embedding_provider.strip().lower()
    if provider == "openai":
        # 针对 OpenAI 的特殊校验，必须要有 API Key
        if embedding.embedding_api_key is None:
            raise ValueError("embedding_provider=openai 时必须配置 embedding_api_key。")
        return embedding
    if provider == "local":
        # 预留占位：当前系统尚未集成离线模型依赖
        raise ValueError("当前阶段未安装本地 sentence-transformers 向量器依赖。")

    # 无法识别的供应商
    raise ValueError("未知 embedding_provider；请配置 openai 或自定义 vectorizer_factory。")


def _build_embeddings_cache(config: Settings) -> EmbeddingsCache:
    """沿用 original 的 RedisVL cache；连接信息不会传给向量器工厂配置视图。"""
    # 构造 RedisVL 原生的缓存层组件
    return EmbeddingsCache(
        name="sre_embeddings_cache",
        # 从全局 Settings 拿 Redis 密码/URL（这就是为什么 cache 初始化要在工厂外进行，防止向 Embedding 暴露 Redis 权限）
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
    # 兜底逻辑：如果没有自定义工厂，默认只接受 OpenAI 协议
    if provider.strip().lower() != "openai":
        raise ValueError("默认向量器只支持 embedding_provider=openai。")
    if config.embedding_api_key is None:
        raise ValueError("embedding_provider=openai 时必须配置 embedding_api_key。")

    # 延迟导入 redis_core，防止产生模块间的循环导入问题
    from redis_sre_agent.core import redis as redis_core

    # 构建传给底层 API 的参数，解密 SecretStr
    api_config: dict[str, str] = {
        "api_key": config.embedding_api_key.get_secret_value(),
    }
    # 允许覆写代理/私有化模型的 endpoint
    if config.embedding_base_url:
        api_config["base_url"] = config.embedding_base_url

    # 实例化并返回默认的文本向量生成器
    return redis_core.OpenAITextVectorizer(
        model=model or config.embedding_model,
        cache=cache,
        api_config=api_config,
        **kwargs,
    )


def _validate_vectorizer_instance(vectorizer: Any) -> Vectorizer:
    # 反射检查工厂生成的实例是否真正实现了协议所需的方法
    missing = [
        method
        for method in ("aembed", "aembed_many")
        if not callable(getattr(vectorizer, method, None))
    ]
    # 如果缺胳膊少腿，直接在启动/初始化时报错，避免在运行时查不到数据才抛异常
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
    # --- 这是整个模块对外的核心门面函数 (Facade) ---
    # 1. 决定用哪个配置（传入的，或全局的）
    cfg = config or settings
    # 2. 剥离并校验 Embedding 的专属配置
    embedding_config = validate_embedding_config(cfg)
    # 3. 决定用哪个工厂（自定义加载的优先，否则用系统默认的 OpenAI 兼容工厂）
    factory = _resolve_factory(embedding_config) or _default_vectorizer_factory
    # 4. 构建 Redis 缓存组件
    cache = _build_embeddings_cache(cfg)
    # 5. 调用工厂方法，把所有材料（供应商、模型、配置、缓存）丢进去
    vectorizer = factory(
        provider=embedding_config.embedding_provider,
        model=model or embedding_config.embedding_model,
        config=embedding_config,
        cache=cache,
        **kwargs,
    )
    # 6. 验证产出的对象合法性并返回
    return _validate_vectorizer_instance(vectorizer)
