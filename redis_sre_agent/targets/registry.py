"""

"""

from __future__ import annotations

import importlib
from threading import Lock
from typing import Any, Dict, Optional

from redis_sre_agent.core.config import settings

from .contracts import (
    AuthenticatedClientFactory,
    DiscoveryCandidate,
    TargetBindingStrategy,
    TargetDiscoveryBackend,
)

_DEFAULT_REGISTRY: "TargetIntegrationRegistry | None" = None
_DEFAULT_REGISTRY_LOCK = Lock()

# 按需加载：把所有的class的文件地址写入一个yaml或json，根据需要查出来地址放到这个函数里然后加载那个模块/class
def _load_object(class_path: str) -> Any:
    module_path, attr_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)

# 没看懂代码，也不知道具体是在调用什么。但大致知道是去调用各种不同干活的工人/class的
class TargetIntegrationRegistry:
    """运行期 target 集成注册表。"""

    def __init__(
        self,
        *,
        default_discovery_backend: str,
        default_binding_strategy: str,
    ) -> None:
        self.default_discovery_backend = default_discovery_backend
        self.default_binding_strategy = default_binding_strategy
        self._discovery_backends: Dict[str, TargetDiscoveryBackend] = {}
        self._binding_strategies: Dict[str, TargetBindingStrategy] = {}
        self._client_factories: Dict[str, AuthenticatedClientFactory] = {}

    def register_discovery_backend(
        self, backend: TargetDiscoveryBackend, *, name: Optional[str] = None
    ) -> None:
        self._discovery_backends[name or backend.backend_name] = backend

    def register_binding_strategy(
        self, strategy: TargetBindingStrategy, *, name: Optional[str] = None
    ) -> None:
        self._binding_strategies[name or strategy.strategy_name] = strategy

    def register_client_factory(
        self, factory: AuthenticatedClientFactory, *, family: Optional[str] = None
    ) -> None:
        self._client_factories[family or factory.client_family] = factory

    def get_discovery_backend(self, name: Optional[str] = None) -> TargetDiscoveryBackend:
        backend_name = name or self.default_discovery_backend
        try:
            return self._discovery_backends[backend_name]
        except KeyError as exc:
            raise ValueError(f"Unknown target discovery backend: {backend_name}") from exc

    def get_binding_strategy(self, name: str) -> TargetBindingStrategy:
        try:
            return self._binding_strategies[name]
        except KeyError as exc:
            raise ValueError(f"Unknown target binding strategy: {name}") from exc

    def get_client_factory(self, family: str) -> AuthenticatedClientFactory:
        try:
            return self._client_factories[family]
        except KeyError as exc:
            raise ValueError(f"Unknown target client factory: {family}") from exc

    def validate_candidate(self, candidate: DiscoveryCandidate) -> None:
        self.get_binding_strategy(candidate.binding_strategy)

    @classmethod
    def from_settings(cls) -> "TargetIntegrationRegistry":
        integrations = settings.target_integrations
        registry = cls(
            default_discovery_backend=integrations.default_discovery_backend,
            default_binding_strategy=integrations.default_binding_strategy,
        )
        for name, config in integrations.discovery_backends.items():
            backend_cls = _load_object(config.class_path)
            registry.register_discovery_backend(backend_cls(**dict(config.config or {})), name=name)
        for name, config in integrations.binding_strategies.items():
            strategy_cls = _load_object(config.class_path)
            registry.register_binding_strategy(
                strategy_cls(**dict(config.config or {})), name=name
            )
        for family, config in integrations.client_factories.items():
            factory_cls = _load_object(config.class_path)
            registry.register_client_factory(factory_cls(**dict(config.config or {})), family=family)
        return registry

# 如果没有TargetIntegrationRegistry就拉去一个；有了就直接用它；加锁避免同时拉队伍导致混乱
def get_target_integration_registry() -> TargetIntegrationRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        with _DEFAULT_REGISTRY_LOCK:
            if _DEFAULT_REGISTRY is None:
                _DEFAULT_REGISTRY = TargetIntegrationRegistry.from_settings()
    return _DEFAULT_REGISTRY

# 把当前的TargetIntegrationRegistry取消
def reset_target_integration_registry() -> None:
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        _DEFAULT_REGISTRY = None
