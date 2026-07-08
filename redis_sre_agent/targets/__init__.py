"""可插拔 Redis target discovery/binding 运行时。"""

from .contracts import (
    BindingRequest,
    BindingResult,
    DiscoveryCandidate,
    DiscoveryRequest,
    DiscoveryResponse,
    ProviderLoadRequest,
    PublicTargetBinding,
    PublicTargetMatch,
    TargetBindingStrategy,
    TargetDiscoveryBackend,
    TargetHandleRecord,
)
from .handle_store import RedisTargetHandleStore, get_target_handle_store
from .registry import TargetIntegrationRegistry, get_target_integration_registry
from .services import TargetBindingService, TargetDiscoveryService

__all__ = [
    "BindingRequest",
    "BindingResult",
    "DiscoveryCandidate",
    "DiscoveryRequest",
    "DiscoveryResponse",
    "ProviderLoadRequest",
    "PublicTargetBinding",
    "PublicTargetMatch",
    "RedisTargetHandleStore",
    "TargetBindingService",
    "TargetBindingStrategy",
    "TargetDiscoveryBackend",
    "TargetDiscoveryService",
    "TargetHandleRecord",
    "TargetIntegrationRegistry",
    "get_target_handle_store",
    "get_target_integration_registry",
]
