"""统一 Redis target catalog、解析和绑定 helper。

阶段三的核心是先回答两个问题：用户想操作哪个 Redis target，以及 ToolManager
应该给这个 target 绑定哪些工具。这里构建的是安全目录，只包含名称、环境、用途等
公开信息，不包含 Redis URL、密码、token 或 DSN。
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from redis_sre_agent.core.clusters import RedisCluster, get_cluster_by_id, get_clusters
from redis_sre_agent.core.instances import RedisInstance, get_instance_by_id, get_instances
from redis_sre_agent.targets import TargetBindingService, TargetDiscoveryService
from redis_sre_agent.targets.contracts import (
    DiscoveryCandidate,
    DiscoveryRequest,
    DiscoveryResponse,
    PublicTargetBinding,
    PublicTargetMatch,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HOSTNAME_RE = re.compile(
    r"\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+\b"
)
_ENV_ALIASES = {
    "prod": "production",
    "production": "production",
    "stage": "staging",
    "staging": "staging",
    "dev": "development",
    "development": "development",
    "test": "test",
    "qa": "test",
}
_USAGE_TERMS = {"cache", "queue", "session", "analytics", "custom"}
_INSTANCE_HINTS = {"instance", "database", "db"}
_CLUSTER_HINTS = {"cluster", "subscription"}
_TYPE_HINTS = {
    "enterprise": "redis_enterprise",
    "cloud": "redis_cloud",
    "oss": "oss_single",
    "clustered": "oss_cluster",
}
_HEALTHY_STATUSES = {"healthy", "ok", "active", "available", "connected"}

TargetBinding = PublicTargetBinding
ResolvedTargetMatch = PublicTargetMatch
TargetResolutionResult = DiscoveryResponse


class TargetCatalogDoc(BaseModel):
    """用于 target discovery 的安全目录文档。"""

    target_id: str
    target_kind: str
    resource_id: str
    display_name: str
    name: str
    environment: Optional[str] = None
    status: Optional[str] = None
    target_type: Optional[str] = None
    usage: Optional[str] = None
    description: Optional[str] = None
    notes: Optional[str] = None
    repo_url: Optional[str] = None
    repo_slug: Optional[str] = None
    monitoring_identifier: Optional[str] = None
    logging_identifier: Optional[str] = None
    cluster_id: Optional[str] = None
    redis_cloud_subscription_id: Optional[str] = None
    redis_cloud_database_id: Optional[str] = None
    redis_cloud_database_name: Optional[str] = None
    search_text: str = ""
    search_aliases: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: Optional[str] = None


class ThreadTargetState(BaseModel):
    """线程 target 状态的阶段三兼容插槽。"""

    attached_target_handles: List[str] = Field(default_factory=list)
    active_target_handle: Optional[str] = None
    target_toolset_generation: int = 0
    target_bindings: List[TargetBinding] = Field(default_factory=list)


class MaterializedTargetScope(BaseModel):
    """一次 target 绑定后得到的上下文更新。"""

    selected_bindings: List[TargetBinding] = Field(default_factory=list)
    attached_bindings: List[TargetBinding] = Field(default_factory=list)
    target_toolset_generation: int = 0
    context_updates: Dict[str, Any] = Field(default_factory=dict)


class BoundTargetScope(BaseModel):
    """ToolManager 动态绑定后的统一返回值。"""

    bindings: List[TargetBinding] = Field(default_factory=list)
    toolset_generation: int = 0
    context_updates: Dict[str, Any] = Field(default_factory=dict)


async def resolve_target_query(
    *,
    query: str,
    user_id: Optional[str] = None,
    allow_multiple: bool = False,
    max_results: int = 5,
    preferred_capabilities: Optional[Sequence[str]] = None,
) -> TargetResolutionResult:
    service = TargetDiscoveryService()
    return await service.resolve(
        DiscoveryRequest(
            query=query,
            allow_multiple=allow_multiple,
            max_results=max_results,
            preferred_capabilities=list(preferred_capabilities or []),
            user_id=user_id,
        )
    )


async def list_known_targets(
    *,
    user_id: Optional[str] = None,
    target_kind: Optional[str] = None,
    environment: Optional[str] = None,
    capability: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    include_aliases: bool = False,
) -> Dict[str, Any]:
    docs = await get_target_catalog(user_id=user_id)
    normalized_kind = _normalize(target_kind)
    normalized_environment = _normalize_environment(environment)
    normalized_capability = _normalize(capability)

    filtered: List[TargetCatalogDoc] = []
    for doc in docs:
        if normalized_kind and _normalize(doc.target_kind) != normalized_kind:
            continue
        if normalized_environment and _normalize_environment(doc.environment) != normalized_environment:
            continue
        if normalized_capability:
            supported = {_normalize(item) for item in doc.capabilities}
            if normalized_capability not in supported:
                continue
        filtered.append(doc)

    total = len(filtered)
    bounded_limit = max(1, min(int(limit), 100))
    bounded_offset = max(0, int(offset))
    page = filtered[bounded_offset : bounded_offset + bounded_limit]
    return {
        "status": "ok",
        "total_known_targets": total,
        "returned_targets": len(page),
        "offset": bounded_offset,
        "limit": bounded_limit,
        "has_more": (bounded_offset + len(page)) < total,
        "targets": [
            build_public_target_inventory_entry(doc, include_aliases=include_aliases)
            for doc in page
        ],
    }


async def get_target_catalog(*, user_id: Optional[str] = None) -> List[TargetCatalogDoc]:
    """从阶段二资源层读取 Redis target 安全目录。"""

    docs = build_target_catalog_docs(await get_instances(), await get_clusters())
    filtered = [
        doc for doc in docs if not user_id or doc.user_id in {None, "", user_id}
    ]
    filtered.sort(key=lambda doc: doc.updated_at, reverse=True)
    return filtered


def build_target_catalog_docs(
    instances: Sequence[RedisInstance],
    clusters: Sequence[RedisCluster],
) -> List[TargetCatalogDoc]:
    docs = [build_target_doc_from_instance(instance) for instance in instances]
    docs.extend(build_target_doc_from_cluster(cluster) for cluster in clusters)
    return docs


def build_target_doc_from_instance(instance: RedisInstance) -> TargetCatalogDoc:
    """从 RedisInstance 构建不含 secret 的 target 文档。"""

    repo_slug, repo_tokens = _safe_repo_tokens(instance.repo_url)
    aliases = _extract_safe_aliases(instance.extension_data)
    cloud_subscription_id = (
        str(instance.redis_cloud_subscription_id)
        if instance.redis_cloud_subscription_id is not None
        else None
    )
    cloud_database_id = (
        str(instance.redis_cloud_database_id)
        if instance.redis_cloud_database_id is not None
        else None
    )
    safe_bits = _dedupe(
        [
            instance.name,
            instance.environment,
            instance.usage,
            instance.description,
            instance.notes,
            instance.monitoring_identifier,
            instance.logging_identifier,
            instance.redis_cloud_database_name,
            repo_slug,
            *repo_tokens,
            *aliases,
        ]
    )
    instance_type = (
        instance.instance_type.value
        if hasattr(instance.instance_type, "value")
        else str(instance.instance_type)
    )
    return TargetCatalogDoc(
        target_id=f"instance:{instance.id}",
        target_kind="instance",
        resource_id=instance.id,
        display_name=instance.name,
        name=instance.name,
        environment=_normalize_environment(instance.environment),
        status=_normalize(instance.status),
        target_type=_normalize(instance_type),
        usage=_normalize(instance.usage),
        description=instance.description,
        notes=instance.notes,
        repo_url=instance.repo_url,
        repo_slug=repo_slug,
        monitoring_identifier=instance.monitoring_identifier,
        logging_identifier=instance.logging_identifier,
        cluster_id=instance.cluster_id,
        redis_cloud_subscription_id=cloud_subscription_id,
        redis_cloud_database_id=cloud_database_id,
        redis_cloud_database_name=instance.redis_cloud_database_name,
        search_text=" ".join(safe_bits),
        search_aliases=aliases,
        capabilities=_instance_capabilities(instance),
        updated_at=instance.updated_at,
        created_at=instance.created_at,
        user_id=instance.user_id,
    )


def build_target_doc_from_cluster(cluster: RedisCluster) -> TargetCatalogDoc:
    """从 RedisCluster 构建不含 secret 的 target 文档。"""

    aliases = _extract_safe_aliases(cluster.extension_data)
    cluster_type = (
        cluster.cluster_type.value if hasattr(cluster.cluster_type, "value") else str(cluster.cluster_type)
    )
    safe_bits = _dedupe(
        [cluster.name, cluster.environment, cluster.description, cluster.notes, *aliases]
    )
    return TargetCatalogDoc(
        target_id=f"cluster:{cluster.id}",
        target_kind="cluster",
        resource_id=cluster.id,
        display_name=cluster.name,
        name=cluster.name,
        environment=_normalize_environment(cluster.environment),
        status=_normalize(cluster.status),
        target_type=_normalize(cluster_type),
        description=cluster.description,
        notes=cluster.notes,
        search_text=" ".join(safe_bits),
        search_aliases=aliases,
        capabilities=_cluster_capabilities(cluster),
        updated_at=cluster.updated_at,
        created_at=cluster.created_at,
        user_id=cluster.user_id,
    )


def build_public_target_inventory_entry(
    doc: TargetCatalogDoc,
    *,
    include_aliases: bool = False,
) -> Dict[str, Any]:
    public_metadata: Dict[str, Any] = {
        key: value
        for key, value in {"usage": doc.usage, "status": doc.status}.items()
        if value not in (None, "")
    }
    if include_aliases and doc.search_aliases:
        public_metadata["aliases"] = list(doc.search_aliases)
    return {
        "display_name": doc.display_name,
        "target_kind": doc.target_kind,
        "environment": doc.environment,
        "target_type": doc.target_type,
        "capabilities": list(doc.capabilities or []),
        "public_metadata": public_metadata,
    }


async def build_seed_hint_candidates(
    *,
    bindings: Optional[Sequence[TargetBinding]] = None,
    instance_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
) -> List[DiscoveryCandidate]:
    """从显式 instance_id/cluster_id 构造候选项，给 CLI 和测试留兼容入口。"""

    if instance_id and cluster_id:
        raise ValueError("Please provide only one of instance_id or cluster_id")

    candidates: List[DiscoveryCandidate] = []
    seen: set[tuple[str, str]] = set()

    async def _append_candidate(
        *,
        target_kind: str,
        binding_subject: str,
        binding: Optional[TargetBinding] = None,
    ) -> None:
        subject = str(binding_subject or "").strip()
        kind = str(target_kind or "").strip().lower()
        if not subject or kind not in {"instance", "cluster"}:
            return
        identity = (kind, subject)
        if identity in seen:
            return
        seen.add(identity)

        doc: Optional[TargetCatalogDoc] = None
        if kind == "instance":
            instance = await get_instance_by_id(subject)
            if instance is not None:
                doc = build_target_doc_from_instance(instance)
        elif kind == "cluster":
            cluster = await get_cluster_by_id(subject)
            if cluster is not None:
                doc = build_target_doc_from_cluster(cluster)

        if doc is not None:
            public_match = build_public_match_from_doc(doc, match_reasons=[f"matched {kind}_id"])
        else:
            public_match = PublicTargetMatch(
                target_kind=kind,
                display_name=(binding.display_name if binding else None) or subject,
                capabilities=list((binding.capabilities if binding else None) or []),
                confidence=1.0,
                match_reasons=[f"matched {kind}_id"],
                public_metadata=dict((binding.public_metadata if binding else None) or {}),
                resource_id=subject,
                score=100.0,
            )

        candidates.append(DiscoveryCandidate.from_public_match(public_match))

    for binding in bindings or []:
        await _append_candidate(
            target_kind=binding.target_kind,
            binding_subject=binding.resource_id or "",
            binding=binding,
        )
    if instance_id:
        await _append_candidate(target_kind="instance", binding_subject=instance_id)
    if cluster_id:
        await _append_candidate(target_kind="cluster", binding_subject=cluster_id)
    return candidates


def build_public_match_from_doc(
    doc: TargetCatalogDoc,
    *,
    match_reasons: Optional[Sequence[str]] = None,
    score: float = 100.0,
    confidence: float = 1.0,
) -> PublicTargetMatch:
    return PublicTargetMatch(
        target_kind=doc.target_kind,
        display_name=doc.display_name,
        environment=doc.environment,
        target_type=doc.target_type,
        capabilities=list(doc.capabilities or []),
        confidence=confidence,
        match_reasons=list(match_reasons or []),
        public_metadata={
            key: value
            for key, value in {"usage": doc.usage, "status": doc.status}.items()
            if value not in (None, "")
        },
        resource_id=doc.resource_id,
        score=score,
    )


def _score_target_doc(
    query: str,
    doc: TargetCatalogDoc,
    *,
    preferred_capabilities: Optional[Sequence[str]] = None,
    hints: Optional[Dict[str, Any]] = None,
) -> tuple[float, List[str]]:
    hints = hints or _parse_query_hints(query)
    normalized = hints["normalized"]
    query_tokens = hints["tokens"]
    hostname_terms = set(hints.get("hostname_terms") or [])
    reasons: List[str] = []
    score = 0.0

    exact_target_mentioned = _query_mentions_exact_target(doc, hints)
    if hostname_terms:
        exact_terms = _exact_target_terms(doc)
        exact_hostname_matches = sorted(hostname_terms & exact_terms)
        if not exact_hostname_matches and not exact_target_mentioned:
            return 0.0, []
        if exact_hostname_matches:
            score += 8.5
            reasons.append(f"matched hostname={exact_hostname_matches[0]}")

    if exact_target_mentioned:
        score += 7.5
        reasons.append("matched exact target reference")
    if normalized and normalized in {_normalize(doc.display_name), _normalize(doc.name)}:
        score += 8.0
        reasons.append("matched exact target name")

    exact_alias_matches = [alias for alias in doc.search_aliases if _normalize(alias) == normalized]
    if exact_alias_matches:
        score += 7.0
        reasons.append(f"matched alias={exact_alias_matches[0]}")

    name_tokens = set(_tokenize(doc.display_name)) | set(_tokenize(doc.name))
    alias_tokens = set()
    for alias in doc.search_aliases:
        alias_tokens.update(_tokenize(alias))
    hint_tokens = (
        set(hints["environments"])
        | set(hints["usages"])
        | set(hints["preferred_kinds"])
        | {token for target_type in hints["target_types"] for token in _tokenize(target_type)}
    )
    token_overlap = sorted(
        (query_tokens - hint_tokens)
        & (name_tokens | alias_tokens | set(_tokenize(doc.search_text)))
    )
    if token_overlap:
        score += min(4.0, len(token_overlap) * 0.9)
        reasons.append(f"matched tokens={','.join(token_overlap[:4])}")

    normalized_environment = _normalize_environment(doc.environment)
    if normalized_environment and normalized_environment in hints["environments"]:
        score += 3.0
        reasons.append(f"matched environment={normalized_environment}")
    normalized_usage = _normalize(doc.usage)
    if normalized_usage and normalized_usage in hints["usages"]:
        score += 2.5
        reasons.append(f"matched usage={normalized_usage}")
    if hints["preferred_kinds"]:
        if doc.target_kind in hints["preferred_kinds"]:
            score += 2.0
            reasons.append(f"matched kind={doc.target_kind}")
        else:
            score -= 0.5
    if hints["target_types"] and _normalize(doc.target_type) in hints["target_types"]:
        score += 2.0
        reasons.append(f"matched type={doc.target_type}")
    if preferred_capabilities:
        preferred = {_normalize(capability) for capability in preferred_capabilities if capability}
        supported = {_normalize(capability) for capability in doc.capabilities}
        matched = sorted(preferred & supported)
        if matched:
            score += min(2.0, len(matched) * 0.75)
            reasons.append(f"matched capabilities={','.join(matched[:3])}")
    if _normalize(doc.status) in _HEALTHY_STATUSES:
        score += 0.2
    return score, reasons


def _parse_query_hints(query: str) -> Dict[str, Any]:
    normalized = _normalize(query)
    token_list = _tokenize(normalized)
    tokens = set(token_list)
    environments = {_ENV_ALIASES[token] for token in tokens if token in _ENV_ALIASES}
    usages = {token for token in tokens if token in _USAGE_TERMS}
    preferred_kinds = set()
    if tokens & _INSTANCE_HINTS:
        preferred_kinds.add("instance")
    if tokens & _CLUSTER_HINTS:
        preferred_kinds.add("cluster")
    target_types = {_TYPE_HINTS[token] for token in tokens if token in _TYPE_HINTS}
    hostname_terms = {_normalize(term) for term in _HOSTNAME_RE.findall(normalized)}
    return {
        "normalized": normalized,
        "tokens": tokens,
        "token_list": token_list,
        "environments": environments,
        "usages": usages,
        "preferred_kinds": preferred_kinds,
        "target_types": target_types,
        "hostname_terms": hostname_terms,
    }


def _query_mentions_exact_target(doc: TargetCatalogDoc, hints: Dict[str, Any]) -> bool:
    normalized = hints["normalized"]
    token_list = hints["token_list"]
    for term in _exact_target_terms(doc):
        if normalized == term:
            return True
        term_tokens = _tokenize(term)
        if term_tokens and _contains_token_sequence(token_list, term_tokens):
            return True
    return False


def _exact_target_terms(doc: TargetCatalogDoc) -> set[str]:
    terms = {
        _normalize(doc.display_name),
        _normalize(doc.name),
        _normalize(doc.monitoring_identifier),
        _normalize(doc.logging_identifier),
        _normalize(doc.redis_cloud_database_name),
        _normalize(doc.repo_slug),
    }
    terms.update(_normalize(alias) for alias in doc.search_aliases)
    return {term for term in terms if term}


def _contains_token_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    last_start = len(haystack) - len(needle)
    for idx in range(last_start + 1):
        if list(haystack[idx : idx + len(needle)]) == list(needle):
            return True
    return False


def _confidence_from_score(score: float) -> float:
    if score <= 0:
        return 0.0
    if score >= 10:
        return 0.99
    return round(min(0.99, 0.45 + (score / 20.0)), 2)


def _safe_repo_tokens(repo_url: Optional[str]) -> tuple[Optional[str], List[str]]:
    if not repo_url:
        return None, []
    try:
        parsed = urlparse(repo_url)
        path_bits = [bit for bit in parsed.path.strip("/").split("/") if bit]
        if len(path_bits) >= 2:
            slug = "/".join(path_bits[-2:])
        elif path_bits:
            slug = path_bits[-1]
        else:
            slug = None
        tokens = _tokenize(slug or "")
        return slug, tokens
    except Exception:
        return None, []


def _extract_safe_aliases(extension_data: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(extension_data, dict):
        return []
    aliases: List[Any] = []
    raw_aliases = extension_data.get("aliases")
    if isinstance(raw_aliases, list):
        aliases.extend(raw_aliases)
    target_discovery = extension_data.get("target_discovery")
    if isinstance(target_discovery, dict):
        nested_aliases = target_discovery.get("aliases")
        if isinstance(nested_aliases, list):
            aliases.extend(nested_aliases)
    return _dedupe(aliases)


def _instance_capabilities(instance: RedisInstance) -> List[str]:
    capabilities = ["redis", "diagnostics", "metrics", "logs"]
    instance_type = _normalize(
        instance.instance_type.value
        if hasattr(instance.instance_type, "value")
        else instance.instance_type
    )
    if instance_type == "redis_enterprise":
        capabilities.append("admin")
    if instance_type == "redis_cloud":
        capabilities.append("cloud")
    return _dedupe(capabilities)


def _cluster_capabilities(cluster: RedisCluster) -> List[str]:
    cluster_type = _normalize(
        cluster.cluster_type.value if hasattr(cluster.cluster_type, "value") else cluster.cluster_type
    )
    capabilities = ["redis", "diagnostics", "metrics", "logs"]
    if cluster_type == "redis_enterprise":
        capabilities.append("admin")
    if cluster_type == "redis_cloud":
        capabilities.append("cloud")
    return _dedupe(capabilities)


def _normalize_environment(value: Any) -> str:
    normalized = _normalize(value)
    return _ENV_ALIASES.get(normalized, normalized)


def _tokenize(value: Any) -> List[str]:
    return _TOKEN_RE.findall(_normalize(value))


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _dedupe(values: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


async def bind_target_matches(
    *,
    matches: Sequence[DiscoveryCandidate],
    thread_id: Optional[str] = None,
    task_id: Optional[str] = None,
    replace_existing: bool = False,
    manager: Optional[Any] = None,
) -> BoundTargetScope:
    materialized = await materialize_bound_target_scope(
        matches=matches,
        thread_id=thread_id,
        task_id=task_id,
        replace_existing=replace_existing,
    )
    attached_bindings = list(materialized.attached_bindings)
    generation = materialized.target_toolset_generation
    if manager and attached_bindings:
        await manager.attach_bound_targets(attached_bindings, generation=generation)
        updated_generation = manager.get_toolset_generation()
        if inspect.isawaitable(updated_generation):
            updated_generation = await updated_generation
        generation = int(updated_generation)
    return BoundTargetScope(
        bindings=attached_bindings,
        toolset_generation=generation,
        context_updates=build_bound_target_scope_context(
            attached_bindings,
            generation=generation,
            active_handle=materialized.context_updates.get("active_target_handle"),
        ),
    )


async def attach_target_matches(
    *,
    thread_id: str,
    matches: Sequence[DiscoveryCandidate],
    task_id: Optional[str] = None,
    replace_existing: bool = False,
) -> tuple[List[TargetBinding], int]:
    """把 target matches 附着到 thread 的阶段三兼容入口。"""
    materialized = await materialize_bound_target_scope(
        matches=matches,
        thread_id=thread_id,
        task_id=task_id,
        replace_existing=replace_existing,
    )
    return materialized.attached_bindings, materialized.target_toolset_generation


async def materialize_bound_target_scope(
    *,
    matches: Sequence[DiscoveryCandidate],
    thread_id: Optional[str] = None,
    task_id: Optional[str] = None,
    replace_existing: bool = False,
) -> MaterializedTargetScope:
    binding_service = TargetBindingService()
    selected_bindings = await binding_service.build_and_persist_records(
        matches,
        thread_id=thread_id,
        task_id=task_id,
    )
    attached_bindings = list(selected_bindings)
    generation = 1 if selected_bindings else 0
    active_handle = selected_bindings[0].target_handle if selected_bindings else None
    context_updates = build_bound_target_scope_context(
        attached_bindings,
        generation=generation,
        active_handle=active_handle,
    )
    if thread_id:
        try:
            from redis_sre_agent.core.threads import ThreadManager

            await ThreadManager().update_thread_context(thread_id, context_updates)
        except Exception:
            pass
    return MaterializedTargetScope(
        selected_bindings=list(selected_bindings),
        attached_bindings=attached_bindings,
        target_toolset_generation=generation,
        context_updates=context_updates,
    )


def build_bound_target_scope_context(
    bindings: Sequence[TargetBinding],
    *,
    generation: int,
    active_handle: Optional[str] = None,
) -> Dict[str, Any]:
    attached_bindings = list(bindings)
    attached_handles = [binding.target_handle for binding in attached_bindings]
    resolved_active_handle = active_handle or (
        attached_bindings[0].target_handle if attached_bindings else ""
    )
    return {
        "attached_target_handles": attached_handles,
        "active_target_handle": resolved_active_handle or "",
        "target_toolset_generation": generation,
        "target_bindings": [binding.public_dump() for binding in attached_bindings],
        "instance_id": "",
        "cluster_id": "",
    }


async def get_thread_target_state(thread_id: str) -> ThreadTargetState:
    """从 thread context 读取已绑定 target 状态。"""

    try:
        from redis_sre_agent.core.threads import ThreadManager

        thread = await ThreadManager().get_thread(thread_id)
    except Exception:
        thread = None
    if thread is None:
        return ThreadTargetState()
    return ThreadTargetState(
        attached_target_handles=get_attached_target_handles_from_context(thread.context),
        active_target_handle=str(thread.context.get("active_target_handle") or "") or None,
        target_toolset_generation=int(thread.context.get("target_toolset_generation") or 0),
        target_bindings=get_target_bindings_from_context(thread.context),
    )


def get_attached_target_handles_from_context(context: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(context, dict):
        return []
    raw_handles = context.get("attached_target_handles") or []
    if not isinstance(raw_handles, list):
        return []
    return [str(handle).strip() for handle in raw_handles if str(handle or "").strip()]


def get_target_bindings_from_context(context: Optional[Dict[str, Any]]) -> List[TargetBinding]:
    if not isinstance(context, dict):
        return []
    raw_bindings = context.get("target_bindings") or []
    if not isinstance(raw_bindings, list):
        return []
    bindings: List[TargetBinding] = []
    for raw_binding in raw_bindings:
        try:
            bindings.append(TargetBinding.model_validate(raw_binding))
        except Exception:
            continue
    return bindings


async def resolve_binding_subject(binding: Optional[TargetBinding]) -> Optional[str]:
    """从 public binding 摘要解析 private subject 的阶段三兼容入口。"""
    if binding is None:
        return None
    return binding.resource_id
