"""Redis 集群模型和存储函数。

集群描述的是 Redis Enterprise、Redis Cloud 或 OSS cluster 这一层资源。实例可以通过
`cluster_id` 关联到集群。和实例一样，集群的管理密码等敏感字段写入 Redis 前必须加密。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, SecretStr, field_serializer, field_validator, model_validator

from redis_sre_agent.core.encryption import encrypt_secret, get_secret_value
from redis_sre_agent.core.keys import RedisKeys
from redis_sre_agent.core.redis import get_clusters_index, get_redis_client
from redis_sre_agent.core.redisearch import CountQuery, FilterQuery, Tag, tag_contains_expression

logger = logging.getLogger(__name__)


class RedisClusterType(str, Enum):
    """Redis 集群类型。"""

    oss_cluster = "oss_cluster"
    redis_enterprise = "redis_enterprise"
    redis_cloud = "redis_cloud"
    unknown = "unknown"


class RedisCluster(BaseModel):
    """Redis 集群配置模型。"""

    id: str
    name: str
    cluster_type: RedisClusterType = RedisClusterType.unknown
    environment: str = Field(..., description="环境，例如 development/staging/production/test。")
    description: str
    notes: Optional[str] = None
    admin_url: Optional[str] = None
    admin_username: Optional[str] = None
    admin_password: Optional[SecretStr] = None
    status: Optional[str] = "unknown"
    version: Optional[str] = None
    last_checked: Optional[str] = None
    extension_data: Optional[Dict[str, Any]] = None
    extension_secrets: Optional[Dict[str, SecretStr]] = None
    created_by: str = Field(default="user", description="只能是 user 或 agent。")
    user_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_serializer("admin_password", when_used="json")
    def dump_secret(self, value: Any) -> Any:
        #SecretStr 是 Pydantic 提供的“密码字符串”类型。它的特点是：平时打印时会隐藏真实内容为******
        #该函数作用是在导出 JSON 的那一刻，把 SecretStr 里的真实密码取出来
        if value is None:
            return None
        return value.get_secret_value() if isinstance(value, SecretStr) else value

    @field_serializer("extension_secrets", when_used="json")
    def dump_secret_dict(self, value: Optional[Dict[str, SecretStr]]) -> Optional[Dict[str, str]]:
        #和上个一样，只不过这个处理的是一组密钥
        if value is None:
            return None
        return {
            key: item.get_secret_value() if isinstance(item, SecretStr) else str(item)
            for key, item in value.items()
        }

    @field_validator("name")# 处理RedisCluster类的name字段时，自动触发此函数
    @classmethod #专门用于创建对象之前。如果不用cls，用的是self，做数据校验时，可能该实例还没出生，但用cls，就相当于在创建之前就校验name了，因为这是对该类通用的
    def validate_name(cls, value: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError("集群名称不能为空。")
        return normalized

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        normalized = (value or "").strip().lower()
        if normalized not in allowed:
            raise ValueError("environment 必须是 development、staging、production 或 test。")
        return normalized

    @field_validator("created_by")
    @classmethod
    def validate_created_by(cls, value: str) -> str:
        if value not in {"user", "agent"}:
            raise ValueError("created_by 必须是 user 或 agent。")
        return value

    # 企业版必须凭证都在，开源版必须凭证都空
    @model_validator(mode="after")
    def validate_enterprise_admin_fields(self) -> "RedisCluster":
        has_url = bool((self.admin_url or "").strip())
        has_username = bool((self.admin_username or "").strip())
        has_password = bool(self.admin_password)

        if self.cluster_type == RedisClusterType.redis_enterprise:
            if not (has_url and has_username and has_password):
                raise ValueError("redis_enterprise 集群必须提供 admin_url、admin_username、admin_password。")
            return self

        if has_url or has_username or has_password:
            raise ValueError("admin_url/admin_username/admin_password 只适用于 redis_enterprise 集群。")
        return self


@dataclass
class ClusterQueryResult:
    """集群查询结果。"""

    clusters: List[RedisCluster]
    total: int
    limit: int
    offset: int

# 将各种格式的“时间字符串”标准化转换为“浮点数时间戳”：把 ISO 时间、Z 后缀时间或数字字符串转换为 Unix 时间戳。
def _to_epoch(timestamp: Optional[str]) -> float:
    if not timestamp:
        return 0.0
    try:
        if timestamp.endswith("Z"):
            timestamp = timestamp.replace("Z", "+00:00")
        return datetime.fromisoformat(timestamp).timestamp()
    except Exception:
        try:
            return float(timestamp)
        except Exception:
            return 0.0

#把密钥字典里的每个 value 加密，保留 key 不变，用于写入 Redis 前保护 extension_secrets。
def _encrypt_secret_dict(data: Optional[dict[str, Any]]) -> Optional[dict[str, str]]:
    if data is None:
        return None
    return {key: encrypt_secret(str(value)) for key, value in data.items()}

#把 Redis 里读出的加密密钥字典解密，并重新包成 SecretStr(*****)，避免运行时打印泄露敏感值
def _decrypt_secret_dict(data: Optional[dict[str, Any]]) -> Optional[dict[str, SecretStr]]:
    if data is None:
        return None
    return {key: SecretStr(get_secret_value(str(value))) for key, value in data.items()}

#把 RedisCluster 对象转成可存储的 dict，并在写入前加密 admin_password 和 extension_secrets。
def _cluster_to_storage_dict(cluster: RedisCluster) -> dict[str, Any]:
    data = cluster.model_dump(mode="json")
    if data.get("admin_password"):
        data["admin_password"] = encrypt_secret(data["admin_password"])
    if data.get("extension_secrets"):
        data["extension_secrets"] = _encrypt_secret_dict(data["extension_secrets"])
    return data

#把 Redis 里读出的 dict 还原成 RedisCluster 对象，同时解密其中的敏感字段。
def _cluster_from_storage_dict(data: dict[str, Any]) -> RedisCluster:
    restored = dict(data)
    if restored.get("admin_password"):
        restored["admin_password"] = get_secret_value(restored["admin_password"])
    if restored.get("extension_secrets"):
        restored["extension_secrets"] = _decrypt_secret_dict(restored["extension_secrets"])
    return RedisCluster(**restored)

#检查集群查询索引是否存在；没有就创建，保证后续能按字段检索 RedisCluster。
async def _ensure_clusters_index_exists() -> None:
    try:
        index = await get_clusters_index()
        if not await index.exists():
            await index.create()
    except Exception:
        return

# 从 Redis 索引中“全量拉取”所有已配置的集群数据，
# 并将其反序列化为内存中的 RedisCluster 对象列表。
async def _load_clusters_from_index() -> List[RedisCluster]:
    await _ensure_clusters_index_exists()
    index = await get_clusters_index()
    try:
        total = await index.query(CountQuery(filter_expression="*"))
    except Exception:
        total = 1000
    if not total:
        return []
    results = await index.query(
        FilterQuery(filter_expression="*", return_fields=["data"], num_results=int(total))
    )
    clusters: List[RedisCluster] = []
    for doc in results or []:
        try:
            raw = doc.get("data")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw:
                clusters.append(_cluster_from_storage_dict(json.loads(raw)))
        except Exception:
            logger.warning("跳过无法解析的集群文档。")
    return clusters

# 针对_load_clusters_from_index查询集群的两种策略，严格和容错模式
async def get_clusters_strict() -> List[RedisCluster]:
    return await _load_clusters_from_index()
#容错模式
async def get_clusters() -> List[RedisCluster]:
    try:
        return await _load_clusters_from_index()
    except Exception:
        logger.warning("读取集群列表失败。")
        return []

# 精准查询，不同于get_clusters:按条件在服务端过滤集群，并返回分页结果
async def query_clusters(
    *,
    environment: Optional[str] = None,
    status: Optional[str] = None,
    cluster_type: Optional[str] = None,
    user_id: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> ClusterQueryResult:
    try:
        await _ensure_clusters_index_exists()
        index = await get_clusters_index()
        filter_expr = None

        for field_name, value in (
            ("environment", environment.lower() if environment else None),
            ("status", status.lower() if status else None),
            ("cluster_type", cluster_type.lower() if cluster_type else None),
            ("user_id", user_id),
        ):
            if value:
                expr = Tag(field_name) == value
                filter_expr = expr if filter_expr is None else (filter_expr & expr)

        if search and search.strip():
            expr = tag_contains_expression("name", search.strip())
            filter_expr = expr if filter_expr is None else (filter_expr & expr)

        count_expr = filter_expr if filter_expr is not None else "*"
        try:
            total = int(await index.query(CountQuery(filter_expression=count_expr)))
        except Exception:
            total = 0
        if total == 0:
            return ClusterQueryResult([], 0, limit, offset)

        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        query = FilterQuery(return_fields=["data"], num_results=limit).sort_by(
            "updated_at", asc=False
        )
        if filter_expr is not None:
            query.set_filter(filter_expr)
        query.paging(offset, limit)
        results = await index.query(query)

        clusters: List[RedisCluster] = []
        for doc in results or []:
            try:
                raw = doc.get("data")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                if raw:
                    clusters.append(_cluster_from_storage_dict(json.loads(raw)))
            except Exception:
                logger.warning("跳过无法解析的集群查询结果。")
        return ClusterQueryResult(clusters, total, limit, offset)
    except Exception:
        logger.warning("查询集群失败。")
        return ClusterQueryResult([], 0, limit, offset)

# 将 RedisCluster 存至 Redis 持久化存储
async def _upsert_cluster_index_doc(cluster: RedisCluster) -> bool:
    """
    本函数作为“数据模型到存储层”的适配器，执行以下原子化步骤：
    1. 预检：确保 RediSearch 索引存在，保证查询引擎可用。
    2. 序列化：将 Pydantic 对象转换为 JSON，并对 admin_password 进行加密处理（安全隔离）。
    3. 规范化：将时间戳清洗为数值类型（优化范围查询性能），将 Enum 降级为字符串（降低存储耦合）。
    4. 持久化：通过 HSET 将数据与索引元数据写入以 cluster.id 为键的 Redis Hash。
    
    返回：
        bool：写入成功返回 True，否则返回 False。
    """
    try:
        await _ensure_clusters_index_exists()
        client = get_redis_client()
        key = RedisKeys.cluster_doc(cluster.id)
        data = _cluster_to_storage_dict(cluster)
        cluster_type = (
            cluster.cluster_type.value
            if isinstance(cluster.cluster_type, RedisClusterType)
            else str(cluster.cluster_type)
        )
        await client.hset(
            key,
            mapping={
                "name": cluster.name or "",
                "environment": (cluster.environment or "").lower(),
                "cluster_type": cluster_type,
                "user_id": cluster.user_id or "",
                "status": (cluster.status or "unknown").lower(),
                "created_at": _to_epoch(cluster.created_at),
                "updated_at": _to_epoch(cluster.updated_at)
                or datetime.now(timezone.utc).timestamp(),
                "data": json.dumps(data),
            },
        )
        return True
    except Exception:
        logger.warning("保存集群索引文档失败。")
        return False

#批量写入集群索引文档，任一失败则最终返回 False
async def _upsert_clusters_index_docs(clusters: List[RedisCluster]) -> bool:
    ok = True
    for cluster in clusters:
        ok = await _upsert_cluster_index_doc(cluster) and ok
    return ok

#按集群 ID 删除 Redis 中对应的集群索引文档
async def delete_cluster_index_doc(cluster_id: str) -> None:
    try:
        await get_redis_client().delete(RedisKeys.cluster_doc(cluster_id))
    except Exception:
        return

# 通过‘每个集群独立的 Hash 文档 + 搜索索引’的方式来持久化集群数据
# 该函数执行声明式的全量状态同步，自动把传入的新列表进行新增或覆盖更新。
# 同时，它会找出数据库中存在但新列表中缺失的差集，将多余的旧集群彻底物理删除
async def save_clusters(clusters: List[RedisCluster]) -> bool:
    try:
        client = get_redis_client()
        await _ensure_clusters_index_exists()
        if not await _upsert_clusters_index_docs(clusters):
            return False

        keep_ids = {cluster.id for cluster in clusters}
        stale_ids: list[str] = []
        try:
            index = await get_clusters_index()
            total = await index.query(CountQuery(filter_expression="*"))
            if total:
                results = await index.query(
                    FilterQuery(
                        filter_expression="*",
                        return_fields=["data"],
                        num_results=int(total),
                    )
                )
                for doc in results or []:
                    raw = doc.get("data")
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    if raw:
                        doc_id = json.loads(raw).get("id")
                        if doc_id and doc_id not in keep_ids:
                            stale_ids.append(doc_id)
        except Exception:
            pass

        for stale_id in stale_ids:
            await client.delete(RedisKeys.cluster_doc(stale_id))
        return True
    except Exception:
        logger.warning("保存集群列表失败。")
        return False

async def get_cluster_by_id(cluster_id: str) -> Optional[RedisCluster]:
    try:
        data = await get_redis_client().hget(RedisKeys.cluster_doc(cluster_id), "data")
        if not data:
            return None
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return _cluster_from_storage_dict(json.loads(data))
    except Exception:
        logger.warning("按 id 读取集群失败。")
        return None
        