"""Redis 实例模型和存储函数。

实例是被诊断的 Redis 数据库或 Redis 服务端点。资源层负责把实例保存成 Redis hash 文档，
同时把可查询字段放在 hash 顶层，把完整 JSON 放在 `data` 字段里。敏感字段在写入 `data`
前会被加密，读取时再还原为 Pydantic 的 SecretStr。

本文件既定义了“一个 Redis 实例信息长什么样”
又包揽了“怎么把这些信息安全地存进数据库、怎么再查出来”的所有底层逻辑。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, field_serializer, field_validator, model_validator

from redis_sre_agent.core.encryption import encrypt_secret, get_secret_value
from redis_sre_agent.core.keys import RedisKeys
from redis_sre_agent.core.redis import get_instances_index, get_redis_client
from redis_sre_agent.core.redisearch import CountQuery, FilterQuery, Tag, tag_contains_expression

logger = logging.getLogger(__name__)

# 把该变成*的变成*
def mask_redis_url(url: Any) -> str:

    try:
        if isinstance(url, SecretStr):
            url = url.get_secret_value()
        elif not isinstance(url, str):
            url = str(url)

        parsed = urlparse(url)
        if parsed.username or parsed.password:
            host = parsed.hostname or ""
            if parsed.port:
                host += f":{parsed.port}"
            masked_url = f"{parsed.scheme}://***:***@{host}{parsed.path}"
            if parsed.query:
                masked_url += f"?{parsed.query}"
            if parsed.fragment:
                masked_url += f"#{parsed.fragment}"
            return masked_url
        return url
    except Exception:
        logger.warning("遮蔽 Redis URL 失败，返回通用占位 URL。")
        return "redis://***:***@<host>:<port>"

#规定了一个 Redis 只能是单机版（oss_single）、集群版（oss_cluster）或者企业云版等几种固定状态
class RedisInstanceType(str, Enum):
    """Redis 实例类型。"""

    oss_single = "oss_single"
    oss_cluster = "oss_cluster"
    redis_enterprise = "redis_enterprise"
    redis_cloud = "redis_cloud"
    unknown = "unknown"

#规定了一个 Redis 实例必须包含 id、name、connection_url（连接地址）、environment等几十个属性。
#其中用了 SecretStr 来标记密码和 URL。这告诉系统：“这是敏感信息，打印日志的时候绝对不能明文显示出来”
class RedisInstance(BaseModel):
    """Redis 实例配置模型。"""

    id: str
    name: str
    connection_url: SecretStr = Field(..., description="Redis 连接 URL。")
    environment: str = Field(..., description="环境，例如 development/staging/production/test。")
    usage: str = Field(..., description="用途，例如 cache/session/queue/custom。")
    description: str
    repo_url: Optional[str] = None
    notes: Optional[str] = None
    monitoring_identifier: Optional[str] = None
    logging_identifier: Optional[str] = None
    instance_type: RedisInstanceType = RedisInstanceType.unknown
    admin_url: Optional[str] = None
    admin_username: Optional[str] = None
    admin_password: Optional[SecretStr] = None
    cluster_id: Optional[str] = None
    redis_cloud_subscription_id: Optional[int] = None
    redis_cloud_database_id: Optional[int] = None
    redis_cloud_subscription_type: Optional[str] = None
    redis_cloud_database_name: Optional[str] = None
    status: Optional[str] = "unknown"
    version: Optional[str] = None
    memory: Optional[str] = None
    connections: Optional[int] = None
    last_checked: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extension_data: Optional[Dict[str, Any]] = None
    extension_secrets: Optional[Dict[str, SecretStr]] = None
    created_by: str = Field(default="user", description="只能是 user 或 agent。")
    user_id: Optional[str] = None

    #处理一个密码
    @field_serializer("connection_url", "admin_password", when_used="json")
    def dump_secret(self, value: Any) -> Any:
        if value is None:
            return None
        return value.get_secret_value() if isinstance(value, SecretStr) else value
    #处理一个字典里的密码
    @field_serializer("extension_secrets", when_used="json")
    def dump_secret_dict(self, value: Optional[Dict[str, SecretStr]]) -> Optional[Dict[str, str]]:
        if value is None:
            return None
        return {
            key: item.get_secret_value() if isinstance(item, SecretStr) else str(item)
            for key, item in value.items()
        }
    #当有人尝试创建一个新的 Redis 实例记录时，它会死死盯着 created_by（创建者）这个字段。它规定，创建者只能是 "user"（真实人类用户）或者 "agent"（自动化程序）。如果你随便传个 "admin" 或者 "test" 进来，它会直接报错（raise ValueError），拒绝创建这个对象。
    @field_validator("created_by")
    @classmethod
    def validate_created_by(cls, value: str) -> str:
        if value not in {"user", "agent"}:
            raise ValueError("created_by 必须是 user 或 agent。")
        return value

    #这个函数在整个实例对象“拼装完成”后（mode="after"）进行最后一遍检查：校验格式等等
    @model_validator(mode="after")
    def normalize_admin_fields(self) -> "RedisInstance":
        if isinstance(self.cluster_id, str):
            self.cluster_id = self.cluster_id.strip() or None
        if isinstance(self.admin_url, str):
            self.admin_url = self.admin_url.strip() or None
        if isinstance(self.admin_username, str):
            self.admin_username = self.admin_username.strip() or None

        password = (
            self.admin_password.get_secret_value()
            if isinstance(self.admin_password, SecretStr)
            else None
        )
        has_any = bool(self.admin_url) or bool(self.admin_username) or bool(password)
        has_all = bool(self.admin_url) and bool(self.admin_username) and bool(password)
        if has_any and not has_all:
            raise ValueError("admin_url、admin_username、admin_password 必须同时提供。")
        return self

# 分页查询结果，包括查出的数据/实例列表instances和条数
@dataclass
class InstanceQueryResult:
    instances: List[RedisInstance]
    total: int
    limit: int
    offset: int


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

#加密解密：存进数据库之前，把密码变成乱码；从数据库读出来要用的时候，再把乱码还原成真密码。
def _encrypt_secret_dict(data: Optional[dict[str, Any]]) -> Optional[dict[str, str]]:
    if data is None:
        return None
    return {key: encrypt_secret(str(value)) for key, value in data.items()}
def _decrypt_secret_dict(data: Optional[dict[str, Any]]) -> Optional[dict[str, SecretStr]]:
    if data is None:
        return None
    return {key: SecretStr(get_secret_value(str(value))) for key, value in data.items()}

#把立体的对象“压扁”成一个字典（JSON）
# 并且在压扁的过程中，调用加密机，把密码字段替换成密文，准备存入数据库。
def _instance_to_storage_dict(instance: RedisInstance) -> dict[str, Any]:
    data = instance.model_dump(mode="json")
    if data.get("connection_url"):
        data["connection_url"] = encrypt_secret(data["connection_url"])
    if data.get("admin_password"):
        data["admin_password"] = encrypt_secret(data["admin_password"])
    if data.get("extension_secrets"):
        data["extension_secrets"] = _encrypt_secret_dict(data["extension_secrets"])
    return data

#上面反向操作。把从数据库里读出来的扁平数据，调用解密机还原密码，然后重新变成一个立体的 RedisInstance 对象，供业务代码使用。
def _instance_from_storage_dict(data: dict[str, Any]) -> RedisInstance:
    restored = dict(data)
    if restored.get("connection_url"):
        restored["connection_url"] = get_secret_value(restored["connection_url"])
    if restored.get("admin_password"):
        restored["admin_password"] = get_secret_value(restored["admin_password"])
    if restored.get("extension_secrets"):
        restored["extension_secrets"] = _decrypt_secret_dict(restored["extension_secrets"])
    return RedisInstance(**restored)

#确保用来搜索的 RediSearch 索引已经建好了，如果没有就现场建一个。
async def _ensure_instances_index_exists() -> None:
    try:
        index = await get_instances_index()
        if not await index.exists():
            await index.create()
    except Exception:
        return

# 跑到 Redis 数据库里，把所有存着的实例配置数据一次性全捞出来。
async def _load_instances_from_index() -> List[RedisInstance]:
    await _ensure_instances_index_exists()
    index = await get_instances_index()

    try:
        total = await index.query(CountQuery(filter_expression="*"))
    except Exception:
        total = 1000
    if not total:
        return []

    results = await index.query(
        FilterQuery(filter_expression="*", return_fields=["data"], num_results=int(total))
    )

    out: List[RedisInstance] = []
    for doc in results or []:
        try:
            raw = doc.get("data")
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            out.append(_instance_from_storage_dict(json.loads(raw)))
        except Exception:
            logger.warning("跳过无法解析的实例文档。")
    return out


async def get_instances_strict() -> List[RedisInstance]:
    return await _load_instances_from_index()

async def get_instances() -> List[RedisInstance]:
    try:
        return await _load_instances_from_index()
    except Exception:
        logger.warning("读取实例列表失败。")
        return []

#可以传入各种条件的搜索，调用redisearch里的语法糖(Tag 类的 __eq__ 魔法方法、FilterExpression 类的 __and__ 魔法方法···)
#把这些条件拼装成 RediSearch 能看懂的 @environment:{production}，然后去数据库里分页捞数据出来。
async def query_instances(
    *,
    environment: Optional[str] = None,
    usage: Optional[str] = None,
    status: Optional[str] = None,
    instance_type: Optional[str] = None,
    user_id: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> InstanceQueryResult:
    try:
        await _ensure_instances_index_exists()
        index = await get_instances_index()
        filter_expr = None

        for field_name, value in (
            ("environment", environment.lower() if environment else None),
            ("usage", usage.lower() if usage else None),
            ("status", status.lower() if status else None),
            ("instance_type", instance_type.lower() if instance_type else None),
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
            return InstanceQueryResult([], 0, limit, offset)

        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        query = FilterQuery(return_fields=["data"], num_results=limit).sort_by(
            "updated_at", asc=False
        )
        if filter_expr is not None:
            query.set_filter(filter_expr)
        query.paging(offset, limit)
        results = await index.query(query)

        instances: List[RedisInstance] = []
        for doc in results or []:
            try:
                raw = doc.get("data")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                if raw:
                    instances.append(_instance_from_storage_dict(json.loads(raw)))
            except Exception:
                logger.warning("跳过无法解析的实例查询结果。")
        return InstanceQueryResult(instances, total, limit, offset)
    except Exception:
        logger.warning("查询实例失败。")
        return InstanceQueryResult([], 0, limit, offset)

# 保存数据的核心逻辑。它会把前面准备好的 JSON 数据存进 Redis 的 Hash 结构里
# 并且把诸如 name、environment 提取到外面作为可搜索的字段。
async def _upsert_instance_index_doc(instance: RedisInstance) -> bool:
    try:
        await _ensure_instances_index_exists()
        client = get_redis_client()
        key = RedisKeys.instance_doc(instance.id)
        data = _instance_to_storage_dict(instance)
        instance_type = (
            instance.instance_type.value
            if isinstance(instance.instance_type, RedisInstanceType)
            else str(instance.instance_type)
        )

        await client.hset(
            key,
            mapping={
                "name": instance.name or "",
                "environment": (instance.environment or "").lower(),
                "usage": (instance.usage or "").lower(),
                "instance_type": instance_type,
                "cluster_id": instance.cluster_id or "",
                "user_id": instance.user_id or "",
                "status": (instance.status or "unknown").lower(),
                "created_at": _to_epoch(instance.created_at),
                "updated_at": _to_epoch(instance.updated_at)
                or datetime.now(timezone.utc).timestamp(),
                "data": json.dumps(data),
            },
        )
        return True
    except Exception:
        logger.warning("保存实例索引文档失败。")
        return False
# 批量保存
async def _upsert_instances_index_docs(instances: List[RedisInstance]) -> bool:
    ok = True
    for instance in instances:
        ok = await _upsert_instance_index_doc(instance) and ok
    return ok

# 给它一个具体的实例 ID（比如 redis-prod-xxx），它会根据这个 ID 拼出底层 Redis 的键名（Key），然后直接发指令给 Redis 客户端：“把这个键对应的数据彻底删掉”。
async def delete_instance_index_doc(instance_id: str) -> None:
    try:
        await get_redis_client().delete(RedisKeys.instance_doc(instance_id))
    except Exception:
        return


async def save_instances(instances: List[RedisInstance]) -> bool:
    try:
        client = get_redis_client()
        await _ensure_instances_index_exists()
        if not await _upsert_instances_index_docs(instances):
            return False

        keep_ids = {instance.id for instance in instances}
        stale_ids: list[str] = []
        try:
            index = await get_instances_index()
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
            await client.delete(RedisKeys.instance_doc(stale_id))
        return True
    except Exception:
        logger.warning("保存实例列表失败。")
        return False


async def create_instance(
    *,
    name: str,
    connection_url: str,
    environment: str,
    usage: str,
    description: str,
    created_by: str = "agent",
    user_id: Optional[str] = None,
    repo_url: Optional[str] = None,
    notes: Optional[str] = None,
    instance_type: RedisInstanceType = RedisInstanceType.unknown,
) -> RedisInstance:
    instances = await get_instances()
    if any(instance.name == name for instance in instances):
        raise ValueError("同名 Redis 实例已经存在。")

    instance = RedisInstance(
        id=f"redis-{environment}-{uuid.uuid4().hex[:12]}",
        name=name,
        connection_url=connection_url,
        environment=environment,
        usage=usage,
        description=description,
        repo_url=repo_url,
        notes=notes,
        created_by=created_by,
        user_id=user_id,
        instance_type=instance_type,
    )
    if not await save_instances([*instances, instance]):
        raise ValueError("保存 Redis 实例失败。")
    return instance


async def get_instance_by_id(instance_id: str) -> Optional[RedisInstance]:
    try:
        data = await get_redis_client().hget(RedisKeys.instance_doc(instance_id), "data")
        if not data:
            return None
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return _instance_from_storage_dict(json.loads(data))
    except Exception:
        logger.warning("按 id 读取实例失败。")
        return None


async def get_instance_by_name(instance_name: str) -> Optional[RedisInstance]:
    result = await query_instances(search=instance_name, limit=1)
    for instance in result.instances:
        if instance.name == instance_name:
            return instance
    return result.instances[0] if result.instances else None

#调用全量查询把所有数据捞出来（拿到的是一个 List）。然后，它把这个列表转换成一个字典结构
async def get_instance_map() -> Dict[str, RedisInstance]:
    return {instance.id: instance for instance in await get_instances()}


async def get_instance_name(instance_id: str) -> Optional[str]:
    instance = await get_instance_by_id(instance_id)
    return instance.name if instance else None
