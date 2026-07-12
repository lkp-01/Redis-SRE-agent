"""
【核心定位】：整个系统的“底层数据库通讯基站”和“档案柜说明书”。

本文件不包含任何复杂的业务逻辑（比如怎么诊断问题、怎么生成报告），它只专注解决两件事：
1. 怎么连上 Redis 数据库？（造一根能通通往数据库的网络连接线）
2. 存进 Redis 里的数据，应该按什么格式建立“搜索目录”？（定义 RediSearch 的索引结构，也就是定义好“档案柜的分类标签”）

一大堆 SRE_XXX_INDEX 常量这些就是系统里所有“档案柜”的名字
SRE_INSTANCES_SCHEMA 和 SRE_CLUSTERS_SCHEMA（档案柜说明书）这里定义的是“怎样在柜子里建立搜索卡片”。
class LightweightSearchIndex:exist/query/create

get_redis_client（制造钥匙）
这是整个文件最重要的“造连接”工厂。
任何其他代码想操作数据库，不能自己瞎连，必须调用这个函数。它会去配置文件（settings）里把加密的连接地址（URL）拿出来，然后生成一个可以通信的 Redis 客户端对象交给你。

test_redis_connection（网络探针）这就是个“网络测线仪”。它利用上面的 get_redis_client 拿到客户端，然后发送一个 ping 信号给数据库。如果数据库回了 pong，说明网是通的

【第一性原理解释】：
如果把整个系统比作一家“大型图书馆”，那么：
- 其他模块负责借书、还书、处理读者投诉（业务逻辑）。
- 本模块仅仅负责：
    a. 拿到大门的钥匙，能推开图书馆数据库的大门（`get_redis_client` 和 `test_redis_connection`）。
    b. 规定好几个主要的图书检索柜长什么样，比如“实例柜”里必须有名字、环境、用途等标签卡片，方便以后检索（`SRE_INSTANCES_SCHEMA` 和 `SRE_CLUSTERS_SCHEMA`）。

"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from redis.asyncio import Redis

from redis_sre_agent.core.config import Settings, settings
from redis_sre_agent.core.redisearch import CountQuery, FilterQuery

logger = logging.getLogger(__name__)

SRE_KNOWLEDGE_INDEX = "sre_knowledge"
SRE_SKILLS_INDEX = "sre_skills"
SRE_SUPPORT_TICKETS_INDEX = "sre_support_tickets"
SRE_SCHEDULES_INDEX = "sre_schedules"
SRE_THREADS_INDEX = "sre_threads"
SRE_TASKS_INDEX = "sre_tasks"
SRE_INSTANCES_INDEX = "sre_instances"
SRE_CLUSTERS_INDEX = "sre_clusters"
SRE_TARGETS_INDEX = "sre_targets"

SRE_INSTANCES_SCHEMA = {
    "index": {
        "name": SRE_INSTANCES_INDEX,
        "prefix": f"{SRE_INSTANCES_INDEX}:",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "name", "type": "tag"},
        {"name": "environment", "type": "tag"},
        {"name": "usage", "type": "tag"},
        {"name": "instance_type", "type": "tag"},
        {"name": "cluster_id", "type": "tag"},
        {"name": "user_id", "type": "tag"},
        {"name": "status", "type": "tag"},
        {"name": "created_at", "type": "numeric"},
        {"name": "updated_at", "type": "numeric"},
    ],
}

SRE_CLUSTERS_SCHEMA = {
    "index": {
        "name": SRE_CLUSTERS_INDEX,
        "prefix": f"{SRE_CLUSTERS_INDEX}:",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "name", "type": "tag"},
        {"name": "environment", "type": "tag"},
        {"name": "cluster_type", "type": "tag"},
        {"name": "user_id", "type": "tag"},
        {"name": "status", "type": "tag"},
        {"name": "created_at", "type": "numeric"},
        {"name": "updated_at", "type": "numeric"},
    ],
}

SRE_TARGETS_SCHEMA = {
    "index": {
        "name": SRE_TARGETS_INDEX,
        "prefix": f"{SRE_TARGETS_INDEX}:",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "target_kind", "type": "tag"},
        {"name": "display_name", "type": "tag"},
        {"name": "name", "type": "tag"},
        {"name": "environment", "type": "tag"},
        {"name": "target_type", "type": "tag"},
        {"name": "search_text", "type": "text"},
        {"name": "user_id", "type": "tag"},
        {"name": "updated_at", "type": "numeric"},
    ],
}

SRE_THREADS_SCHEMA = {
    "index": {
        "name": SRE_THREADS_INDEX,
        "prefix": f"{SRE_THREADS_INDEX}:",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "subject", "type": "text"},
        {"name": "user_id", "type": "tag"},
        {"name": "instance_id", "type": "tag"},
        {"name": "priority", "type": "numeric"},
        {"name": "created_at", "type": "numeric"},
        {"name": "updated_at", "type": "numeric"},
        {"name": "tags", "type": "tag"},
    ],
}


class LightweightSearchIndex:
    """阶段二轻量索引对象。

    原项目使用 redisvl 的 AsyncSearchIndex。阶段二只需要保存 schema、暴露 exists/create/query
    这几个资源层入口，并允许测试用 fake 或 monkeypatch 替换。
    """

    #把外面传进来的索引配置和redis客户端绑到对象自己身上
    def __init__(self, schema: dict[str, Any], redis_client: Redis) -> None:
        self.schema = schema
        self._redis_client = redis_client
        self.name = schema["index"]["name"]

    # 检查 Redis 里某个搜索索引是否已经存在。
    # 向 Redis 发 FT.INFO 索引名，如果 Redis 能返回信息，说明索引存在。
    async def exists(self) -> bool:
        try:
            await self._redis_client.execute_command("FT.INFO", self.name)
            return True
        except Exception:
            return False

    async def create(self) -> None:
        """创建索引的轻量入口。

        完整 FT.CREATE 语句不是阶段二重点。真实部署后可以替换为 redisvl 或更完整的 schema
        同步逻辑；当前测试会用 fake index，不触达真实 Redis。
        """

        return None

    async def query(self, query: Any) -> list[dict[str, Any]] | int:
        """在没有 RediSearch module 时，对小型 Hash 目录执行兼容查询。

        这里只解释本项目 `core.redisearch` 生成的 CountQuery、FilterQuery 和 AND TAG
        表达式。使用 SCAN 而不是 KEYS，避免阻塞系统 Redis；完整全文/向量搜索仍不在本阶段。
        """

        prefix = str(self.schema["index"]["prefix"])
        documents: list[dict[str, Any]] = []
        async for raw_key in self._redis_client.scan_iter(match=f"{prefix}*"):
            raw_document = await self._redis_client.hgetall(raw_key)
            document = {
                self._decode(key): self._decode(value)
                for key, value in (raw_document or {}).items()
            }
            if self._matches_filter(document, getattr(query, "filter_expression", "*")):
                documents.append(document)

        if isinstance(query, CountQuery):
            return len(documents)
        if not isinstance(query, FilterQuery):
            raise TypeError(f"Unsupported lightweight index query: {type(query).__name__}")

        sort_field = query.sort_field
        if sort_field:
            numeric_fields = {
                field["name"]
                for field in self.schema.get("fields", [])
                if field.get("type") == "numeric"
            }

            def sort_value(document: dict[str, Any]) -> Any:
                value = document.get(sort_field, "")
                if sort_field in numeric_fields:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return float("-inf")
                return str(value).casefold()

            documents.sort(key=sort_value, reverse=not query.sort_asc)

        offset = max(0, int(query.offset))
        limit = max(0, int(query.limit))
        page = documents[offset : offset + limit]
        if not query.return_fields:
            return page
        return [
            {field: document.get(field) for field in query.return_fields}
            for document in page
        ]

    @staticmethod
    def _decode(value: Any) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    @staticmethod
    def _matches_filter(document: dict[str, Any], filter_expression: Any) -> bool:
        expression = str(filter_expression or "*").strip()
        if expression in {"", "*"}:
            return True

        clauses = re.findall(r"@([A-Za-z0-9_]+):\{((?:\\.|[^}])*)\}", expression)
        if not clauses:
            return False
        for field, raw_expected in clauses:
            contains = raw_expected.startswith("*") and raw_expected.endswith("*")
            expected = raw_expected[1:-1] if contains else raw_expected
            expected = re.sub(r"\\(.)", r"\1", expected).casefold()
            actual = str(document.get(field, "")).casefold()
            if contains:
                if expected not in actual:
                    return False
            elif actual != expected:
                return False
        return True

# 这个函数负责创建 Redis 客户端。上层代码想访问 Redis，必须先有一个客户端对象；这个函数就是统一的“造客户端入口”。
# 如果调用者传了 url，就用传入的地址；如果没传，就从配置 settings 里拿 redis_url。
# 它每次调用都会新建客户端，减少异步环境里复用旧连接带来的麻烦。
def get_redis_client(url: Optional[str] = None, config: Optional[Settings] = None) -> Redis:
    cfg = config or settings
    redis_url = url or cfg.redis_url.get_secret_value()
    return Redis.from_url(redis_url, decode_responses=False)

# 创建“Redis 实例索引”的轻量对象/实例，作为搜索入口提供给外界调用；
# 可通过其封装好的方法，操作底层 Redis 数据库中的真实索引，从而检索/管理各类 Redis 实例（服务器节点）的元数据。
async def get_instances_index(config: Optional[Settings] = None) -> LightweightSearchIndex:
    cfg = config or settings
    client = get_redis_client(url=cfg.redis_url.get_secret_value(), config=cfg)
    return LightweightSearchIndex(SRE_INSTANCES_SCHEMA, client)

# 负责创建“Redis 集群索引”的轻量对象
async def get_clusters_index(config: Optional[Settings] = None) -> LightweightSearchIndex:
    cfg = config or settings
    client = get_redis_client(url=cfg.redis_url.get_secret_value(), config=cfg)
    return LightweightSearchIndex(SRE_CLUSTERS_SCHEMA, client)


# 目标目录索引插槽。阶段三的 core.targets 默认直接从实例/集群资源层构建目录；
# 这个入口保留给后续需要持久化 target catalog 时复用。
async def get_targets_index(config: Optional[Settings] = None) -> LightweightSearchIndex:
    cfg = config or settings
    client = get_redis_client(url=cfg.redis_url.get_secret_value(), config=cfg)
    return LightweightSearchIndex(SRE_TARGETS_SCHEMA, client)


async def get_threads_index(config: Optional[Settings] = None) -> LightweightSearchIndex:
    """返回 Thread 搜索索引入口；本阶段不扩展其他索引行为。"""

    cfg = config or settings
    client = get_redis_client(url=cfg.redis_url.get_secret_value(), config=cfg)
    return LightweightSearchIndex(SRE_THREADS_SCHEMA, client)

#检查 Redis 是否能连通
async def test_redis_connection(
    url: Optional[str] = None,
    config: Optional[Settings] = None,
) -> bool:
    """检查 Redis 是否可 ping 通。

    测试里会 monkeypatch `get_redis_client`，所以不会访问真实 Redis。
    """

    client = None
    try:
        client = get_redis_client(url=url, config=config)
        await client.ping()
        return True
    except Exception:
        logger.warning("Redis 连接检查失败。")
        return False
    finally:
        if client is not None:
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result
