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
from dataclasses import dataclass
from typing import Any, Optional

from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize import OpenAITextVectorizer as OpenAITextVectorizer

from redis_sre_agent.core.config import Settings, settings
from redis_sre_agent.core.redisearch import CountQuery, FilterQuery
from redis_sre_agent.core.vectorizer_helpers import (
    Vectorizer,
    create_vectorizer,
    validate_embedding_config,
)

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


def _build_document_schema(
    index_name: str,
    include_pinned: bool,
    *,
    vector_dim: Optional[int] = None,
) -> dict[str, Any]:
    """按 original 字段和 RedisVL 形状构造单一 knowledge hash 索引。"""

    fields: list[dict[str, Any]] = [
        {"name": "id", "type": "tag"},
        {"name": "title", "type": "text"},
        {"name": "content", "type": "text"},
        {"name": "content_hash", "type": "tag"},
        {"name": "document_hash", "type": "tag"},
        {"name": "source", "type": "tag"},
        {"name": "category", "type": "tag"},
        {"name": "doc_type", "type": "tag"},
        {"name": "name", "type": "tag"},
        {"name": "summary", "type": "text"},
        {"name": "priority", "type": "tag"},
        {"name": "severity", "type": "tag"},
        {"name": "product_labels", "type": "tag"},
        {"name": "product_label_tags", "type": "tag"},
        {"name": "version", "type": "tag"},
        {"name": "chunk_index", "type": "numeric"},
        {"name": "created_at", "type": "numeric"},
        {
            "name": "vector",
            "type": "vector",
            "attrs": {
                "dims": int(vector_dim if vector_dim is not None else settings.vector_dim),
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            },
        },
    ]
    if include_pinned:
        chunk_position = next(
            (i for i, field in enumerate(fields) if field["name"] == "chunk_index"),
            len(fields),
        )
        fields.insert(chunk_position, {"name": "pinned", "type": "tag"})
    return {
        "index": {
            "name": index_name,
            "prefix": f"{index_name}:",
            "storage_type": "hash",
        },
        "fields": fields,
    }


SRE_KNOWLEDGE_SCHEMA = _build_document_schema(
    SRE_KNOWLEDGE_INDEX,
    include_pinned=True,
    vector_dim=settings.vector_dim,
)

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


@dataclass(frozen=True)
class RAGReadiness:
    """可安全展示的 RAG 三态结果；message 中不放底层连接异常。"""

    state: str
    reason_code: str
    message: str

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason_code": self.reason_code,
            "message": self.message,
            "ready": self.ready,
        }


class RAGNotReadyError(RuntimeError):
    """显式管理入口和检索入口共享的安全错误。"""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


_RAG_MESSAGES = {
    "disabled": "RAG 未启用。",
    "embedding_config_invalid": "embedding 配置无效或不完整。",
    "redis_search_unavailable": "Redis Search/Vector 能力不可用。",
    "index_missing": "knowledge index 尚未创建；请通过显式摄取或索引管理入口创建。",
    "schema_mismatch": "knowledge index schema 与当前 embedding 维度或字段契约不一致。",
    "ready": "RAG 已就绪。",
}


def _decode_redis_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_decode_redis_value(item) for item in value]
    return value


def _pairs_to_dict(values: list[Any]) -> dict[str, Any]:
    normalized = _decode_redis_value(values)
    return {
        str(normalized[index]): normalized[index + 1]
        for index in range(0, len(normalized) - 1, 2)
    }


def _expected_field_definitions(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for field in schema.get("fields", []):
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        field_type = str(field.get("type") or "").upper()
        definition: dict[str, Any] = {"type": field_type}
        if field_type == "VECTOR":
            attrs = field.get("attrs") or {}
            definition["attrs"] = {
                "algorithm": str(attrs.get("algorithm") or "").upper(),
                "data_type": str(attrs.get("datatype") or "").upper(),
                "dim": int(attrs.get("dims")),
                "distance_metric": str(attrs.get("distance_metric") or "").upper(),
            }
        definitions[name] = definition
    return definitions


def _actual_field_definitions(raw_info: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw_info, dict):
        info = {
            str(_decode_redis_value(key)): _decode_redis_value(value)
            for key, value in raw_info.items()
        }
    else:
        info = _pairs_to_dict(list(raw_info or []))
    definitions: dict[str, dict[str, Any]] = {}
    for raw_attribute in info.get("attributes") or []:
        if isinstance(raw_attribute, dict):
            attribute = {
                str(_decode_redis_value(key)): _decode_redis_value(value)
                for key, value in raw_attribute.items()
            }
        else:
            attribute = _pairs_to_dict(list(raw_attribute or []))
        name = str(attribute.get("attribute") or attribute.get("identifier") or "").strip()
        if not name:
            continue
        field_type = str(attribute.get("type") or "").upper()
        definition: dict[str, Any] = {"type": field_type}
        if field_type == "VECTOR":
            try:
                dim: Any = int(attribute.get("dim"))
            except (TypeError, ValueError):
                dim = attribute.get("dim")
            definition["attrs"] = {
                "algorithm": str(attribute.get("algorithm") or "").upper(),
                "data_type": str(attribute.get("data_type") or "").upper(),
                "dim": dim,
                "distance_metric": str(attribute.get("distance_metric") or "").upper(),
            }
        definitions[name] = definition
    return definitions


def _knowledge_schema_matches(expected: dict[str, Any], raw_info: Any) -> bool:
    expected_fields = _expected_field_definitions(expected)
    actual_fields = _actual_field_definitions(raw_info)
    return expected_fields == actual_fields


def _get_index_client(index: Any) -> Any:
    client = getattr(index, "_redis_client", None) or getattr(index, "client", None)
    if client is None:
        raise RuntimeError("knowledge index 没有可用 Redis client。")
    return client


async def _close_index_client(index: Any) -> None:
    try:
        client = _get_index_client(index)
    except Exception:
        return
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close is None:
        return
    try:
        result = close()
        if hasattr(result, "__await__"):
            await result
    except Exception:
        logger.debug("关闭 knowledge readiness client 失败。", exc_info=True)


def _command_info_has_all(command_info: Any, expected_count: int) -> bool:
    if isinstance(command_info, dict):
        values = list(command_info.values())
        return len(values) >= expected_count and all(value for value in values[:expected_count])
    if isinstance(command_info, (list, tuple)):
        return len(command_info) >= expected_count and all(
            value is not None and value is not False for value in command_info[:expected_count]
        )
    return False


async def _redis_search_available(client: Any) -> bool:
    try:
        command_info = await client.execute_command(
            "COMMAND",
            "INFO",
            "FT.SEARCH",
            "FT.CREATE",
        )
    except Exception:
        return False
    return _command_info_has_all(command_info, 2)


def get_vectorizer(config: Optional[Settings] = None) -> Vectorizer:
    """返回只使用 embedding 配置的向量器。"""

    return create_vectorizer(config=config)


async def get_knowledge_index(config: Optional[Settings] = None) -> AsyncSearchIndex:
    """只构造 knowledge index；这里绝不执行 FT.CREATE。"""

    cfg = config or settings
    schema_dict = _build_document_schema(
        SRE_KNOWLEDGE_INDEX,
        include_pinned=True,
        vector_dim=cfg.vector_dim,
    )
    client = get_redis_client(config=cfg)
    return AsyncSearchIndex(
        schema=IndexSchema.from_dict(schema_dict),
        redis_client=client,
    )


async def ensure_knowledge_index(
    config: Optional[Settings] = None,
    create_if_missing: bool = False,
) -> AsyncSearchIndex:
    """检查 knowledge index；只有显式参数为真时才允许创建。"""

    cfg = config or settings
    if not cfg.rag_enabled:
        raise RAGNotReadyError("disabled", _RAG_MESSAGES["disabled"])
    try:
        validate_embedding_config(cfg)
    except Exception as exc:
        raise RAGNotReadyError(
            "embedding_config_invalid",
            _RAG_MESSAGES["embedding_config_invalid"],
        ) from exc

    index: Any = None
    try:
        index = await get_knowledge_index(cfg)
        client = _get_index_client(index)
        if not await _redis_search_available(client):
            raise RAGNotReadyError(
                "redis_search_unavailable",
                _RAG_MESSAGES["redis_search_unavailable"],
            )

        exists = bool(await index.exists())
        if not exists:
            if not create_if_missing:
                raise RAGNotReadyError("index_missing", _RAG_MESSAGES["index_missing"])
            try:
                await index.create()
            except Exception as exc:
                raise RAGNotReadyError(
                    "redis_search_unavailable",
                    _RAG_MESSAGES["redis_search_unavailable"],
                ) from exc
            if not await index.exists():
                raise RAGNotReadyError("index_missing", _RAG_MESSAGES["index_missing"])

        expected_schema = _build_document_schema(
            SRE_KNOWLEDGE_INDEX,
            include_pinned=True,
            vector_dim=cfg.vector_dim,
        )
        try:
            raw_info = await client.execute_command("FT.INFO", SRE_KNOWLEDGE_INDEX)
        except Exception as exc:
            raise RAGNotReadyError(
                "redis_search_unavailable",
                _RAG_MESSAGES["redis_search_unavailable"],
            ) from exc
        if not _knowledge_schema_matches(expected_schema, raw_info):
            raise RAGNotReadyError("schema_mismatch", _RAG_MESSAGES["schema_mismatch"])

        try:
            await client.execute_command(
                "FT.SEARCH",
                SRE_KNOWLEDGE_INDEX,
                "*",
                "LIMIT",
                0,
                0,
            )
        except Exception as exc:
            raise RAGNotReadyError(
                "redis_search_unavailable",
                _RAG_MESSAGES["redis_search_unavailable"],
            ) from exc
        return index
    except RAGNotReadyError:
        if index is not None:
            await _close_index_client(index)
        raise
    except Exception as exc:
        if index is not None:
            await _close_index_client(index)
        raise RAGNotReadyError(
            "redis_search_unavailable",
            _RAG_MESSAGES["redis_search_unavailable"],
        ) from exc


async def get_rag_readiness(config: Optional[Settings] = None) -> RAGReadiness:
    """普通读取路径使用的三态检查；不会创建或修改索引。"""

    cfg = config or settings
    if not cfg.rag_enabled:
        return RAGReadiness("disabled", "disabled", _RAG_MESSAGES["disabled"])
    try:
        validate_embedding_config(cfg)
    except Exception:
        return RAGReadiness(
            "not_ready",
            "embedding_config_invalid",
            _RAG_MESSAGES["embedding_config_invalid"],
        )

    try:
        index = await ensure_knowledge_index(cfg, create_if_missing=False)
    except RAGNotReadyError as exc:
        return RAGReadiness("not_ready", exc.reason_code, str(exc))
    await _close_index_client(index)
    return RAGReadiness("ready", "ready", _RAG_MESSAGES["ready"])

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
