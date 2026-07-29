"""knowledge index 的最小纯向量检索 helper。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from redisvl.query import VectorQuery, VectorRangeQuery
from redisvl.query.filter import Tag

from redis_sre_agent.core.config import Settings, settings
from redis_sre_agent.core.redis import (
    _close_index_client,
    get_knowledge_index,
    get_rag_readiness,
    get_vectorizer,
)

_SEARCH_RETURN_FIELDS = [
    "id",
    "document_hash",
    "chunk_index",
    "title",
    "content",
    "source",
    "category",
    "doc_type",
    "name",
    "summary",
    "priority",
    "pinned",
    "version",
]


def _coerce_non_negative_int(value: Any, *, default: int) -> int:
    """强制将输入值转换为非负整数 (>= 0)。"""
    try:
        # 尝试将传入的值转换为整数 (int)
        # 然后使用 max() 比较，如果转换出的整数小于 0，则强制截断为 0
        return max(int(value), 0)
    except (TypeError, ValueError):
        # 如果传入的值无法转换为整数（例如传入了 None, 列表，或者 "abc"）
        # 捕获异常，并安全地返回调用者指定的默认值
        return default


def _coerce_positive_int(value: Any, *, default: int) -> int:
    """强制将输入值转换为正整数 (>= 1)。通常用于限制拉取数量(limit)。"""
    # 先调用上方的函数将值转为非负整数
    # 然后将其与 1 比较取最大值，确保结果至少为 1（不能为 0）
    return max(_coerce_non_negative_int(value, default=default), 1)


def _decode(value: Any) -> Any:
    """将 Redis 返回的 bytes 数据安全解码为 UTF-8 字符串。"""
    # 检查传入的 value 是否属于 bytes 类型
    # 如果是，使用 utf-8 解码，如果遇到无法解码的特殊字符，使用 "replace" 策略（通常替换为 ），防止抛出 UnicodeDecodeError
    # 如果不是 bytes（已经是 str 或数字），则原样返回
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _normalized_chunk_index(value: Any) -> Any:
    """清洗和规范化文档的 chunk_index。"""
    # 如果值为空（None 或者空字符串），说明该文档没有分块，直接返回 None
    if value in (None, ""):
        return None
    try:
        # 先对值进行 bytes 解码，然后尝试转换为整数格式的索引
        return int(_decode(value))
    except (TypeError, ValueError):
        # 如果转换整数失败（可能索引碰巧包含了字母如 "A-1"），
        # 则退一步，只返回解码后的字符串格式
        return _decode(value)


def _normalized_score(document: Dict[str, Any]) -> float:
    """从检索返回的文档字典中提取并格式化相似度分数。"""
    # 尝试提取名为 "score" 的字段
    value = document.get("score")
    # 如果没有找到 "score"，尝试找 "vector_distance"（Redis 默认常用的距离字段名）
    if value is None:
        value = document.get("vector_distance")
    # 如果还是没有找到，尝试找通用的 "distance" 字段
    if value is None:
        value = document.get("distance")

    try:
        # 如果找到了任意一个字段的值，先解码，然后强制转换为浮点数 (float) 返回
        # 如果经过三轮提取 value 仍然是 None，则在这一步之前判定并抛出给 except，或者直接走条件分支
        return float(_decode(value)) if value is not None else 0.0
    except (TypeError, ValueError):
        # 如果提取出的值是乱码、字符串或无法转换为 float 的类型，作为兜底返回 0.0
        return 0.0


def _unavailable_result(
    *,
    query: str,
    reason_code: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "reason_code": reason_code,
        "message": message,
        "query": query,
        "retrieval_kind": "knowledge_search",
        "retrieval_label": "Knowledge search",
        "results_count": 0,
        "results": [],
    }


async def search_knowledge_base_helper(
        query: str,  # 用户输入的自然语言搜索查询
        category: Optional[str] = None,  # 可选：按特定的知识分类过滤
        doc_type: Optional[str] = None,  # 可选：按特定的文档类型过滤
        limit: int = 10,  # 限制返回结果的最大数量，默认 10 条
        offset: int = 0,  # 分页查询的偏移量，默认从第 0 条开始
        distance_threshold: Optional[float] = 0.8,  # 向量匹配的距离阈值，默认 0.8（通常距离越小越相似）
        version: Optional[str] = "latest",  # 可选：按文档版本过滤，默认只查 "latest" 版本
        config: Optional[Settings] = None,  # 可选：系统配置对象，用于依赖注入
) -> Dict[str, Any]:
    """用 query embedding 在唯一 knowledge index 中执行向量检索。"""

    # --- 步骤 1：获取配置与系统就绪检查 ---
    cfg = config or settings
    # 检查系统的 RAG 基础服务（如 Redis 和模型服务）是否准备就绪
    readiness = await get_rag_readiness(cfg)
    if not readiness.ready:
        # 如果未就绪，提前返回预设的“服务不可用”结果字典
        return _unavailable_result(
            query=query,
            reason_code=readiness.reason_code,
            message=readiness.message,
        )

    # --- 步骤 2：参数规范化 (Sanitization) ---
    # 确保 limit 最小为 1，最大不超过 50，防止单次请求拉取过多数据导致性能问题
    normalized_limit = min(_coerce_positive_int(limit, default=10), 50)
    # 确保 offset 是一个非负整数
    normalized_offset = _coerce_non_negative_int(offset, default=0)
    # 计算 Redis 向量查询所需拉取的总边界（类似于 SQL 的 LIMIT offset, count）
    fetch_limit = normalized_limit + normalized_offset

    # --- 步骤 3：文本向量化 (Embedding) ---
    try:
        # 获取配置好的向量化工具模型
        vectorizer = get_vectorizer(cfg)
        # 异步调用模型，将文本查询 (query) 转换为浮点数向量
        vectors = await vectorizer.aembed_many([query])
        # 提取生成的向量（处理可能为空的情况）
        query_vector = vectors[0] if vectors else []
        # 校验生成的向量维度，是否与 Redis 索引配置中要求的维度严格一致
        if len(query_vector) != cfg.vector_dim:
            return _unavailable_result(
                query=query,
                reason_code="embedding_config_invalid",
                message="query embedding 维度与 knowledge index 不一致。",
            )
    except Exception:
        # 捕获调用 Embedding API 时的任何网络或服务异常
        return _unavailable_result(
            query=query,
            reason_code="embedding_unavailable",
            message="embedding provider 当前不可用。",
        )

    # --- 步骤 4：构建元数据过滤条件 (Metadata Filtering) ---
    filter_expression = None
    if version is not None:
        # 使用 redisvl 的 Tag 语法构建基于版本的过滤条件
        filter_expression = Tag("version") == version
    if category is not None:
        category_filter = Tag("category") == category
        # 使用位运算符 '&' (AND) 拼接过滤条件
        filter_expression = (
            category_filter
            if filter_expression is None
            else filter_expression & category_filter
        )
    if doc_type is not None:
        type_filter = Tag("doc_type") == doc_type
        # 继续叠加文档类型的过滤条件
        filter_expression = (
            type_filter if filter_expression is None else filter_expression & type_filter
        )

    # --- 步骤 5：组装 Redis 向量查询对象 ---
    query_kwargs = {
        "vector": query_vector,  # 上一步生成的查询向量
        "vector_field_name": "vector",  # Redis 中存储向量数据的字段名
        "return_fields": list(_SEARCH_RETURN_FIELDS),  # 指定查询成功后要返回的字段（减少不必要的 I/O）
        "filter_expression": filter_expression,  # 上一步拼接好的元数据过滤树
        "num_results": fetch_limit,  # 设置要检索的文档数量
    }

    # 根据是否设置了距离阈值，选择不同策略的查询模式
    if distance_threshold is None:
        # 如果没有阈值，执行基础的 KNN（K-Nearest Neighbors，K个最近邻）查询
        vector_query: Any = VectorQuery(**query_kwargs)
    else:
        # 如果有阈值，执行 Range Query（范围查询），过滤掉距离大于设定值的无关文档
        vector_query = VectorRangeQuery(
            **query_kwargs,
            distance_threshold=float(distance_threshold),
        )
    # 为查询对象附加分页参数
    vector_query.paging(normalized_offset, normalized_limit)

    # --- 步骤 6：连接 Redis 执行查询 ---
    index = None
    try:
        # 获取 RedisVL 索引客户端
        index = await get_knowledge_index(cfg)
        # 将组装好的 vector_query 发送给 Redis 执行，获取原始检索结果
        raw_results = await index.query(vector_query)
    except Exception:
        # 捕获 Redis 连接超时或查询语法错误等异常
        return _unavailable_result(
            query=query,
            reason_code="redis_search_unavailable",
            message="Redis Search/Vector 查询失败。",
        )
    finally:
        # 【关键安全与性能步骤】：无论查询成功与否，必须释放/关闭数据库连接
        if index is not None:
            await _close_index_client(index)

    # --- 步骤 7：解析和清洗返回的数据 ---
    results: List[Dict[str, Any]] = []
    for raw_document in raw_results or []:
        # Redis 返回的数据键值对经常是 bytes (字节流) 格式，这里统一解码为 UTF-8 字符串
        document = {
            str(_decode(key)): _decode(value)
            for key, value in dict(raw_document).items()
        }
        # 将清洗后的字段组装成规范化的标准字典，对于缺失的字段提供合理的默认值
        results.append(
            {
                "id": document.get("id", ""),
                "document_hash": document.get("document_hash", ""),
                "chunk_index": _normalized_chunk_index(document.get("chunk_index")),
                "title": document.get("title", ""),
                "content": document.get("content", ""),
                "source": document.get("source", ""),
                "category": document.get("category", ""),
                "doc_type": document.get("doc_type", "knowledge"),
                "name": document.get("name", ""),
                "summary": document.get("summary", ""),
                "priority": document.get("priority", "normal"),
                "pinned": document.get("pinned", "false"),
                "version": document.get("version", "latest"),
                # 提取相似度得分（通常包含 distance 或 score 字段）
                "score": _normalized_score(document),
            }
        )

    # --- 步骤 8：组装最终的 API 响应载荷 ---
    return {
        "status": "success",
        "reason_code": "ready",
        "query": query,
        "category": category,
        "doc_type": doc_type,
        "version": version,
        "offset": normalized_offset,
        "limit": normalized_limit,
        "distance_threshold": distance_threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),  # 记录检索发生的 UTC 时间戳
        "retrieval_kind": "knowledge_search",
        "retrieval_label": "Knowledge search",
        "results_count": len(results),  # 实际查找到的合法记录数
        "results": results,  # 返回清洗后文档详情列表
    }
