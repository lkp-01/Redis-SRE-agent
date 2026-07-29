"""本地 Markdown 的顺序摄取实现，不包含 scraper、skills 或多索引逻辑。"""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from redis_sre_agent.core.config import Settings, settings
from redis_sre_agent.core.redis import RAGNotReadyError

from .deduplication import DocumentDeduplicator
from .document_processor import DocumentProcessor
from .pipeline_workflow_mixin import PipelineWorkflowMixin
from .processor_indexing_helpers import index_processed_document
from .processor_source_helpers import (
    create_scraped_document_from_markdown,
    find_markdown_files,
)

logger = logging.getLogger(__name__)


# 异步获取知识库索引的辅助函数
async def get_knowledge_index(config: Optional[Settings] = None):
    # 局部/延迟导入核心模块的函数，避免循环依赖，同时提升模块加载速度
    from redis_sre_agent.core.redis import get_knowledge_index as _get_knowledge_index
    # 调用并返回核心逻辑中的获取索引方法
    return await _get_knowledge_index(config=config)


# 异步确保知识库索引已就绪的辅助函数
async def ensure_knowledge_index(
        config: Optional[Settings] = None,
        *,
        create_if_missing: bool,  # 关键字传参，指示如果索引缺失是否强制创建
):
    # 同样使用局部导入
    from redis_sre_agent.core.redis import ensure_knowledge_index as _ensure
    # 调用并返回结果
    return await _ensure(config=config, create_if_missing=create_if_missing)


# 获取向量化器（模型）的辅助函数
def get_vectorizer(config: Optional[Settings] = None):
    # 局部导入向量化器工厂函数
    from redis_sre_agent.core.redis import get_vectorizer as _get_vectorizer
    # 返回对应的向量化器实例（此处为同步调用）
    return _get_vectorizer(config=config)

class IngestionPipeline(PipelineWorkflowMixin):
    """先显式确保索引，再逐个处理本地 Markdown 文档。"""

    # 类的初始化方法，用于配置摄取流水线的各类依赖
    def __init__(
            self,
            storage: Any = None,  # 存储介质实例，用于批量文件读写
            config: Optional[Dict[str, Any]] = None,  # 自定义配置字典
            knowledge_settings: Any = None,  # 知识库特定配置
            *,
            settings_config: Optional[Settings] = None,  # 全局系统设置（覆盖用）
            index: Any = None,  # 允许外部注入已创建的索引
            vectorizer: Any = None,  # 允许外部注入已实例化的向量化器
    ) -> None:
        self.storage = storage
        self.config = config or {}  # 若未提供 config，则初始化为空字典
        self.knowledge_settings = knowledge_settings
        self.settings = settings_config or settings  # 优先使用传入的设置，否则退化为全局的 settings

        # 实例化文档处理器，后续负责文本的分块(chunking)等操作
        self.processor = DocumentProcessor(self.config, knowledge_settings)
        self._index = index  # 内部缓存索引
        self._vectorizer = vectorizer  # 内部缓存向量化器

    # 私有方法：解析并准备运行时所需的依赖（索引与向量化器）
    async def _resolve_runtime(self) -> tuple[Any, Any]:
        # 安全检查：确认系统级别是否开启了 RAG（检索增强生成）
        if not self.settings.rag_enabled:
            raise RAGNotReadyError("disabled", "RAG 未启用。")

        index = self._index
        # 如果尚未获取到索引实例
        if index is None:
            # 确保索引存在，不存在则通知底层自动创建
            index = await ensure_knowledge_index(
                self.settings,
                create_if_missing=True,
            )

        vectorizer = self._vectorizer
        # 如果尚未获取到向量化器实例
        if vectorizer is None:
            # 依据当前配置获取默认的向量化模型
            vectorizer = get_vectorizer(self.settings)

        # 返回准备就绪的依赖元组
        return index, vectorizer

    # 核心方法 1：摄取原始 Markdown 目录文档
    async def ingest_source_documents(self, source_dir: Path | str) -> List[Dict[str, Any]]:
        source_path = Path(source_dir)  # 统一转换为 Path 对象
        # 校验传入目录是否合法（存在且确实是一个目录）
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError("source directory 不存在或不是目录。")

        # 获取预备好的索引实例和向量化模型
        index, vectorizer = await self._resolve_runtime()

        # 初始化文档去重器，依赖索引以查验内容是否已存在
        deduplicator = DocumentDeduplicator(
            index,
            key_prefix="sre_knowledge",  # Redis 存储时的键前缀
            vector_dim=self.settings.vector_dim,  # 当前使用的向量维度大小
        )

        results: List[Dict[str, Any]] = []  # 记录每篇文档的处理状态和统计

        # 递归或遍历目录下找寻所有的 Markdown 文件
        for markdown_file in find_markdown_files(source_path):
            # 提取文件相对路径的字符串形式，用于记录和展示
            relative_name = markdown_file.relative_to(source_path).as_posix()
            try:
                # 解析本地 Markdown 文件，构造成标准化的文档对象
                document = create_scraped_document_from_markdown(
                    markdown_file,
                    source_path,
                )
                # 利用 processor 将长文档切割成符合模型大小的 chunk（块）
                chunks = self.processor.chunk_document(document)

                # 执行最核心的入库动作：将文档内容、块进行向量化并存入 Redis
                indexed = await index_processed_document(
                    document=document,
                    chunks=chunks,
                    vectorizer=vectorizer,
                    deduplicator=deduplicator,
                )

                # 获取去重逻辑反馈的操作状态 (例如：新增、跳过、更新等)
                change = indexed["source_document_change"]

                # 记录该文档处理成功的详细流水
                results.append(
                    {
                        "file": relative_name,
                        "title": document.title,
                        "category": document.category.value,
                        "severity": document.severity.value,
                        "status": "success",
                        "action": change["action"],
                        "chunks_created": indexed["chunks_created"],
                        "chunks_indexed": indexed["chunks_indexed"],
                        "document_hash": document.document_hash,
                    }
                )
            except Exception as exc:
                # 外部 provider/Redis 异常文本可能带连接信息，用户输出只保留安全错误码。
                # 防止敏感内网信息外泄，仅在日志打印具体类型
                logger.warning("本地 Markdown 摄取失败：%s", type(exc).__name__)

                # 追加错误结果，对外部仅暴露宽泛的失败信息和异常大类
                results.append(
                    {
                        "file": relative_name,
                        "status": "error",
                        "error": "文档摄取失败。",
                        "error_type": type(exc).__name__,
                    }
                )
        # 返回批次中所有文档的状态集
        return results

    # 核心方法 2：摄取按时间批次划分的预处理 JSON 制品 (Batch artifacts)
    async def ingest_batch(self, batch_date: str) -> Dict[str, Any]:
        """顺序摄取一个本地 artifact batch，不做并行或 stale-scope 清理。"""

        # 若要读批次 json 数据，必须配置 storage 引擎
        if self.storage is None:
            raise ValueError("prepared batch 摄取需要 ArtifactStorage。")

        # 尝试查询该日期的清单配置
        manifest = self.storage.get_batch_manifest(batch_date)
        if not manifest:
            raise ValueError(f"No manifest found for batch {batch_date}")

        # 构建并检查该日期批次的对应物理路径是否存在
        batch_path = self.storage.base_path / batch_date
        if not batch_path.exists():
            raise ValueError(f"Batch directory not found: {batch_date}")

        # 在 storage 状态中设定当前操作批次
        self.storage.set_batch_date(batch_date)

        # 初始化整个批次的统计指标字典
        stats: Dict[str, Any] = {
            "batch_date": batch_date,
            "started_at": datetime.now(timezone.utc).isoformat(),  # 记录标准的 UTC 开始时间
            "documents_processed": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "categories_processed": {},  # 细分存放不同类别的进度
            "errors": [],
            "success": False,
        }

        # 同样需要获取运行时的索引和向量化器
        index, vectorizer = await self._resolve_runtime()

        # 初始化处理批次制品的去重器
        deduplicator = DocumentDeduplicator(
            index,
            key_prefix="sre_knowledge",
            vector_dim=self.settings.vector_dim,
        )

        # 遍历三大标准知识分类 (开源、企业内部、通用共享)
        for category in ("oss", "enterprise", "shared"):
            category_path = batch_path / category

            # 初始化该类别专属的数据统计板
            category_stats = {
                "documents_processed": 0,
                "chunks_created": 0,
                "chunks_indexed": 0,
                "errors": [],
            }

            # 如果对应类别的文件夹存在，开始处理
            if category_path.exists():
                # glob 寻找所有 json 文件，并进行字典排序以保证顺序一致性
                for artifact_path in sorted(category_path.glob("*.json")):
                    try:
                        # 读出 JSON 文件内容，并反序列化为 Python 对象
                        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                        if not isinstance(payload, dict):
                            raise ValueError("artifact 必须是 JSON object。")

                        # 局部导入实体模型
                        from redis_sre_agent.pipelines.scraper.base import ScrapedDocument

                        # 用字典里的数据重建标准的文档实体对象
                        document = ScrapedDocument.from_dict(payload)
                        # 执行文档切块
                        chunks = self.processor.chunk_document(document)

                        # 向量化并把结果压入 Redis
                        indexed = await index_processed_document(
                            document=document,
                            chunks=chunks,
                            vectorizer=vectorizer,
                            deduplicator=deduplicator,
                        )

                        # 成功入库后，更新该类别的各项统计累加值
                        category_stats["documents_processed"] += 1
                        category_stats["chunks_created"] += indexed["chunks_created"]
                        category_stats["chunks_indexed"] += indexed["chunks_indexed"]
                    except Exception as exc:
                        # 如遇异常，不中止整个大流程，记录在案即可
                        category_stats["errors"].append(
                            {
                                "file": artifact_path.name,
                                "error": "artifact 摄取失败。",
                                "error_type": type(exc).__name__,
                            }
                        )

            # 单一类别处理完后，将类别统计挂载到总统计表上
            stats["categories_processed"][category] = category_stats
            # 累加至全局总指标
            stats["documents_processed"] += category_stats["documents_processed"]
            stats["chunks_created"] += category_stats["chunks_created"]
            stats["chunks_indexed"] += category_stats["chunks_indexed"]
            stats["errors"].extend(category_stats["errors"])  # 合并所有错误列表

        # 全部类别执行完毕，记录结束时间
        stats["completed_at"] = datetime.now(timezone.utc).isoformat()
        # 根据错误列表是否为空，判定本次大批次摄取的最终状态是否成功
        stats["success"] = not stats["errors"]

        # 利用 storage 固化/存档该批次的摄取统计报告
        self.storage.save_ingestion_manifest(batch_date, stats)

        # 返回最终的总报表
        return stats