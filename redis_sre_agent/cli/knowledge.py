"""knowledge base 的只读向量搜索命令。"""

from __future__ import annotations

import asyncio
from typing import Optional

import click

from redis_sre_agent.core.knowledge_helpers import search_knowledge_base_helper


@click.group()
def knowledge() -> None:
    """查询已经就绪的 Redis SRE knowledge index。"""


@knowledge.command("search")
@click.argument("query", nargs=-1, required=True)
@click.option("--category", "-c", type=str, help="按文档 category 过滤。")
@click.option("--limit", "-l", default=10, show_default=True, type=int)
@click.option("--offset", "-o", default=0, show_default=True, type=int)
@click.option("--distance-threshold", "-d", type=float)
@click.option("--version", "-v", type=str, default="latest", show_default=True)
def knowledge_search(
    query: tuple[str, ...],
    category: Optional[str],
    limit: int,
    offset: int,
    distance_threshold: Optional[float],
    version: Optional[str],
) -> None:
    """对本地已摄取文档执行纯向量检索。"""

    query_text = " ".join(query).strip()

    async def _run() -> None:
        result = await search_knowledge_base_helper(
            query=query_text,
            category=category,
            limit=limit,
            offset=offset,
            distance_threshold=distance_threshold,
            version=version,
        )
        if result.get("status") != "success":
            reason_code = str(result.get("reason_code") or "rag_unavailable")
            message = str(result.get("message") or "RAG 当前不可用。")
            raise click.ClickException(f"RAG 未就绪（{reason_code}）：{message}")
        documents = result.get("results") or []
        if not documents:
            click.echo("检索成功，但没有找到匹配文档。")
            return
        click.echo(f"找到 {len(documents)} 条 knowledge 结果：")
        for index, document in enumerate(documents, 1):
            content = str(document.get("content") or "").strip()
            preview = content if len(content) <= 240 else f"{content[:237]}..."
            click.echo(f"[{index}] {document.get('title') or '-'}")
            click.echo(f"  source={document.get('source') or '-'}")
            click.echo(
                "  document_hash="
                f"{document.get('document_hash') or '-'} "
                f"chunk={document.get('chunk_index')} "
                f"score={document.get('score')}"
            )
            click.echo(f"  {preview}")

    asyncio.run(_run())
