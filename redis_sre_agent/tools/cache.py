"""
系统里有很多“工具”（比如用来查询服务器状态、内存等信息的工具）。
如果短时间内，系统用相同的条件去查同一个东西，每次都去真实服务器上查就太慢、太费力了。
这个文件里的代码的作用就是：把第一次查到的结果存到一个叫 Redis 的缓存里。
如果一会又问了一模一样的问题，就直接从缓存里把之前的答案翻出来用。
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from redis.asyncio import Redis

from redis_sre_agent.core.config import settings

logger = logging.getLogger(__name__)

# 不同工具的查询结果在缓存里能存活多久
DEFAULT_TOOL_TTLS: Dict[str, int] = {
    "ping": 30,
    "info": 60,
    "memory_stats": 60,
    "config_get": 300,
    "slowlog": 60,
    "client_list": 30,
    "cluster_info": 60,
    "knowledge_search": 300,
}
CACHE_PREFIX = "sre_cache:tool"# 存到Redis里的数据加一个统一的前缀标签


class ToolCache:
    """按实例和工具参数缓存只读工具结果。"""

    def __init__(
        self,
        redis_client: Redis,
        instance_id: str,
        ttl_overrides: Optional[Dict[str, int]] = None,
        enabled: bool = True,
    ):
        self._redis = redis_client
        self._instance_id = instance_id
        self._ttl_overrides = ttl_overrides or {}
        self._enabled = enabled

    #给查出来的结果加标签
    def build_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        args_json = json.dumps(args, sort_keys=True, default=str)
        args_hash = hashlib.sha256(args_json.encode()).hexdigest()[:16]
        return f"{CACHE_PREFIX}:{self._instance_id}:{tool_name}:{args_hash}"

    #查这个答案应该多久过期
    def _get_ttl(self, tool_name: str) -> int:
        for key, ttl in self._ttl_overrides.items():
            if key in tool_name.lower():
                return ttl
        for key, ttl in DEFAULT_TOOL_TTLS.items():
            if key in tool_name.lower():
                return ttl
        return settings.tool_cache_default_ttl

    #检查工具返回的结果，不正常就不要存
    @staticmethod
    def _should_cache(result: Any) -> bool:
        if isinstance(result, dict):
            status = str(result.get("status", "")).lower()
            if status in {"error", "failed", "failure"}:
                return False
        return True

    #查答案(根据build_key)
    async def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[Any]:
        if not self._enabled:
            return None
        try:
            data = await self._redis.get(self.build_key(tool_name, args))
            if not data:
                return None
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return json.loads(data)
        except Exception as exc:
            logger.warning("Cache get failed: %s", exc)
            return None

    # 存答案(也根据build_key)
    async def set(self, tool_name: str, args: Dict[str, Any], result: Any) -> bool:
        if not self._enabled or not self._should_cache(result):
            return False
        try:
            await self._redis.setex(
                self.build_key(tool_name, args),
                self._get_ttl(tool_name),
                json.dumps(result, default=str),
            )
            return True
        except Exception as exc:
            logger.warning("Cache set failed: %s", exc)
            return False
