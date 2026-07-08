"""
专门用来读取、存储和管理.contracts的TargetHandleRecord而用
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Iterable, Optional

from redis_sre_agent.core.redis import get_redis_client

from .contracts import TargetHandleRecord

logger = logging.getLogger(__name__)

_TARGET_HANDLE_STORE_PREFIX = "sre_target_handles"
_DEFAULT_TTL_SECONDS = 24 * 60 * 60
_DEFAULT_STORE: "RedisTargetHandleStore | None" = None
_DEFAULT_STORE_LOCK = Lock()

# 专门用来管“敏感档案”(TargetHandleRecord)的管理员
class RedisTargetHandleStore:

    def __init__(self, *, key_prefix: str = _TARGET_HANDLE_STORE_PREFIX):
        self.key_prefix = key_prefix

    #给该对象存进来的每份档案生成一个专属key
    def _key(self, target_handle: str) -> str:
        return f"{self.key_prefix}:{target_handle}"

    # 计算该档案的距离销毁还有多久
    @staticmethod
    def _ttl_seconds(record: TargetHandleRecord) -> int:
        expires_at = record.expires_at
        if not expires_at:
            return _DEFAULT_TTL_SECONDS
        try:
            value = expires_at.replace("Z", "+00:00") if expires_at.endswith("Z") else expires_at
            return int(
                max(
                    1,
                    (datetime.fromisoformat(value) - datetime.now(timezone.utc)).total_seconds(),
                )
            )
        except Exception:
            return _DEFAULT_TTL_SECONDS

    # 把档案存成json字符串，然后加key并记个销毁时间
    async def save_record(self, record: TargetHandleRecord) -> None:
        client = get_redis_client()
        try:
            await client.set(
                self._key(record.target_handle),
                record.model_dump_json(),
                ex=self._ttl_seconds(record),
            )
        except Exception:
            logger.warning("Target handle store unavailable while saving %s", record.target_handle)

    async def save_records(self, records: Iterable[TargetHandleRecord]) -> None:
        record_list = list(records)
        if not record_list:
            return
        client = get_redis_client()
        try:
            async with client.pipeline(transaction=True) as pipe:
                for record in record_list:
                    pipe.set(
                        self._key(record.target_handle),
                        record.model_dump_json(),
                        ex=self._ttl_seconds(record),
                    )
                await pipe.execute()
        except Exception:
            logger.warning("Target handle store unavailable while saving %s records", len(record_list))

    async def get_record(self, target_handle: str) -> Optional[TargetHandleRecord]:
        client = get_redis_client()
        try:
            raw = await client.get(self._key(target_handle))
        except Exception:
            return None
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return TargetHandleRecord.model_validate_json(raw)
        except Exception:
            logger.warning("Invalid target handle record for %s", target_handle)
            return None

    async def get_records(self, target_handles: Iterable[str]) -> Dict[str, TargetHandleRecord]:
        handle_list = list(target_handles)
        if not handle_list:
            return {}
        client = get_redis_client()
        keys = [self._key(target_handle) for target_handle in handle_list]
        records: Dict[str, TargetHandleRecord] = {}
        try:
            raw_values = await client.mget(*keys)
        except Exception:
            return {}
        for target_handle, raw in zip(handle_list, raw_values or []):
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                records[target_handle] = TargetHandleRecord.model_validate_json(raw)
            except Exception:
                logger.warning("Invalid target handle record for %s", target_handle)
        return records

# 单例模式：整个系统只需要一个管理员RedisTargetHandleStore。系统任何地方需要存取档，就调用这个函数。
# 如果已经有了管理员，就直接让有的这个工作；如果没有，就创建一个。
def get_target_handle_store() -> RedisTargetHandleStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        with _DEFAULT_STORE_LOCK:
            if _DEFAULT_STORE is None:
                _DEFAULT_STORE = RedisTargetHandleStore()
    return _DEFAULT_STORE

# 销毁该管理员RedisTargetHandleStore
def reset_target_handle_store() -> None:
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        _DEFAULT_STORE = None
