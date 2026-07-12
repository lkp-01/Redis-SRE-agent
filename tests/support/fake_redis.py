"""覆盖 ThreadManager Redis 命令面的轻量 fake。"""

from __future__ import annotations

from collections import defaultdict
from fnmatch import fnmatch
from typing import Any


class FakePipeline:
    def __init__(self, redis: "FakeRedis") -> None:
        self.redis = redis
        self.commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def __aenter__(self) -> "FakePipeline":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __getattr__(self, name: str):
        def queue(*args: Any, **kwargs: Any) -> "FakePipeline":
            self.commands.append((name, args, kwargs))
            return self

        return queue

    async def execute(self) -> list[Any]:
        results = []
        for name, args, kwargs in self.commands:
            results.append(await getattr(self.redis, name)(*args, **kwargs))
        return results


class FakeRedis:
    """在一个对象内模拟 Redis List/Hash/String 与 TTL。"""

    def __init__(self) -> None:
        self.strings: dict[str, Any] = {}
        self.lists: dict[str, list[Any]] = defaultdict(list)
        self.hashes: dict[str, dict[Any, Any]] = defaultdict(dict)
        self.expirations: dict[str, int] = {}

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)

    async def exists(self, key: str) -> int:
        return int(key in self.strings or key in self.lists or key in self.hashes)

    async def type(self, key: str) -> bytes:
        if key in self.lists:
            return b"list"
        if key in self.hashes:
            return b"hash"
        if key in self.strings:
            return b"string"
        return b"none"

    async def rpush(self, key: str, *values: Any) -> int:
        self.lists[key].extend(values)
        return len(self.lists[key])

    async def lrange(self, key: str, start: int, end: int) -> list[Any]:
        values = self.lists.get(key, [])
        return list(values[start:] if end == -1 else values[start : end + 1])

    async def hset(
        self,
        key: str,
        field: Any = None,
        value: Any = None,
        mapping: dict[Any, Any] | None = None,
    ) -> int:
        target = self.hashes[key]
        updates = dict(mapping or {})
        if field is not None:
            updates[field] = value
        target.update(updates)
        return len(updates)

    async def hgetall(self, key: str) -> dict[Any, Any]:
        return dict(self.hashes.get(key, {}))

    async def scan_iter(self, match: str = "*"):
        keys = set(self.strings) | set(self.lists) | set(self.hashes)
        for key in sorted(keys):
            if fnmatch(str(key), match):
                yield key

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            existed = key in self.strings or key in self.lists or key in self.hashes
            self.strings.pop(key, None)
            self.lists.pop(key, None)
            self.hashes.pop(key, None)
            self.expirations.pop(key, None)
            deleted += int(existed)
        return deleted

    async def expire(self, key: str, seconds: int) -> bool:
        if not await self.exists(key):
            return False
        self.expirations[key] = int(seconds)
        return True

    async def ttl(self, key: str) -> int:
        return self.expirations.get(key, -1 if await self.exists(key) else -2)

    async def setex(self, key: str, seconds: int, value: Any) -> bool:
        self.strings[key] = value
        self.expirations[key] = int(seconds)
        return True

    async def get(self, key: str) -> Any:
        return self.strings.get(key)

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None
