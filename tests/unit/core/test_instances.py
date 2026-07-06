"""Redis 实例模型和存储测试。

存储测试使用内存 fake，不连接真实 Redis。测试值都是本地占位值，不代表真实连接串或密码。
"""

from __future__ import annotations

import base64
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from redis_sre_agent.core.encryption import is_encrypted
from redis_sre_agent.core.instances import (
    RedisInstance,
    RedisInstanceType,
    get_instance_by_id,
    get_instance_map,
    get_instance_name,
    mask_redis_url,
    query_instances,
    save_instances,
)
from redis_sre_agent.core.keys import RedisKeys


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.deleted: list[str] = []

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = mapping

    async def hget(self, key: str, field: str):
        return self.hashes.get(key, {}).get(field)

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.hashes.pop(key, None)


class FakeIndex:
    def __init__(self, redis_client: FakeRedis) -> None:
        self.redis_client = redis_client
        self.queries: list[object] = []

    async def exists(self) -> bool:
        return True

    async def create(self) -> None:
        return None

    async def query(self, query):
        self.queries.append(query)
        docs = [
            {"data": mapping["data"]}
            for key, mapping in self.redis_client.hashes.items()
            if key.startswith("sre_instances:")
        ]
        if query.__class__.__name__ == "CountQuery":
            return len(docs)
        return docs


@pytest.fixture
def master_key_env():
    key = base64.b64encode(os.urandom(32)).decode("ascii")
    with patch.dict(os.environ, {"REDIS_SRE_MASTER_KEY": key}):
        yield


def _instance() -> RedisInstance:
    return RedisInstance(
        id="redis-test-1",
        name="测试实例",
        connection_url=SecretStr("redis://user:fake-secret@example.invalid:6379/0"),
        environment="test",
        usage="cache",
        description="本地测试实例",
        instance_type=RedisInstanceType.oss_single,
        extension_secrets={"token": SecretStr("FAKE_EXTENSION_SECRET")},
    )


def test_instance_model_and_masking() -> None:
    instance = _instance()

    assert instance.created_by == "user"
    masked = mask_redis_url(instance.connection_url)
    assert "fake-secret" not in masked
    assert "***:***@" in masked


def test_invalid_created_by_rejected() -> None:
    with pytest.raises(ValueError, match="created_by"):
        RedisInstance(
            id="redis-test-2",
            name="bad",
            connection_url="redis://localhost:6379/0",
            environment="test",
            usage="cache",
            description="bad",
            instance_type=RedisInstanceType.oss_single,
            created_by="system",
        )


@pytest.mark.asyncio
async def test_save_and_read_instances_encrypts_sensitive_fields(master_key_env) -> None:
    fake_redis = FakeRedis()
    fake_index = FakeIndex(fake_redis)
    instance = _instance()

    with (
        patch("redis_sre_agent.core.instances.get_redis_client", return_value=fake_redis),
        patch("redis_sre_agent.core.instances.get_instances_index", return_value=fake_index),
    ):
        assert await save_instances([instance]) is True
        stored = fake_redis.hashes[RedisKeys.instance_doc(instance.id)]
        stored_data = json.loads(stored["data"])

        assert stored_data["connection_url"] != instance.connection_url.get_secret_value()
        assert is_encrypted(stored_data["connection_url"]) is True
        assert is_encrypted(stored_data["extension_secrets"]["token"]) is True

        restored = await get_instance_by_id(instance.id)

    assert restored is not None
    assert restored.connection_url.get_secret_value() == instance.connection_url.get_secret_value()
    assert restored.extension_secrets["token"].get_secret_value() == "FAKE_EXTENSION_SECRET"


@pytest.mark.asyncio
async def test_query_and_lookup_helpers_use_fake_index(master_key_env) -> None:
    fake_redis = FakeRedis()
    fake_index = FakeIndex(fake_redis)
    instance = _instance()

    with (
        patch("redis_sre_agent.core.instances.get_redis_client", return_value=fake_redis),
        patch("redis_sre_agent.core.instances.get_instances_index", return_value=fake_index),
    ):
        await save_instances([instance])
        result = await query_instances(search="测试|实例")
        instance_map = await get_instance_map()
        instance_name = await get_instance_name(instance.id)

    assert result.total == 1
    assert result.instances[0].id == instance.id
    assert instance_map[instance.id].name == "测试实例"
    assert instance_name == "测试实例"
    assert any(r"@name:{*测试\|实例*}" in str(query) for query in fake_index.queries)
