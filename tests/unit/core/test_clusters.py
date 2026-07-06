"""Redis 集群模型和存储测试。

所有 Redis 行为都由内存 fake 提供，不访问真实 Redis。
"""

from __future__ import annotations

import base64
import json
import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from redis_sre_agent.core.clusters import (
    RedisCluster,
    RedisClusterType,
    get_cluster_by_id,
    query_clusters,
    save_clusters,
)
from redis_sre_agent.core.encryption import is_encrypted
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
            if key.startswith("sre_clusters:")
        ]
        if query.__class__.__name__ == "CountQuery":
            return len(docs)
        return docs


@pytest.fixture
def master_key_env():
    key = base64.b64encode(os.urandom(32)).decode("ascii")
    with patch.dict(os.environ, {"REDIS_SRE_MASTER_KEY": key}):
        yield


def _cluster() -> RedisCluster:
    return RedisCluster(
        id="cluster-test-1",
        name="测试集群",
        cluster_type=RedisClusterType.redis_enterprise,
        environment="production",
        description="本地测试集群",
        admin_url="https://cluster.example.invalid:9443",
        admin_username="admin@example.invalid",
        admin_password=SecretStr("FAKE_ADMIN_SECRET"),
        extension_secrets={"token": SecretStr("FAKE_CLUSTER_EXTENSION_SECRET")},
    )


def test_cluster_model_validation() -> None:
    cluster = _cluster()
    assert cluster.environment == "production"

    with pytest.raises(ValueError, match="admin_url"):
        RedisCluster(
            id="cluster-test-2",
            name="bad",
            cluster_type=RedisClusterType.redis_enterprise,
            environment="production",
            description="bad",
            admin_url="https://cluster.example.invalid:9443",
            admin_username="admin@example.invalid",
        )

    with pytest.raises(ValueError, match="只适用于"):
        RedisCluster(
            id="cluster-test-3",
            name="bad",
            cluster_type=RedisClusterType.oss_cluster,
            environment="production",
            description="bad",
            admin_url="https://cluster.example.invalid:9443",
            admin_username="admin@example.invalid",
            admin_password=SecretStr("FAKE_ADMIN_SECRET"),
        )


@pytest.mark.asyncio
async def test_save_and_read_clusters_encrypts_sensitive_fields(master_key_env) -> None:
    fake_redis = FakeRedis()
    fake_index = FakeIndex(fake_redis)
    cluster = _cluster()

    with (
        patch("redis_sre_agent.core.clusters.get_redis_client", return_value=fake_redis),
        patch("redis_sre_agent.core.clusters.get_clusters_index", return_value=fake_index),
    ):
        assert await save_clusters([cluster]) is True
        stored = fake_redis.hashes[RedisKeys.cluster_doc(cluster.id)]
        stored_data = json.loads(stored["data"])

        assert stored_data["admin_password"] != "FAKE_ADMIN_SECRET"
        assert is_encrypted(stored_data["admin_password"]) is True
        assert is_encrypted(stored_data["extension_secrets"]["token"]) is True

        restored = await get_cluster_by_id(cluster.id)

    assert restored is not None
    assert restored.admin_password.get_secret_value() == "FAKE_ADMIN_SECRET"
    assert restored.extension_secrets["token"].get_secret_value() == "FAKE_CLUSTER_EXTENSION_SECRET"


@pytest.mark.asyncio
async def test_query_clusters_uses_escaped_search(master_key_env) -> None:
    fake_redis = FakeRedis()
    fake_index = FakeIndex(fake_redis)
    cluster = _cluster()

    with (
        patch("redis_sre_agent.core.clusters.get_redis_client", return_value=fake_redis),
        patch("redis_sre_agent.core.clusters.get_clusters_index", return_value=fake_index),
    ):
        await save_clusters([cluster])
        result = await query_clusters(environment="production", search="测试|集群")

    assert result.total == 1
    assert result.clusters[0].id == cluster.id
    assert r"@name:{*测试\|集群*}" in str(fake_index.queries[-1])
