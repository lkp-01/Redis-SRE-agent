"""Redis 资源层基础测试。

这里用 mock 替代真实 Redis 客户端，所以不会访问真实 Redis 或网络。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from redis_sre_agent.core.redis import (
    SRE_CLUSTERS_SCHEMA,
    SRE_INSTANCES_SCHEMA,
    get_clusters_index,
    get_instances_index,
    get_redis_client,
    test_redis_connection,
)


def test_instance_and_cluster_schemas_are_present() -> None:
    assert SRE_INSTANCES_SCHEMA["index"]["name"] == "sre_instances"
    assert SRE_CLUSTERS_SCHEMA["index"]["name"] == "sre_clusters"
    assert {"name", "environment", "status"}.issubset(
        {field["name"] for field in SRE_INSTANCES_SCHEMA["fields"]}
    )


@patch("redis_sre_agent.core.redis.Redis")
def test_get_redis_client_creates_client_without_caching(mock_redis) -> None:
    mock_client = Mock()
    mock_redis.from_url.return_value = mock_client

    assert get_redis_client(url="redis://localhost:6379/0") is mock_client
    assert get_redis_client(url="redis://localhost:6379/0") is mock_client
    assert mock_redis.from_url.call_count == 2


@pytest.mark.asyncio
async def test_get_indices_return_lightweight_index_objects() -> None:
    with patch("redis_sre_agent.core.redis.get_redis_client", return_value=AsyncMock()):
        instances_index = await get_instances_index()
        clusters_index = await get_clusters_index()

    assert instances_index.name == "sre_instances"
    assert clusters_index.name == "sre_clusters"


@pytest.mark.asyncio
async def test_redis_connection_uses_mock_client() -> None:
    client = AsyncMock()
    client.ping.return_value = True

    with patch("redis_sre_agent.core.redis.get_redis_client", return_value=client):
        assert await test_redis_connection() is True

    client.ping.assert_awaited_once()
