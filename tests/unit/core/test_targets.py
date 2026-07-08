"""阶段三 target catalog 与解析测试。"""

from __future__ import annotations

from pydantic import SecretStr
import pytest

from redis_sre_agent.core.instances import RedisInstance
from redis_sre_agent.core.redis import SRE_TARGETS_SCHEMA
from redis_sre_agent.core import targets as targets_module
from redis_sre_agent.targets.registry import reset_target_integration_registry


def make_instance(
    *,
    instance_id: str = "inst-prod-checkout-cache",
    name: str = "Prod Checkout Cache",
    user_id: str | None = None,
) -> RedisInstance:
    return RedisInstance(
        id=instance_id,
        name=name,
        connection_url=SecretStr("FAKE_TEST_REDIS_CONNECTION_REF"),
        environment="production",
        usage="cache",
        description="Checkout service cache",
        repo_url="https://github.com/example/checkout-service",
        monitoring_identifier="checkout-cache-prod",
        status="healthy",
        extension_data={"aliases": ["checkout prod", "checkout-cache"]},
        user_id=user_id,
    )


def test_targets_schema_is_stage_three_index_slot() -> None:
    assert SRE_TARGETS_SCHEMA["index"]["name"] == "sre_targets"
    assert {"target_kind", "display_name", "environment", "search_text"}.issubset(
        {field["name"] for field in SRE_TARGETS_SCHEMA["fields"]}
    )


def test_target_doc_never_exposes_connection_secret() -> None:
    doc = targets_module.build_target_doc_from_instance(make_instance())
    payload = doc.model_dump_json()

    assert doc.target_kind == "instance"
    assert doc.environment == "production"
    assert "checkout prod" in doc.search_aliases
    assert "FAKE_TEST_REDIS_CONNECTION_REF" not in payload


@pytest.mark.asyncio
async def test_resolve_target_query_returns_secret_safe_match(monkeypatch) -> None:
    reset_target_integration_registry()
    instance = make_instance()

    async def fake_get_instances():
        return [instance]

    async def fake_get_clusters():
        return []

    monkeypatch.setattr(targets_module, "get_instances", fake_get_instances)
    monkeypatch.setattr(targets_module, "get_clusters", fake_get_clusters)

    response = await targets_module.resolve_target_query(
        query=instance.name,
        preferred_capabilities=["diagnostics"],
    )
    payload = response.public_dump()

    assert response.status == "resolved"
    assert response.matches[0].display_name == instance.name
    assert response.selected_matches[0].resource_id == instance.id
    assert "FAKE_TEST_REDIS_CONNECTION_REF" not in str(payload)
