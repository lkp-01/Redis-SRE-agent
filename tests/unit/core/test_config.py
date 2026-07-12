"""阶段二配置测试。

测试只验证 Settings 的本地解析能力。这里不会读取真实 OpenAI key，也不会访问网络。
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import SecretStr, ValidationError

from redis_sre_agent.core.config import (
    DEFAULT_CONFIG_PATHS,
    MCPServerConfig,
    MCPToolConfig,
    Settings,
)
from redis_sre_agent.tools.models import ToolActionKind, ToolCapability


def test_settings_can_be_created_without_real_external_secrets() -> None:
    # config 模块会按 original 形状加载本机 .env；本测试显式隔离环境，不能依赖开发者
    # 是否已经配置 chat/embedding key。
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)

    assert settings.app_name == "Redis SRE Agent"
    assert settings.debug is False
    assert isinstance(settings.redis_url, SecretStr)
    assert settings.openai_api_key is None
    assert settings.openai_base_url == "https://api.deepseek.com"
    assert settings.openai_model == "deepseek-v4-pro"
    assert settings.openai_model_mini == "deepseek-v4-flash"
    assert settings.openai_model_nano == "deepseek-v4-flash"
    assert settings.llm_timeout == 180.0
    assert settings.llm_failover_enabled is True
    assert settings.deepseek_thinking_mode == "disabled"
    assert settings.rag_enabled is False
    assert settings.embedding_api_key is None
    assert settings.embedding_base_url is None
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.vector_dim == 1536


def test_secretstr_does_not_leak_in_repr() -> None:
    settings = Settings(
        _env_file=None,
        redis_url=SecretStr("FAKE_TEST_REDIS_CONNECTION_REF"),
        embedding_api_key=SecretStr("FAKE_TEST_EMBEDDING_KEY"),
    )

    rendered = repr(settings)
    assert "FAKE_TEST_REDIS_CONNECTION_REF" not in rendered
    assert "FAKE_TEST_EMBEDDING_KEY" not in rendered
    assert "SecretStr" in rendered


def test_environment_overrides_defaults() -> None:
    env = {
        "APP_NAME": "local-test-agent",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "HOST": "127.0.0.1",
        "PORT": "9001",
        "REDIS_URL": "LOCAL_TEST_REDIS_REFERENCE",
        "OPENAI_MODEL": "primary-test-model",
        "OPENAI_MODEL_MINI": "fallback-test-model",
        "LLM_FAILOVER_ENABLED": "false",
        "DEEPSEEK_THINKING_MODE": "enabled",
        "RAG_ENABLED": "true",
        "EMBEDDING_API_KEY": "SEPARATE_EMBEDDING_KEY",
        "EMBEDDING_BASE_URL": "https://embedding.example.invalid/v1",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = Settings(_env_file=None)

    assert settings.app_name == "local-test-agent"
    assert settings.debug is True
    assert settings.log_level == "DEBUG"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9001
    assert settings.redis_url.get_secret_value() == "LOCAL_TEST_REDIS_REFERENCE"
    assert settings.openai_model == "primary-test-model"
    assert settings.openai_model_mini == "fallback-test-model"
    assert settings.llm_failover_enabled is False
    assert settings.deepseek_thinking_mode == "enabled"
    assert settings.rag_enabled is True
    assert settings.embedding_api_key is not None
    assert settings.embedding_api_key.get_secret_value() == "SEPARATE_EMBEDDING_KEY"
    assert settings.embedding_base_url == "https://embedding.example.invalid/v1"


def test_yaml_config_file_loads_and_env_wins(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"app_name": "from-yaml", "debug": False, "mcp_servers": {"demo": {"command": "echo"}}}),
        encoding="utf-8",
    )

    with patch.dict(
        os.environ,
        {"SRE_AGENT_CONFIG": str(config_path), "DEBUG": "true"},
        clear=True,
    ):
        settings = Settings(_env_file=None)

    assert settings.app_name == "from-yaml"
    assert settings.debug is True
    assert "demo" in settings.mcp_servers
    server = settings.mcp_servers["demo"]
    if isinstance(server, MCPServerConfig):
        assert server.command == "echo"
    else:
        assert server["command"] == "echo"


def test_default_config_paths_are_stable() -> None:
    assert DEFAULT_CONFIG_PATHS == [
        "config.yaml",
        "config.yml",
        "config.toml",
        "config.json",
        "sre_agent_config.yaml",
        "sre_agent_config.yml",
        "sre_agent_config.toml",
        "sre_agent_config.json",
    ]


def test_mcp_server_config_accepts_exactly_one_connection_mode() -> None:
    stdio = MCPServerConfig(command="python", args=["fake_server.py"])
    streamable = MCPServerConfig(url="https://mcp.example.invalid/v1")

    assert stdio.command == "python"
    assert stdio.url is None
    assert stdio.transport in {None, "stdio"}
    assert streamable.command is None
    assert streamable.url == "https://mcp.example.invalid/v1"
    assert streamable.transport == "streamable_http"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"command": "python", "url": "https://mcp.example.invalid/v1"},
        {"command": "python", "transport": "sse"},
        {"url": "https://mcp.example.invalid/v1", "transport": "stdio"},
        {"url": "ftp://mcp.example.invalid/v1", "transport": "streamable_http"},
    ],
)
def test_mcp_server_config_rejects_ambiguous_or_invalid_transports(payload) -> None:
    with pytest.raises(ValidationError):
        MCPServerConfig.model_validate(payload)


def test_mcp_server_config_requires_opt_in_for_non_loopback_plain_http() -> None:
    with pytest.raises(ValidationError):
        MCPServerConfig(
            url="http://mcp.example.invalid/v1",
            transport="streamable_http",
        )

    opted_in = MCPServerConfig(
        url="http://mcp.example.invalid/v1",
        transport="streamable_http",
        allow_insecure_http=True,
    )
    loopback = MCPServerConfig(
        url="http://127.0.0.1:8765/mcp",
        transport="sse",
    )

    assert opted_in.allow_insecure_http is True
    assert loopback.allow_insecure_http is False


def test_mcp_tool_allowlist_parses_capability_description_and_action_kind() -> None:
    config = MCPServerConfig.model_validate(
        {
            "command": "python",
            "tools": {
                "read_status": {
                    "capability": "diagnostics",
                    "description": "读取外部状态。",
                    "action_kind": "read",
                }
            },
        }
    )

    assert isinstance(config.tools["read_status"], MCPToolConfig)
    tool = config.tools["read_status"]
    assert tool.capability is ToolCapability.DIAGNOSTICS
    assert tool.description == "读取外部状态。"
    assert tool.action_kind is ToolActionKind.READ


def test_mcp_config_repr_and_validation_errors_do_not_expose_secrets() -> None:
    sentinel = "MCP_CONFIG_SENTINEL_SECRET"
    server = MCPServerConfig(
        url="https://mcp.example.invalid/v1",
        headers={"Authorization": f"Bearer {sentinel}"},
        tools={"read_status": MCPToolConfig(action_kind=ToolActionKind.READ)},
    )
    settings = Settings(
        _env_file=None,
        mcp_servers={
            "stdio": {
                "command": "python",
                "env": {"MCP_TEST_TOKEN": sentinel},
                "tools": {"read_status": {"action_kind": "read"}},
            },
            "remote": server,
        },
    )

    assert sentinel not in repr(server)
    assert sentinel not in repr(settings)

    with pytest.raises(ValidationError) as exc_info:
        MCPServerConfig.model_validate(
            {
                "command": "python",
                "url": "https://mcp.example.invalid/v1",
                "env": {"MCP_TEST_TOKEN": sentinel},
                "headers": {"Authorization": f"Bearer {sentinel}"},
            }
        )

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)
