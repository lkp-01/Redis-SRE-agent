"""项目级导入测试。

这些测试只验证 Python 能不能找到项目里的包和模块。它们不访问网络，不连接 Redis，不读取
真实密钥，也不依赖任何外部服务。
"""

from __future__ import annotations

import importlib


def test_redis_sre_agent_imports() -> None:
    import redis_sre_agent

    assert redis_sre_agent.__name__ == "redis_sre_agent"
    assert isinstance(redis_sre_agent.__version__, str)


def test_placeholder_packages_import() -> None:
    module_names = [
        "redis_sre_agent.core",
        "redis_sre_agent.cli",
        "redis_sre_agent.agent",
        "redis_sre_agent.tools",
        "redis_sre_agent.targets",
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_stage_two_core_modules_import() -> None:
    module_names = [
        "redis_sre_agent.core.config",
        "redis_sre_agent.core.encryption",
        "redis_sre_agent.core.keys",
        "redis_sre_agent.core.redis",
        "redis_sre_agent.core.redisearch",
        "redis_sre_agent.core.instances",
        "redis_sre_agent.core.clusters",
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_cli_main_module_imports() -> None:
    module = importlib.import_module("redis_sre_agent.cli.main")

    assert callable(module.main)
