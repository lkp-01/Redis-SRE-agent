"""阶段一导入测试。

这些测试只验证 Python 能不能找到项目里的包和模块。
测试不访问网络，不连接 Redis，不读取密钥，也不依赖任何真实外部服务。

这样做的第一性原理是：项目骨架必须先能被 Python 导入，后续业务功能才有落脚点。
如果导入都失败，说明问题在包结构、安装方式或路径配置上，而不是 Redis 或 Agent 逻辑上。
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
        # importlib 可以用字符串导入模块，适合测试一组固定导入路径是否都还存在。
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_cli_main_module_imports() -> None:
    module = importlib.import_module("redis_sre_agent.cli.main")

    assert callable(module.main)
