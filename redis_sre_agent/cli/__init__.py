"""命令行入口目录。

`pyproject.toml` 里的 `redis-sre-agent = "redis_sre_agent.cli:main"`
会先导入这个包，再取出这里暴露的 `main` 函数。
"""

from .main import main

__all__ = ["main"]
