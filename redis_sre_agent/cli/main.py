"""CLI interface for Redis SRE Agent."""

# 根命令只负责 Click 入口、命令注册和 lazy loading。具体 query 生命周期在
# redis_sre_agent.cli.query 中，避免把 Agent 主逻辑堆进 main.py。

from __future__ import annotations

import importlib

import click

from redis_sre_agent import __version__
from redis_sre_agent.cli.logging_utils import (
    configure_cli_logging,
    log_cli_exception,
    was_cli_exception_logged,
)

configure_cli_logging() #这一次调用确保了在进入 Click 核心路由链路之前，底层的日志系统就已经准备就绪。

_COMMANDS = {
    "query": "redis_sre_agent.cli.query:query",
}

_BUILTIN_COMMANDS = {"version", "status"}


@click.command()
def version() -> None:
    """Show the Redis SRE Agent version."""

    click.echo(f"redis-sre-agent {__version__}")


@click.command()
def status() -> None:
    """Show the current diagnostic slice status."""

    click.echo("redis-sre-agent 诊断切片：Stage 5 LangGraph Agent 主链路可以正常启动")


class LazyGroup(click.MultiCommand):
    """Lazy loading of CLI commands to avoid hard dependencies at top level."""

    def list_commands(self, ctx):
        return list(_COMMANDS.keys()) + list(_BUILTIN_COMMANDS)

    def get_command(self, ctx, name):
        configure_cli_logging() #在真正导入并返回子命令目标（如 query）之前，再次强制执行一次，可以强行刷新并覆盖可能被子模块篡改的 root logger 配置，确保后续整个 Agent 链路执行时，输出的日志依然完全符合 CLI 预期的格式。
        if name == "version":
            return version
        if name == "status":
            return status
        target = _COMMANDS.get(name)
        if not target:
            return None
        module_path, attr = target.split(":", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    def invoke(self, ctx):
        try:
            return super().invoke(ctx) #命令行输入命令后，Click会解析参数并调用 invoke执行子命令逻辑
        except (click.exceptions.Exit, click.ClickException, click.Abort, SystemExit): # 这类异常是 Click 框架自带的、或者用户主动触发的正常退出信号
            raise
        except Exception as exc:
            if not was_cli_exception_logged(exc):
                log_cli_exception(__name__, "CLI command failed", exc)
            raise


@click.command(cls=LazyGroup)
def main() -> None:
    """Redis SRE Agent CLI."""

    pass


if __name__ == "__main__":
    main()
