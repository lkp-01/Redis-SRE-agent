"""命令行入口。

阶段三在 `status` 基础上增加 `query` 曳光弹入口。入口沿用原项目 router/response
形状，但内部只运行确定性 mock 链路，不调用 OpenAI，也不执行真实 Redis 诊断命令。
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import json
import sys

from redis_sre_agent import __version__


class ChineseArgumentParser(argparse.ArgumentParser):
    """
    命令行解析工具报错时全是英文（比如 error: unrecognized arguments）。
    这个类把原生的工具包装了一下，作用只有一个
    如果用户在终端里敲错了命令参数，它会把错误提示翻译成中文，然后强制退出程序
    """

    def error(self, message: str) -> None: 
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: 参数错误：{message}\n")
        # 强行终止程序。2 是一种约定俗成的状态码，专门告诉操作系统“这人命令行参数填错了”。
        # 后面的 f"..." 是把程序名字（self.prog）和具体的错误原因（message）拼接成一句中文提示并打印出来。


def build_parser() -> argparse.ArgumentParser:
    """创建 CLI 参数解析器。

    这里仍然延迟导入 Mock Agent，避免 `--help` 或 `status` 提前加载工具系统。
    """
    parser = ChineseArgumentParser(
        prog="redis-sre-agent",
        usage="redis-sre-agent [--version] [status|query] [参数]",
        description="Redis SRE Agent 诊断切片命令行入口。",
        epilog="query 当前是阶段三 Mock 链路，不访问 OpenAI，也不执行真实 Redis 诊断。",
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="显示这段帮助信息后退出。",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"redis-sre-agent {__version__}",
        help="显示当前安装的包版本后退出。",
    )
    parser.add_argument(
        "command",
        nargs="?",
        metavar="命令",
        help="可选命令：status 或 query。",
    )
    parser.add_argument(
        "query",
        nargs="*",
        metavar="查询",
        help="query 命令的查询文本。",
    )
    parser.add_argument(
        "--target",
        help="可选 target 提示，例如 prod checkout cache。",
    )
    parser.add_argument(
        "--instance-id",
        help="可选实例 ID，用于直接构造 target seed。",
    )
    parser.add_argument(
        "--user-id",
        help="可选用户 ID，用于 target catalog 过滤。",
    )
    parser._positionals.title = "位置参数"
    parser._optionals.title = "可选参数"
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行最小 CLI。

    `argv` 是参数列表。测试时可以传入一个列表，真实命令行运行时则让 argparse 自己读取终端参数。
    函数返回整数退出码：`0` 表示正常结束，非零值表示参数错误或运行失败。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        print("redis-sre-agent 诊断切片：阶段三曳光弹入口可以正常启动")
        return 0

    if args.command == "query":
        query_text = " ".join(args.query).strip()
        if not query_text and not args.target and not args.instance_id:
            parser.error("query 命令需要查询文本、--target 或 --instance-id。")
        from redis_sre_agent.agent.models import AgentResponse
        from redis_sre_agent.agent.router import route_to_appropriate_agent
        from redis_sre_agent.core.targets import bind_target_matches, build_seed_hint_candidates
        from redis_sre_agent.tools.manager import ToolManager

        async def _run_query() -> AgentResponse:
            context = {"instance_id": args.instance_id}
            agent_type = await route_to_appropriate_agent(query_text, context=context)
            tool_envelopes = []
            async with ToolManager(user_id=args.user_id) as tool_manager:
                if args.instance_id:
                    candidates = await build_seed_hint_candidates(instance_id=args.instance_id)
                    target_resolution = await bind_target_matches(
                        matches=candidates,
                        manager=tool_manager,
                    )
                    tool_envelopes.append(
                        {
                            "tool_key": "target_binding",
                            "name": "bind_target_matches",
                            "status": "success" if target_resolution.bindings else "no_match",
                            "data": target_resolution.model_dump(mode="json"),
                        }
                    )
                else:
                    discovery_tools = tool_manager.get_tools_by_provider_names(["target_discovery"])
                    resolve_tool = next(
                        (tool.name for tool in discovery_tools if tool.name.endswith("resolve_redis_targets")),
                        None,
                    )
                    if resolve_tool:
                        target_result = await tool_manager.resolve_tool_call(
                            resolve_tool,
                            {
                                "query": args.target or query_text,
                                "allow_multiple": False,
                                "max_results": 5,
                                "attach_tools": True,
                                "preferred_capabilities": ["diagnostics"],
                            },
                        )
                    else:
                        target_result = {"status": "no_target_discovery_tool"}
                    tool_envelopes.append(
                        {
                            "tool_key": resolve_tool or "target_discovery",
                            "name": "resolve_redis_targets",
                            "status": str(target_result.get("status", "unknown")),
                            "data": target_result,
                        }
                    )

                redis_tools = tool_manager.get_tools_by_provider_names(["redis_command"])
                info_tool = next((tool.name for tool in redis_tools if tool.name.endswith("info")), None)
                if info_tool:
                    info_result = await tool_manager.resolve_tool_call(info_tool, {})
                    tool_envelopes.append(
                        {
                            "tool_key": info_tool,
                            "name": "info",
                            "status": str(info_result.get("status", "unknown")),
                            "data": info_result,
                        }
                    )

            status = "completed" if any(env.get("name") == "info" for env in tool_envelopes) else "no_target"
            return AgentResponse(
                response=(
                    f"阶段三曳光弹完成：route={agent_type.value}, status={status}。"
                    if status == "completed"
                    else f"阶段三曳光弹未绑定 Redis target：route={agent_type.value}。"
                ),
                tool_envelopes=tool_envelopes,
            )

        response = asyncio.run(_run_query())
        print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0

    if args.command is not None:
        parser.error(f"未知命令：{args.command}。当前阶段支持 status 或 query。")

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
