from __future__ import annotations

import os

import pytest


# ============================================================
# 1. 注册 pytest 插件
# ============================================================
# py_reporter.py 负责收集和输出 eval 的评测结果。
pytest_plugins = ["evals.py_reporter"]


# ============================================================
# 2. 注册自定义 pytest marker
# ============================================================
# 以后测试可以这样写：
#
# @pytest.mark.eval_category("tool_use")
# @pytest.mark.eval_tier("baseline")
# def test_xxx():
#     ...
#
# eval_category：这个测试在测什么能力
# 例如：
#   tool_use
#   retrieval
#   memory
#   safety
#   reliability
#   diagnosis
#
# eval_tier：这个测试的重要程度 / 用途
# baseline：
#   核心回归测试，原则上每次修改 Agent 后都应该通过
#
# hillclimb：
#   用来衡量 Agent 能力是否提高，不一定要求 100% 通过


def pytest_configure(config: pytest.Config) -> None:
    """注册 Redis SRE Agent evals 使用的自定义 marker。"""

    config.addinivalue_line(
        "markers",
        "eval_category(name): classify an eval by capability/category",
    )

    config.addinivalue_line(
        "markers",
        "eval_tier(name): classify an eval as baseline or hillclimb",
    )


# ============================================================
# 3. 给 pytest 增加 eval 专用命令行参数
# ============================================================
#
# 以后可以这样运行：
#
# pytest evals --model deepseek-chat
#
# 只运行 tool_use：
# pytest evals --model deepseek-chat --eval-category tool_use
#
# 运行多个类别：
# pytest evals \
#     --model deepseek-chat \
#     --eval-category tool_use \
#     --eval-category retrieval
#
# 只跑 baseline：
# pytest evals --model deepseek-chat --eval-tier baseline
#
# 排除 safety：
# pytest evals \
#     --model deepseek-chat \
#     --eval-category-exclude safety


def pytest_addoption(parser: pytest.Parser) -> None:
    """注册 evals 使用的 pytest CLI 参数。"""

    # 指定本次评测使用哪个模型。
    # 后面测试可以通过 model_name fixture 获取这个值。
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Model used to run Redis SRE Agent evals.",
    )

    # 只运行指定类别的 eval。
    # action='append' 表示这个参数可以写多次。
    parser.addoption(
        "--eval-category",
        action="append",
        default=[],
        help=(
            "Run only evals in the specified category. "
            "Can be provided multiple times."
        ),
    )

    # 排除某些类别。
    parser.addoption(
        "--eval-category-exclude",
        action="append",
        default=[],
        help=(
            "Exclude evals in the specified category. "
            "Exclude takes precedence over include."
        ),
    )

    # 只运行某个 tier。
    # 例如 baseline / hillclimb。
    parser.addoption(
        "--eval-tier",
        action="append",
        default=[],
        help=(
            "Run only evals in the specified tier. "
            "For example: baseline or hillclimb."
        ),
    )


# ============================================================
# 4. 通用 marker 筛选器
# ============================================================
#
# 这一段基本属于 pytest 基础设施代码。
#
# 功能：
#
# 假设有三个测试：
#
# A -> tool_use
# B -> retrieval
# C -> safety
#
# 执行：
#
# pytest evals --eval-category tool_use
#
# 那么最终 pytest 只留下 A。
#
# 如果传入一个根本不存在的 category，
# 例如：
#
# --eval-category abc
#
# 会直接提示：
#
# Unknown --eval-category values: ['abc']
#
# 这样可以防止因为拼写错误导致“一个测试都没运行”却没发现。


def _filter_by_marker(
    config: pytest.Config,
    items: list[pytest.Item],
    *,
    option: str,
    marker_name: str,
    exclude_option: str | None = None,
) -> None:
    """根据 pytest marker 对收集到的 eval 测试进行筛选。"""

    # 用户通过 CLI 指定的 include 列表。
    values = config.getoption(option)

    # 某些 marker 支持 exclude，例如 eval_category。
    excluded = (
        config.getoption(exclude_option)
        if exclude_option
        else []
    )

    # 用户没有指定任何过滤条件，直接运行全部测试。
    if not values and not excluded:
        return

    # 收集当前测试集中真实存在的 marker 值。
    #
    # 例如：
    # known = {"tool_use", "retrieval", "safety"}
    known = {
        marker.args[0]
        for item in items
        if (
            marker := item.get_closest_marker(marker_name)
        )
        and marker.args
    }

    # 检查用户传入了不存在的 marker。
    unknown = set(values) - known
    unknown_excluded = set(excluded) - known

    if unknown or unknown_excluded:
        parts: list[str] = []

        if unknown:
            parts.append(
                f"Unknown {option} values: {sorted(unknown)}"
            )

        if unknown_excluded:
            parts.append(
                f"Unknown {exclude_option} values: "
                f"{sorted(unknown_excluded)}"
            )

        message = (
            f"{'; '.join(parts)}. "
            f"Known values: {sorted(known)}"
        )

        pytest.exit(message, returncode=1)

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    # 对 pytest 收集到的每个测试逐一判断。
    for item in items:
        marker = item.get_closest_marker(marker_name)

        marker_value = (
            marker.args[0]
            if marker and marker.args
            else None
        )

        # 没有 include 条件时默认全部包含。
        included = (
            not values
            or marker_value in values
        )

        # exclude 的优先级高于 include。
        is_excluded = marker_value in excluded

        if included and not is_excluded:
            selected.append(item)
        else:
            deselected.append(item)

    # 修改 pytest 最终真正执行的测试集合。
    items[:] = selected

    # 告诉 pytest 哪些测试被 deselect 了，
    # 这样终端会正确显示：
    #
    # 5 selected, 10 deselected
    config.hook.pytest_deselected(items=deselected)


# ============================================================
# 5. pytest 收集完测试后，执行筛选
# ============================================================
#
# pytest_collection_modifyitems 是 pytest 自带 hook。
#
# pytest 收集完所有 test_xxx 后会自动调用这里。
#
# 我们分别按照：
# 1. eval_category
# 2. eval_tier
#
# 过滤一次。


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:

    # 按能力类别过滤。
    _filter_by_marker(
        config,
        items,
        option="--eval-category",
        marker_name="eval_category",
        exclude_option="--eval-category-exclude",
    )

    # 按 baseline / hillclimb 过滤。
    _filter_by_marker(
        config,
        items,
        option="--eval-tier",
        marker_name="eval_tier",
    )


# ============================================================
# 6. model_name fixture
# ============================================================

@pytest.fixture
def model_name(request: pytest.FixtureRequest) -> str | None:
    """返回本次 eval 通过 --model 指定的模型名称。"""

    model = request.config.getoption("--model")

    if model is None:
        return None

    return str(model)


@pytest.fixture
def eval_redis_url() -> str:
    """返回显式配置的专用 Redis eval URL；未配置时跳过真实 Redis 链路测试。"""

    redis_url = os.getenv("EVAL_REDIS_URL")
    if not redis_url:
        pytest.skip(
            "set EVAL_REDIS_URL to a dedicated loopback Redis instance to run real Redis evals"
        )
    return redis_url


