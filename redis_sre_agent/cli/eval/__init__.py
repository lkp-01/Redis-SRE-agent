"""运行任意 outcome 离线 replay 场景的命令行入口。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import click

from redis_sre_agent.evaluation.runner import run_eval_scenario


def _echo_console_safe(value: object) -> None:
    """按当前终端编码降级不可表示字符，避免 Windows GBK 输出令评测失败。"""

    rendered = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        rendered = rendered.encode(encoding, errors="backslashreplace").decode(encoding)
    except LookupError:
        pass
    try:
        click.echo(rendered)
    except UnicodeEncodeError:
        click.echo(rendered.encode("ascii", errors="backslashreplace").decode("ascii"))


@click.group()
def eval() -> None:
    """运行本切片保留的离线评测。"""


@eval.command("run")
@click.argument("scenario_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="输出 JSON 结果。")
def run(scenario_path: Path, as_json: bool) -> None:
    """运行一个 outcome replay 场景。"""

    result = asyncio.run(run_eval_scenario(str(scenario_path)))
    payload = result.model_dump(mode="json")
    if as_json:
        _echo_console_safe(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        click.echo(f"scenario_id={result.scenario_id}")
        passed_label = "invalid" if result.passed is None else str(result.passed).lower()
        click.echo(f"passed={passed_label}")
        if result.judge_result is not None:
            click.echo(f"judge_score={result.judge_result.overall_score:.1f}")
            judge_passed_label = (
                "invalid"
                if not result.judge_result.judge_valid
                else str(bool(result.judge_passed)).lower()
            )
            click.echo(f"judge_passed={judge_passed_label}")
            click.echo(f"judge_valid={str(result.judge_result.judge_valid).lower()}")
            click.echo(f"factual_errors={len(result.judge_result.factual_errors)}")
            for index, error in enumerate(result.judge_result.factual_errors, start=1):
                click.echo(f"factual_error[{index}]={error}")
            click.echo(f"missing_elements={len(result.judge_result.missing_elements)}")
            for index, element in enumerate(result.judge_result.missing_elements, start=1):
                click.echo(f"missing_element[{index}]={element}")
            click.echo(f"partial_elements={len(result.judge_result.partial_elements)}")
            for index, element in enumerate(result.judge_result.partial_elements, start=1):
                click.echo(f"partial_element[{index}]={element}")
            click.echo(
                "violated_forbidden_claims="
                f"{len(result.judge_result.violated_forbidden_claims)}"
            )
            for index, claim in enumerate(
                result.judge_result.violated_forbidden_claims,
                start=1,
            ):
                click.echo(f"violated_forbidden_claim[{index}]={claim}")
            click.echo(
                "advisory_missing_elements="
                f"{len(result.judge_result.advisory_missing_elements)}"
            )
            for index, element in enumerate(
                result.judge_result.advisory_missing_elements,
                start=1,
            ):
                click.echo(f"advisory_missing_element[{index}]={element}")
            click.echo(f"judge_validation_errors={len(result.judge_result.validation_errors)}")
            for index, error in enumerate(result.judge_result.validation_errors, start=1):
                click.echo(f"judge_validation_error[{index}]={error}")
            click.echo(f"judge_feedback={result.judge_result.detailed_feedback}")
        else:
            click.echo("judge_status=not_run")
        click.echo("agent_answer:")
        _echo_console_safe(result.final_answer)
    if result.passed is None:
        raise click.ClickException("Outcome evaluation invalid")
    if not result.passed:
        raise click.ClickException("Outcome evaluation failed")


__all__ = ["eval"]
