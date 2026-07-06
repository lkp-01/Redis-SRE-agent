"""RediSearch 转义测试。"""

from redis_sre_agent.core.redisearch import (
    escape_redisearch_query_value,
    tag_contains_expression,
    tag_equals_expression,
)


def test_escape_redisearch_query_value_escapes_special_characters() -> None:
    assert escape_redisearch_query_value("prod cache/{tenant}:v1?") == (
        r"prod\ cache\/\{tenant\}\:v1\?"
    )
    assert escape_redisearch_query_value("foo|bar/baz[prod]") == (
        r"foo\|bar\/baz\[prod\]"
    )
    assert escape_redisearch_query_value("*literal*") == r"\*literal\*"


def test_tag_expressions_keep_wildcards_outside_user_text() -> None:
    assert str(tag_equals_expression("name", "foo|bar")) == r"@name:{foo\|bar}"
    assert str(tag_contains_expression("name", "*prod/cache?")) == (
        r"@name:{*\*prod\/cache\?*}"
    )
