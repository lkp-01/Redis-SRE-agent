"""RediSearch 查询片段工具。

RediSearch 引擎有其内部的语法规则
RediSearch 的 TAG 查询会把空格、花括号、星号、问号、竖线等字符当成查询语法。
如果用户输入直接拼进查询字符串，轻则查不到数据，重则把用户输入解释成另一段查询。
所以资源层必须先转义用户输入，再拼接查询表达式。

核心目的只有一个：帮后端开发人员安全、优雅地生成给 Redis（具体是 RediSearch 模块）看的查询语句
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#里面存了所有 RediSearch 认为是“特殊指令”的符号
_SPECIAL_CHARS = set(',.<>{}[]\\"\\\':;!@#$%^&*()-+=~/ ?|')


class FilterExpression:
    """封装一段 RediSearch 查询过滤语句的工具类"""

    def __init__(self, expression: Any) -> None:  #接收任意内容（字符串、其他 FilterExpression 对象、数字等），统一转成字符串存到 self.expression
        self.expression = str(expression)

    #当两个对象用 & 连接时自动执行
    #当你写 A & B，底层等价于 A.__and__(B)
    def __and__(self, other: Any) -> "FilterExpression": 
        return FilterExpression(f"({self}) ({other})")

    #当对象被转为字符串时自动执行，比如str(对象),print(对象), f-string里的对象
    def __str__(self) -> str:
        return self.expression

# 你可以直接在代码里写 Tag("status") == "active"，
# 它一执行，就会自动调用上面的翻译官，变成 @status:{active}。这就让写查询条件像写普通判断语句一样直观。
class Tag:
    """TAG 字段过滤器插槽。"""

    def __init__(self, field_name: str) -> None: #创建 Tag 对象时，只需要传入字段名，存起来备用。
        self.field_name = field_name

    def __eq__(self, value: Any) -> FilterExpression:  # 重载 == 运算符，把 Tag(字段) == 值 转换成 RediSearch TAG 精确匹配的过滤表达式
        return tag_equals_expression(self.field_name, value)

#只携带过滤表达式的计数查询。
@dataclass
class CountQuery:

    filter_expression: Any = "*"

    def __str__(self) -> str:
        return str(self.filter_expression)

# 它不仅装了上面拼好的过滤条件（filter_expression），
# 还负责装分页信息（从哪条开始查、查几条）、排序规则（按什么字段排、升序还是降序）、以及你需要返回哪些字段。
class FilterQuery:

    def __init__(
        self,
        filter_expression: Any = "*",
        return_fields: list[str] | None = None,
        num_results: int = 100,
    ) -> None:
        self.filter_expression = filter_expression
        self.return_fields = return_fields or []
        self.num_results = num_results
        self.offset = 0
        self.limit = num_results
        self.sort_field: str | None = None
        self.sort_asc = True

    def set_filter(self, filter_expression: Any) -> "FilterQuery":
        self.filter_expression = filter_expression
        return self

    def sort_by(self, field: str, asc: bool = True) -> "FilterQuery":
        self.sort_field = field
        self.sort_asc = asc
        return self

    def paging(self, offset: int, limit: int) -> "FilterQuery":
        self.offset = offset
        self.limit = limit
        return self

    def __str__(self) -> str:
        return str(self.filter_expression)

#当用户输入一段文字时，这个函数会挨个检查有没有上面黑名单里的符号。如果有，就在它前面加一个反斜杠 \。
def escape_redisearch_query_value(value: Any) -> str:
    """转义用户输入，使其可以安全放进 RediSearch TAG 查询。"""

    text = str(value or "")
    return "".join(f"\\{char}" if char in _SPECIAL_CHARS else char for char in text)

#生成“包含”条件的工具。和上面类似，但在值两边加了星号 *，做模糊匹配。
def tag_equals_expression(field_name: str, value: Any) -> FilterExpression:
    """构造 TAG 精确匹配表达式。"""

    return FilterExpression(f"@{field_name}:{{{escape_redisearch_query_value(value)}}}")


def tag_contains_expression(field_name: str, value: Any) -> FilterExpression:
    """构造 TAG 包含匹配表达式。"""

    return FilterExpression(f"@{field_name}:{{*{escape_redisearch_query_value(value)}*}}")
