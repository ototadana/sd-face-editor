import operator
from typing import Dict

from lark import Lark, Tree

query_grammar = """
    ?start: expression

    ?expression: and_expression ("|" and_expression)*
    ?and_expression: sub_expression ("&" sub_expression)*
    ?sub_expression: condition | "(" expression ")"

    condition: field OPERATOR value

    field: CNAME
    value: CNAME | NUMBER

    %import common.CNAME
    %import common.NUMBER

    OPERATOR: "=" | "<" | ">" | "<=" | ">=" | "!=" | "~=" | "*=" | "=*" | "~*" | "*~"
"""

query_parser = Lark(query_grammar, start="start")


def starts_with(a, b):
    return a.startswith(b)


def ends_with(a, b):
    return a.endswith(b)


def contains(a, b):
    return b in a


def not_contains(a, b):
    return b not in a


operator_mapping = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "=": operator.eq,
    "!=": operator.ne,
    "~=": contains,
    "*=": starts_with,
    "=*": ends_with,
    "~*": not_contains,
}


def split_condition(condition_parts):
    field_name = condition_parts[0].children[0].value
    operator = condition_parts[1].value
    value = condition_parts[2].children[0].value
    return field_name, operator, value


def evaluate_condition(condition, attributes):
    field_name, operator, value = split_condition(condition.children)
    if field_name in attributes:
        attr_value = attributes[field_name]
        try:
            attr_value = float(attr_value)
            value = float(value)
        except ValueError:
            attr_value = str(attr_value).lower()
            value = str(value).lower()
        return operator_mapping[operator](attr_value, value)
    else:
        return False


def evaluate_and_expression(and_expression, attributes):
    return all(evaluate_expression(child, attributes) for child in and_expression.children)


def evaluate_expression(expression, attributes):
    if isinstance(expression, Tree) and expression.data == "condition":
        return evaluate_condition(expression, attributes)
    elif isinstance(expression, Tree) and expression.data == "and_expression":
        return evaluate_and_expression(expression, attributes)
    else:
        return any(evaluate_expression(child, attributes) for child in expression.children)


def evaluate(query: str, attributes: Dict[str, str]) -> bool:
    tree = query_parser.parse(query)
    return evaluate_expression(tree, attributes)


def validate(query: str):
    return evaluate(query, {})
