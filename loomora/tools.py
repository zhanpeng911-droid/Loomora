"""工具注册、参数校验与三个本地工具；不依赖 Agent 框架。"""

from __future__ import annotations

import ast
import copy
import json
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable

from jsonschema import Draft202012Validator, ValidationError


@dataclass(frozen=True)
class ToolContext:
    store: Any
    user_id: str
    session_id: str


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict, ToolContext], Any]


class ToolInputError(ValueError):
    """可安全返回给模型、用于修正下一次调用的错误。"""


def _error(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._validators: dict[str, Draft202012Validator] = {}

    def register(self, tool: Tool) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool.name):
            raise ValueError("Invalid tool name")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        if not tool.description.strip() or not callable(tool.handler):
            raise ValueError("A tool needs a description and callable handler")
        if not isinstance(tool.parameters, dict) or tool.parameters.get("type") != "object":
            raise ValueError("Tool parameters must use an object schema")
        schema = copy.deepcopy(tool.parameters)
        Draft202012Validator.check_schema(schema)
        self._tools[tool.name] = Tool(tool.name, tool.description, schema, tool.handler)
        self._validators[tool.name] = Draft202012Validator(schema)

    def schemas(self) -> list[dict]:
        # 返回副本，调用方无法意外改变已经注册的校验规则。
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": copy.deepcopy(tool.parameters),
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, arguments: str, context: ToolContext) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return _error("unknown_tool", "工具不存在，请使用已注册的工具名称。")
        try:
            if not isinstance(arguments, str) or len(arguments) > 10_000:
                raise ValueError("Invalid argument size")
            params = json.loads(
                arguments, parse_constant=_reject_constant, object_pairs_hook=_unique_object
            )
            # json.loads 会把 1e999 转为 inf，因此还需检查序列化后的有限性。
            json.dumps(params, allow_nan=False)
            self._validators[name].validate(params)
        except (ValueError, TypeError, RecursionError, ValidationError):
            return _error(
                "invalid_arguments",
                "参数不符合工具 Schema；请检查 JSON、必填字段、类型及取值范围。",
            )
        try:
            result = tool.handler(params, context)
            json.dumps(result, allow_nan=False)
            return {"ok": True, "data": result}
        except ToolInputError as exc:
            return _error("invalid_input", str(exc))
        except Exception:
            # 不把异常中的本地路径、凭据或数据库细节放入模型上下文。
            return _error("tool_error", "工具执行失败；请修正输入或稍后重试。")


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _bounded_number(value: Any) -> int | float:
    if type(value) not in (int, float):
        raise ToolInputError("仅支持实数运算。")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolInputError("计算结果必须是有限实数。")
    if abs(value) > 1e100:
        raise ToolInputError("数值过大；绝对值上限为 1e100。")
    return value


def _calculate(params: dict, context: ToolContext) -> dict:
    expression = params["expression"].strip()
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ToolInputError("表达式过于复杂。")

        def visit(node: ast.AST) -> int | float:
            if isinstance(node, ast.Constant):
                return _bounded_number(node.value)
            if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
                value = visit(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 100:
                    raise ToolInputError("幂指数的绝对值不能超过 100。")
                return _bounded_number(_BINARY_OPERATORS[type(node.op)](left, right))
            raise ToolInputError("仅支持数字、括号和 + - * / // % ** 运算。")

        return {"expression": expression, "result": visit(tree.body)}
    except (SyntaxError, OverflowError, ZeroDivisionError, RecursionError):
        raise ToolInputError("表达式无效、除数为零或计算溢出。") from None


_DOCUMENTS = (
    {
        "id": "agent-loop",
        "title": "Agent 基本循环 / Agent loop",
        "text": "Agent 接收用户输入，由模型选择直接回答或调用工具；工具结果写回上下文后继续判断。",
        "keywords": ("agent", "代理", "loop", "循环", "工具", "tool"),
    },
    {
        "id": "session-memory",
        "title": "Session 与记忆 / Session memory",
        "text": "不同 session 的对话与待办独立保存。上下文可由近期完整对话和较早对话的摘要组成。",
        "keywords": ("session", "memory", "会话", "记忆", "上下文", "context", "压缩"),
    },
    {
        "id": "todo-planning",
        "title": "待办管理 / Todo planning",
        "text": "待办有唯一编号、标题和完成状态；更新前先读取当前会话的列表，避免猜测编号。",
        "keywords": ("todo", "task", "待办", "任务", "计划", "planning"),
    },
)


def _search(params: dict, context: ToolContext) -> dict:
    query = params["query"].strip().lower()
    if not query:
        raise ToolInputError("搜索词不能为空。")
    ranked = []
    for document in _DOCUMENTS:
        score = sum(keyword in query for keyword in document["keywords"])
        if score:
            ranked.append((score, document))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return {
        "mock": True,
        "source": "内置静态示例资料，不是实时网络搜索",
        "results": [
            {key: document[key] for key in ("id", "title", "text")} for _, document in ranked[:3]
        ],
    }


def _todo(params: dict, context: ToolContext) -> Any:
    scope = (context.user_id, context.session_id)
    if params["action"] == "add":
        title = params["title"].strip()
        if not title:
            raise ToolInputError("待办标题不能为空。")
        return context.store.add_todo(*scope, title)
    if params["action"] == "list":
        # 限制每页条数而非截断标题，保证分页后仍能取到末尾待办的完整内容。
        after_id = int(params.get("after_id", 0))
        limit = int(params.get("limit", 5))
        remaining = [item for item in context.store.list_todos(*scope) if item["id"] > after_id]
        items = remaining[:limit]
        return {
            "items": items,
            "next_cursor": items[-1]["id"] if len(remaining) > limit else None,
        }
    result = context.store.complete_todo(*scope, params["todo_id"])
    if result is None:
        raise ToolInputError("当前会话中没有这个待办，请先 list 查找正确编号。")
    return result


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="计算实数四则运算，支持括号、//、%、**；需要精确算术结果时使用。",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {"expression": {"type": "string", "minLength": 1, "maxLength": 512}},
                "required": ["expression"],
            },
            handler=_calculate,
        )
    )
    registry.register(
        Tool(
            name="search",
            description="搜索内置 Agent/session/todo 示例资料（mock），最多返回三条；不提供实时天气或互联网事实。",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
                "required": ["query"],
            },
            handler=_search,
        )
    )
    registry.register(
        Tool(
            name="todo",
            description=(
                "管理当前会话的待办：add 添加、list 查看、complete 完成。"
                "list 每页最多5项，返回 items 与 next_cursor；next_cursor 非空时，"
                "以其值作为 after_id 继续分页。指代‘刚才那个’时先 list 确认真实编号，不要猜测。"
            ),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {"type": "string", "enum": ["add", "list", "complete"]},
                    "title": {"type": "string", "minLength": 1, "maxLength": 200},
                    "todo_id": {"type": "integer", "minimum": 1},
                    "after_id": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "仅 list 使用；读取编号大于该值的待办，默认0。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 5,
                        "description": "仅 list 使用；每页条数，默认5。",
                    },
                },
                "required": ["action"],
                "allOf": [
                    {
                        "if": {"properties": {"action": {"const": "add"}}},
                        "then": {
                            "required": ["title"],
                            "not": {
                                "anyOf": [
                                    {"required": ["todo_id"]},
                                    {"required": ["after_id"]},
                                    {"required": ["limit"]},
                                ]
                            },
                        },
                    },
                    {
                        "if": {"properties": {"action": {"const": "list"}}},
                        "then": {
                            "not": {"anyOf": [{"required": ["title"]}, {"required": ["todo_id"]}]}
                        },
                    },
                    {
                        "if": {"properties": {"action": {"const": "complete"}}},
                        "then": {
                            "required": ["todo_id"],
                            "not": {
                                "anyOf": [
                                    {"required": ["title"]},
                                    {"required": ["after_id"]},
                                    {"required": ["limit"]},
                                ]
                            },
                        },
                    },
                ],
            },
            handler=_todo,
        )
    )
    return registry
