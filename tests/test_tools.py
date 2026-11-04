import json

import pytest
from jsonschema import SchemaError

from loomora.tools import Tool, ToolContext, ToolRegistry, build_registry


class FakeStore:
    def __init__(self):
        self.todos = {}

    def add_todo(self, user_id, session_id, title):
        rows = self.todos.setdefault((user_id, session_id), [])
        for row in rows:
            if row["title"] == title and not row["completed"]:
                return row.copy()
        row = {"id": len(rows) + 1, "title": title, "completed": False}
        rows.append(row)
        return row.copy()

    def list_todos(self, user_id, session_id):
        return [row.copy() for row in self.todos.get((user_id, session_id), [])]

    def complete_todo(self, user_id, session_id, todo_id):
        for row in self.todos.get((user_id, session_id), []):
            if row["id"] == todo_id:
                row["completed"] = True
                return row.copy()
        return None


@pytest.fixture
def context():
    return ToolContext(FakeStore(), "alice", "window-1")


def invoke(registry, context, name, **params):
    return registry.execute(name, json.dumps(params), context)


def test_registry_exposes_three_independent_schema_copies(context):
    registry = build_registry()
    schemas = registry.schemas()
    assert [item["function"]["name"] for item in schemas] == ["calculator", "search", "todo"]
    schemas[0]["function"]["parameters"]["required"].clear()
    assert registry.execute("calculator", "{}", context)["error"]["code"] == "invalid_arguments"


def test_registration_rejects_duplicates_invalid_names_and_invalid_schemas():
    registry = ToolRegistry()
    tool = Tool("echo", "echo", {"type": "object"}, lambda params, ctx: params)
    registry.register(tool)
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(tool)
    with pytest.raises(ValueError, match="name"):
        registry.register(Tool("bad name", "echo", {"type": "object"}, tool.handler))
    with pytest.raises(ValueError, match="object"):
        registry.register(Tool("bad_schema", "echo", {"type": "array"}, tool.handler))
    with pytest.raises(SchemaError):
        registry.register(
            Tool(
                "bad_type",
                "echo",
                {"type": "object", "properties": {"x": {"type": "oops"}}},
                tool.handler,
            )
        )


@pytest.mark.parametrize(
    "arguments",
    [
        "not json",
        "[]",
        "null",
        '{"expression":12}',
        "{}",
        '{"expression":"2+2","extra":1}',
        '{"expression":NaN}',
        '{"expression":Infinity}',
        '{"expression":1e999}',
        "x" * 10_001,
        '{"expression":"1+1","expression":"2+2"}',
    ],
)
def test_bad_arguments_are_recoverable(context, arguments):
    result = build_registry().execute("calculator", arguments, context)
    assert result["error"]["code"] == "invalid_arguments"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(12 + 3) * 4", 60),
        ("-2 ** 3 + +1", -7),
        ("7 // 2 + 7 % 2", 4),
        ("1 / 4", 0.25),
        ("2 ** -3", 0.125),
    ],
)
def test_calculator_evaluates_arithmetic(context, expression, expected):
    result = invoke(build_registry(), context, "calculator", expression=expression)
    assert result["ok"]
    assert result["data"]["result"] == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "[1][0]",
        "True",
        "'abc'",
        "x + 1",
        "1 / 0",
        "2 ** 101",
        "1e101",
        "1e99 * 1e99",
        "(-1) ** 0.5",
        "1e999",
        "1 +",
        " ",
        "+".join(["1"] * 80),
        "(1).__class__",
    ],
)
def test_calculator_rejects_code_and_unbounded_arithmetic(context, expression):
    result = invoke(build_registry(), context, "calculator", expression=expression)
    assert not result["ok"]
    assert result["error"]["code"] == "invalid_input"


def test_search_is_explicitly_mock_and_never_invents_missing_results(context):
    registry = build_registry()
    result = invoke(registry, context, "search", query="Agent session todo")
    assert result["data"]["mock"] is True
    assert len(result["data"]["results"]) == 3
    assert invoke(registry, context, "search", query="今天广州天气")["data"]["results"] == []
    assert not invoke(registry, context, "search", query="  ")["ok"]


@pytest.mark.parametrize(
    "params",
    [
        {"action": "add"},
        {"action": "add", "title": "x", "todo_id": 1},
        {"action": "list", "title": "x"},
        {"action": "complete"},
        {"action": "complete", "todo_id": True},
        {"action": "complete", "todo_id": 0},
        {"action": "complete", "todo_id": 1, "title": "x"},
        {"action": "delete"},
        {"action": "add", "title": "x" * 201},
        {"action": "add", "title": "x", "after_id": 0},
        {"action": "add", "title": "x", "limit": 1},
        {"action": "complete", "todo_id": 1, "after_id": 0},
        {"action": "complete", "todo_id": 1, "limit": 1},
        {"action": "list", "after_id": -1},
        {"action": "list", "after_id": True},
        {"action": "list", "after_id": "5"},
        {"action": "list", "limit": 0},
        {"action": "list", "limit": 6},
        {"action": "list", "limit": 1.5},
        {"action": "list", "limit": True},
    ],
)
def test_todo_schema_checks_action_specific_parameters(context, params):
    assert (
        invoke(build_registry(), context, "todo", **params)["error"]["code"] == "invalid_arguments"
    )


def test_todo_scope_and_follow_up(context):
    registry = build_registry()
    added = invoke(registry, context, "todo", action="add", title="  写周报  ")
    assert added["data"]["title"] == "写周报"
    assert (
        invoke(registry, context, "todo", action="add", title="写周报")["data"]["id"]
        == added["data"]["id"]
    )
    other_window = ToolContext(context.store, "alice", "window-2")
    other_user = ToolContext(context.store, "bob", "window-1")
    for other in (other_window, other_user):
        assert invoke(registry, other, "todo", action="list")["data"] == {
            "items": [],
            "next_cursor": None,
        }
        assert not invoke(registry, other, "todo", action="complete", todo_id=added["data"]["id"])[
            "ok"
        ]
    listed = invoke(registry, context, "todo", action="list")["data"]["items"]
    done = invoke(registry, context, "todo", action="complete", todo_id=listed[0]["id"])
    assert done["data"]["completed"]
    assert not invoke(registry, context, "todo", action="add", title="  ")["ok"]


def test_todo_pagination_reaches_the_last_full_title_without_oversized_results(context):
    registry = build_registry()
    expected = []
    for index in range(100):
        title = f"{index:03}" + "\U0001f680" * 197
        expected.append(invoke(registry, context, "todo", action="add", title=title)["data"])
    after_id = 0
    seen = []
    for page_number in range(20):
        result = invoke(registry, context, "todo", action="list", after_id=after_id)
        assert result["ok"]
        assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) < 8000
        page = result["data"]
        assert len(page["items"]) == 5
        seen.extend(page["items"])
        if page_number == 19:
            assert page["next_cursor"] is None
        else:
            assert page["next_cursor"] == page["items"][-1]["id"]
            after_id = page["next_cursor"]
    assert seen == expected
    assert seen[-1]["title"] == "099" + "\U0001f680" * 197
    empty = invoke(registry, context, "todo", action="list", after_id=expected[-1]["id"])
    assert empty["data"] == {"items": [], "next_cursor": None}


def test_todo_custom_page_size_and_cursor_remain_within_session(context):
    registry = build_registry()
    for index in range(4):
        invoke(registry, context, "todo", action="add", title=f"任务 {index}")
    page = invoke(registry, context, "todo", action="list", after_id=1, limit=2)["data"]
    assert [item["id"] for item in page["items"]] == [2, 3]
    assert page["next_cursor"] == 3
    last = invoke(registry, context, "todo", action="list", after_id=page["next_cursor"], limit=2)[
        "data"
    ]
    assert [item["id"] for item in last["items"]] == [4]
    assert last["next_cursor"] is None
    for user_id, session_id in [("alice", "window-2"), ("bob", "window-1")]:
        other = ToolContext(context.store, user_id, session_id)
        isolated = invoke(registry, other, "todo", action="list", after_id=1, limit=2)
        assert isolated["data"] == {"items": [], "next_cursor": None}


def test_todo_pagination_accepts_integral_json_numbers_allowed_by_json_schema(context):
    registry = build_registry()
    invoke(registry, context, "todo", action="add", title="验证整数语义")
    result = invoke(registry, context, "todo", action="list", after_id=0.0, limit=1.0)
    assert result["ok"]
    assert result["data"]["items"][0]["title"] == "验证整数语义"
    assert result["data"]["next_cursor"] is None


def test_unregistered_tools_and_unexpected_exceptions_are_safe(context):
    registry = build_registry()
    assert registry.execute("shell", "{}", context)["error"]["code"] == "unknown_tool"

    def fail(params, ctx):
        raise RuntimeError("secret-api-key local-path")

    registry.register(Tool("broken", "test", {"type": "object"}, fail))
    result = registry.execute("broken", "{}", context)
    assert result["error"]["code"] == "tool_error"
    assert "secret" not in json.dumps(result)


def test_non_finite_custom_results_are_not_reported_as_success(context):
    registry = ToolRegistry()
    registry.register(
        Tool("infinite", "test", {"type": "object"}, lambda params, ctx: float("inf"))
    )
    assert registry.execute("infinite", "{}", context)["error"]["code"] == "tool_error"
