"""脚本化模型 + 真实存储/工具：验证 runtime 行为而非模型措辞。"""

import copy
import json

import pytest

from loomora.context import SYSTEM_PROMPT, ContextManager, payload_size
from loomora.llm import LLMError, ModelReply, ToolCall
from loomora.runtime import Agent
from loomora.store import SessionBusyError, Store
from loomora.tools import Tool, build_registry


class ScriptedClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if not self.replies:
            raise AssertionError("Unexpected extra model request")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def call(call_id, name, **arguments):
    return ToolCall(call_id, name, json.dumps(arguments, ensure_ascii=False))


def tool_results(messages):
    return [json.loads(message["content"]) for message in messages if message["role"] == "tool"]


def assert_paired_calls(messages):
    pending = set()
    for message in messages:
        if message["role"] == "tool":
            assert message["tool_call_id"] in pending
            pending.remove(message["tool_call_id"])
        else:
            assert not pending, "another message cannot interrupt a tool result group"
            calls = message.get("tool_calls", [])
            pending = {item["id"] for item in calls}
            assert len(pending) == len(calls)
    assert not pending


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "runtime-data")


def test_direct_reply_and_plain_follow_up_receive_previous_conversation(store):
    client = ScriptedClient(ModelReply("好的，记住你在广州。"), ModelReply("你在广州。"))
    agent = Agent(client, store)
    first = agent.run("alice", "chat", "我在广州")
    second = agent.run("alice", "chat", "我在哪个城市？")
    assert (first.status, first.steps) == ("completed", 1)
    assert second.answer == "你在广州。"
    assert first.run_id != second.run_id
    history, schemas = client.requests[1]
    assert {item["function"]["name"] for item in schemas} == {"calculator", "search", "todo"}
    assert {"role": "user", "content": "我在广州"} in history
    assert {"role": "assistant", "content": "好的，记住你在广州。"} in history
    assert history[-1] == {"role": "user", "content": "我在哪个城市？"}
    assert len(store.turns("alice", "chat")) == 2
    assert all(turn["status"] == "completed" for turn in store.turns("alice", "chat"))


def test_tool_follow_up_lists_current_todos_before_completing_real_id(store):
    first_client = ScriptedClient(
        ModelReply(tool_calls=[call("add-1", "todo", action="add", title="写周报")]),
        ModelReply("已添加写周报。"),
    )
    assert Agent(first_client, store).run("alice", "work", "记待办：写周报").status == "completed"
    todo_id = store.list_todos("alice", "work")[0]["id"]
    client = ScriptedClient(
        ModelReply(tool_calls=[call("list-1", "todo", action="list")]),
        ModelReply(tool_calls=[call("complete-1", "todo", action="complete", todo_id=todo_id)]),
        ModelReply("刚才那项已完成。"),
    )
    result = Agent(client, store).run("alice", "work", "把刚才那项标记完成")
    assert result.status == "completed"
    assert result.steps == 3
    assert store.list_todos("alice", "work") == [
        {"id": todo_id, "title": "写周报", "completed": True}
    ]
    list_result = tool_results(client.requests[1][0])[-1]
    assert list_result["data"]["items"][0]["id"] == todo_id
    assert list_result["data"]["items"][0]["completed"] is False
    assert tool_results(client.requests[2][0])[-1]["data"]["completed"] is True
    assert_paired_calls(store.turns("alice", "work")[-1]["messages"])


def test_search_calculator_todo_chain_feeds_each_result_back_to_model(store):
    client = ScriptedClient(
        ModelReply(tool_calls=[call("search-1", "search", query="Agent 循环")]),
        ModelReply(tool_calls=[call("calc-1", "calculator", expression="3 * 20")]),
        ModelReply(tool_calls=[call("todo-1", "todo", action="add", title="学习 Agent 60 分钟")]),
        ModelReply("已参考示例资料计算并记下 60 分钟学习待办。"),
    )
    result = Agent(client, store).run("alice", "study", "查 Agent 资料，算 3 次20分钟，并加入待办")
    assert result.status == "completed"
    assert result.steps == 4
    assert tool_results(client.requests[1][0])[-1]["data"]["mock"] is True
    assert tool_results(client.requests[2][0])[-1]["data"]["result"] == 60
    assert tool_results(client.requests[3][0])[-1]["data"]["title"] == "学习 Agent 60 分钟"
    assert_paired_calls(store.turns("alice", "study")[0]["messages"])


def test_same_user_windows_restore_independent_history_and_todos_after_restart(store):
    for session_id, title in [("window-1", "买伞"), ("window-2", "写周报")]:
        client = ScriptedClient(
            ModelReply(tool_calls=[call(f"add-{session_id}", "todo", action="add", title=title)]),
            ModelReply(f"已记录：{title}"),
        )
        assert Agent(client, store).run("alice", session_id, f"记住{title}").status == "completed"
    reopened = Store(store.directory)
    client = ScriptedClient(
        ModelReply(tool_calls=[call("list-restored", "todo", action="list")]),
        ModelReply("之前记的是买伞。"),
    )
    result = Agent(client, reopened).run("alice", "window-1", "之前记了什么？")
    assert result.status == "completed"
    serialized = json.dumps(client.requests, ensure_ascii=False)
    assert "买伞" in serialized
    assert "写周报" not in serialized
    assert [
        todo["title"] for todo in tool_results(client.requests[-1][0])[-1]["data"]["items"]
    ] == ["买伞"]
    assert reopened.list_todos("alice", "window-2")[0]["title"] == "写周报"


def test_bad_tool_arguments_are_returned_for_model_self_correction(store):
    client = ScriptedClient(
        ModelReply(tool_calls=[ToolCall("bad", "calculator", '{"expression":')]),
        ModelReply(tool_calls=[call("fixed", "calculator", expression="7 * 6")]),
        ModelReply("42"),
    )
    result = Agent(client, store).run("alice", "math", "7乘6")
    assert result.answer == "42"
    assert tool_results(client.requests[1][0])[-1]["error"]["code"] == "invalid_arguments"
    assert tool_results(client.requests[2][0])[-1]["data"]["result"] == 42
    assert_paired_calls(store.turns("alice", "math")[0]["messages"])


def test_unexpected_tool_exception_becomes_safe_result_and_loop_continues(store):
    registry = build_registry()

    def broken_tool(arguments, context):
        raise RuntimeError("private-database-path-and-password")

    registry.register(Tool("broken", "测试失败工具", {"type": "object"}, broken_tool))
    client = ScriptedClient(
        ModelReply(tool_calls=[call("broken-1", "broken")]),
        ModelReply("工具失败了，请稍后重试。"),
    )
    result = Agent(client, store, registry).run("alice", "chat", "调用测试工具")
    assert result.status == "completed"
    assert tool_results(client.requests[1][0])[-1]["error"]["code"] == "tool_error"
    assert "password" not in json.dumps(client.requests)


@pytest.mark.parametrize(
    "error", [LLMError("LLM 请求超时"), RuntimeError("private-internal-secret")]
)
def test_model_failure_keeps_completed_side_effect_and_allows_next_turn(store, error):
    client = ScriptedClient(
        ModelReply(tool_calls=[call("saved", "todo", action="add", title="保留这条待办")]),
        error,
    )
    result = Agent(client, store).run("alice", "work", "添加待办")
    assert result.status == "error"
    assert "private-internal-secret" not in result.answer
    assert len(store.list_todos("alice", "work")) == 1
    assert_paired_calls(store.turns("alice", "work")[0]["messages"])
    next_client = ScriptedClient(ModelReply("可以继续。"))
    assert Agent(next_client, store).run("alice", "work", "继续").status == "completed"
    assert tool_results(next_client.requests[0][0])[0]["data"]["title"] == "保留这条待办"


def test_max_steps_executes_final_batch_and_keeps_all_call_result_pairs(store):
    client = ScriptedClient(
        *[
            ModelReply(
                tool_calls=[
                    call(f"calc-{step}", "calculator", expression=f"{step} + 1"),
                    call(f"todo-{step}", "todo", action="add", title=f"任务 {step}"),
                ]
            )
            for step in range(2)
        ]
    )
    result = Agent(client, store, max_steps=2).run("alice", "loop", "执行多个步骤")
    assert result.status == "max_steps"
    assert result.steps == 2
    assert len(client.requests) == 2
    assert len(store.list_todos("alice", "loop")) == 2
    messages = store.turns("alice", "loop")[0]["messages"]
    assert len(tool_results(messages)) == 4
    assert_paired_calls(messages)
    assert messages[-1]["content"] == result.answer


def test_busy_session_rejects_new_turn_before_calling_model(store):
    client = ScriptedClient(ModelReply("不应执行"))
    with store.session("alice", "work"):
        with pytest.raises(SessionBusyError):
            Agent(client, Store(store.directory)).run("alice", "work", "新消息")
    assert client.requests == []
    assert store.turns("alice", "work") == []


def test_trace_records_decisions_without_raw_arguments_or_reasoning(store):
    private_title = "private-title-marker-731"
    private_reasoning = "private-reasoning-marker-952"
    client = ScriptedClient(
        ModelReply(
            tool_calls=[call("trace-call", "todo", action="add", title=private_title)],
            reasoning=private_reasoning,
        ),
        ModelReply("已保存。", reasoning=private_reasoning),
    )
    result = Agent(client, store).run("alice", "private", "private-user-message-641")
    assert result.status == "completed"
    traces = store.traces("alice", "private", limit=100)
    serialized = json.dumps(traces)
    for private_value in (private_title, private_reasoning, "private-user-message-641"):
        assert private_value not in serialized
    assert [item["event"] for item in traces] == [
        "run_started",
        "llm_completed",
        "tool_started",
        "tool_finished",
        "llm_completed",
        "run_finished",
    ]
    assert all(item["metadata"]["run_id"] == result.run_id for item in traces)
    assert traces[1]["metadata"]["has_reasoning"] is True
    # 协议历史仍需保留模型返回的 reasoning_content，trace 与历史职责不同。
    assert client.requests[1][0][-2]["reasoning_content"] == private_reasoning


def test_keyboard_interrupt_closes_pending_calls_without_replaying_side_effects(store):
    registry = build_registry()

    def interrupted_tool(arguments, context):
        context.store.add_todo(context.user_id, context.session_id, "已写入但未返回")
        raise KeyboardInterrupt

    registry.register(Tool("interrupt", "测试中断", {"type": "object"}, interrupted_tool))
    client = ScriptedClient(
        ModelReply(
            tool_calls=[
                call("before", "calculator", expression="2+2"),
                call("interrupted", "interrupt"),
                call("never-run", "todo", action="add", title="不应该写入"),
            ]
        )
    )
    result = Agent(client, store, registry).run("alice", "work", "执行工具批次")
    assert result.status == "cancelled"
    assert [todo["title"] for todo in store.list_todos("alice", "work")] == ["已写入但未返回"]
    turn = store.turns("alice", "work")[0]
    assert turn["status"] == "completed"
    results = tool_results(turn["messages"])
    assert results[0]["data"]["result"] == 4
    assert [item["error"]["code"] for item in results[1:]] == ["interrupted", "interrupted"]
    assert_paired_calls(turn["messages"])
    next_client = ScriptedClient(ModelReply("已查询，请继续。"))
    assert Agent(next_client, store).run("alice", "work", "继续").status == "completed"
    assert len(store.list_todos("alice", "work")) == 1


def test_restart_recovers_running_turn_without_repeating_unknown_operations(store):
    pending = ModelReply(tool_calls=[call("lost-result", "todo", action="add", title="崩溃前待办")])
    with store.session("alice", "work"):
        interrupted_id = store.start_turn(
            "alice",
            "work",
            [
                {"role": "user", "content": "添加待办"},
                pending.as_message(),
            ],
        )
        store.add_todo("alice", "work", "崩溃前待办")
    client = ScriptedClient(
        ModelReply(tool_calls=[call("inspect", "todo", action="list")]),
        ModelReply("上次操作已经写入，现在只有一条。"),
    )
    reopened = Store(store.directory)
    result = Agent(client, reopened).run("alice", "work", "上次成功了吗？")
    assert result.status == "completed"
    recovered = reopened.turns("alice", "work")[0]
    assert recovered["id"] == interrupted_id
    assert recovered["status"] == "completed"
    assert tool_results(recovered["messages"])[0]["error"]["code"] == "interrupted"
    assert_paired_calls(recovered["messages"])
    assert len(reopened.list_todos("alice", "work")) == 1
    assert len(tool_results(client.requests[1][0])[-1]["data"]["items"]) == 1
    assert any(item["event"] == "turn_recovered" for item in reopened.traces("alice", "work"))
    next_client = ScriptedClient(ModelReply("继续。"))
    Agent(next_client, reopened).run("alice", "work", "继续")
    assert (
        sum(item["event"] == "turn_recovered" for item in reopened.traces("alice", "work", 100))
        == 1
    )


def test_reused_call_id_stops_before_second_side_effect(store):
    client = ScriptedClient(
        ModelReply(tool_calls=[call("duplicate", "todo", action="add", title="首次操作")]),
        ModelReply(tool_calls=[call("duplicate", "todo", action="add", title="重复ID不得执行")]),
    )
    result = Agent(client, store).run("alice", "work", "添加两项")
    assert result.status == "error"
    assert "ID" in result.answer
    assert [todo["title"] for todo in store.list_todos("alice", "work")] == ["首次操作"]
    messages = store.turns("alice", "work")[0]["messages"]
    assert_paired_calls(messages)
    assert len(tool_results(messages)) == 1


@pytest.mark.parametrize("text", ["", "  ", "x" * 4001, None])
def test_invalid_user_input_does_not_create_a_turn_or_call_model(store, text):
    client = ScriptedClient()
    with pytest.raises(ValueError):
        Agent(client, store).run("alice", "work", text)
    assert client.requests == []
    assert store.turns("alice", "work") == []


@pytest.mark.parametrize("history_count", [20, 200])
def test_compacted_session_continues_tool_loop_with_one_summary_request(store, history_count):
    # 默认预算下，长摘要会把首个请求填到约 32000 字节；工具回填仍应能继续。
    with store.session("alice", "long-chat"):
        for _ in range(history_count):
            messages = [
                {"role": "user", "content": "旧" * 2000},
                {"role": "assistant", "content": "已了解。"},
            ]
            turn_id = store.start_turn("alice", "long-chat", messages)
            store.save_turn(turn_id, messages, "completed")
    original_history = store.turns("alice", "long-chat")
    client = ScriptedClient(
        ModelReply("事实" * 1000),
        ModelReply(tool_calls=[call("calculate", "calculator", expression="1+1")]),
        ModelReply(tool_calls=[call("remember", "todo", action="add", title="记录计算结果 2")]),
        ModelReply("结果是 2，已记录待办。"),
    )
    result = Agent(client, Store(store.directory)).run("alice", "long-chat", "计算1+1并记待办")
    assert result.status == "completed"
    assert result.steps == 3
    assert sum(not tools for _, tools in client.requests) == 1
    assert len(client.requests) == 4
    assert all(payload_size(messages, tools) <= 32_000 for messages, tools in client.requests)
    assert tool_results(client.requests[2][0])[-1]["data"]["result"] == 2
    assert tool_results(client.requests[3][0])[-1]["data"]["title"] == "记录计算结果 2"
    for messages, schemas in client.requests:
        if schemas:
            assert_paired_calls(messages)
    turns = store.turns("alice", "long-chat")
    assert turns[:-1] == original_history
    assert turns[-1]["status"] == "completed"
    assert_paired_calls(turns[-1]["messages"])
    assert history_count - 4 <= store.memory("alice", "long-chat")[1] <= history_count


def test_overflow_after_first_tool_uses_local_compaction_without_model_summary(store):
    registry = build_registry()
    registry.register(
        Tool("sample", "返回示例资料", {"type": "object"}, lambda args, ctx: {"text": "结果" * 350})
    )
    with store.session("alice", "chat"):
        for _ in range(4):
            messages = [
                {"role": "user", "content": "历史" * 200},
                {"role": "assistant", "content": "已了解。"},
            ]
            turn_id = store.start_turn("alice", "chat", messages)
            store.save_turn(turn_id, messages, "completed")
    incoming = [{"role": "user", "content": "读取资料"}]
    initial = [{"role": "system", "content": SYSTEM_PROMPT}]
    initial += [message for turn in store.turns("alice", "chat") for message in turn["messages"]]
    initial += incoming
    budget = payload_size(initial, registry.schemas()) + 50
    client = ScriptedClient(
        ModelReply(tool_calls=[call("sample-call", "sample")]),
        ModelReply("已读取。"),
    )
    result = Agent(
        client, store, registry, context=ContextManager(client, max_context_bytes=budget)
    ).run("alice", "chat", incoming[0]["content"])
    assert result.status == "completed"
    assert len(client.requests) == 2
    assert all(tools for _, tools in client.requests)
    assert all(payload_size(messages, tools) <= budget for messages, tools in client.requests)
    assert "本轮压缩预算受限" in store.memory("alice", "chat")[0]
    assert tool_results(client.requests[1][0])[-1]["data"]["text"] == "结果" * 350
    assert_paired_calls(client.requests[1][0])


def test_current_tool_result_itself_over_budget_stops_without_truncating_protocol(store):
    registry = build_registry()
    registry.register(
        Tool("large", "返回测试资料", {"type": "object"}, lambda args, ctx: {"text": "x" * 7000})
    )
    incoming = [{"role": "user", "content": "读取资料"}]
    budget = (
        payload_size([{"role": "system", "content": SYSTEM_PROMPT}] + incoming, registry.schemas())
        + 500
    )
    client = ScriptedClient(ModelReply(tool_calls=[call("large-call", "large")]))
    result = Agent(
        client, store, registry, context=ContextManager(client, max_context_bytes=budget)
    ).run("alice", "chat", incoming[0]["content"])
    assert result.status == "error"
    assert "当前请求或工具结果过长" in result.answer
    assert len(client.requests) == 1
    messages = store.turns("alice", "chat")[0]["messages"]
    assert tool_results(messages)[0]["data"]["text"] == "x" * 7000
    assert_paired_calls(messages)
    assert store.memory("alice", "chat") == ("", 0)
