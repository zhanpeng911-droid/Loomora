import copy
import json

import pytest

from loomora.context import SYSTEM_PROMPT, ContextLimitError, ContextManager, payload_size
from loomora.llm import LLMError, ModelReply, ToolCall


class FakeStore:
    def __init__(self, turns=None, summary="", through_id=0):
        self.records = turns or []
        self.saved_summary = summary
        self.through_id = through_id
        self.saved = []
        self.events = []
        self.scopes = []

    def memory(self, user_id, session_id):
        self.scopes.append((user_id, session_id))
        return self.saved_summary, self.through_id

    def turns(self, user_id, session_id, after_id=0):
        self.scopes.append((user_id, session_id))
        return [turn for turn in self.records if turn["id"] > after_id]

    def save_memory(self, user_id, session_id, summary, through_id):
        self.scopes.append((user_id, session_id))
        self.saved_summary, self.through_id = summary, through_id
        self.saved.append((summary, through_id))

    def trace(self, user_id, session_id, event, **metadata):
        self.events.append((event, metadata))


class FakeClient:
    def __init__(self, reply=None):
        self.reply = reply or ModelReply("用户叫小林，项目预算 500 元，待办 7 已完成。")
        self.requests = []

    def complete(self, messages, schemas):
        self.requests.append((copy.deepcopy(messages), schemas))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def turn(index, content="讨论项目计划", status="completed"):
    return {
        "id": index,
        "status": status,
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "已了解。"},
        ],
    }


def tool_turn(index):
    return {
        "id": index,
        "status": "completed",
        "messages": [
            {"role": "user", "content": "算一下 2+3"},
            ModelReply(
                tool_calls=[ToolCall("call_1", "calculator", '{"expression":"2+3"}')]
            ).as_message(),
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true,"data":5}'},
            {"role": "assistant", "content": "结果是 5。"},
        ],
    }


INCOMING = [{"role": "user", "content": "继续刚才的话题。"}]


def test_small_history_keeps_plain_follow_up_without_summary_call():
    client = FakeClient()
    store = FakeStore([turn(1, "我叫小林。")])
    messages = ContextManager(client).prepare(store, "u", "s", INCOMING, [])
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1:] == store.records[0]["messages"] + INCOMING
    assert client.requests == [] and store.saved == []
    assert store.scopes == [("u", "s"), ("u", "s")]


def test_tool_follow_up_keeps_matching_call_and_result():
    store = FakeStore([tool_turn(1)])
    messages = ContextManager(FakeClient()).prepare(store, "u", "s", INCOMING, [])
    assert messages[1:-1] == store.records[0]["messages"]
    assert messages[2]["tool_calls"][0]["id"] == messages[3]["tool_call_id"]


def test_only_completed_turns_are_recalled():
    store = FakeStore([turn(1), turn(2, "不应进入上下文", "failed")])
    messages = ContextManager(FakeClient()).prepare(store, "u", "s", INCOMING, [])
    assert "不应进入上下文" not in json.dumps(messages, ensure_ascii=False)


def test_200_turns_are_compacted_once_and_checkpoint_persists():
    client = FakeClient()
    store = FakeStore([turn(i, "项目背景" * 40) for i in range(1, 201)])
    original = copy.deepcopy(store.records)
    manager = ContextManager(client, max_context_bytes=12_000)
    messages = manager.prepare(store, "u", "s", INCOMING, [])
    assert len(client.requests) == 1
    assert store.through_id == 196
    assert messages[-9:-1] == [m for t in store.records[-4:] for m in t["messages"]]
    assert payload_size(messages, []) <= 12_000
    assert store.records == original
    assert store.events[0][0] == "context_compacted"
    assert store.events[0][1]["compressed_turns"] == 196
    manager.prepare(store, "u", "s", INCOMING, [])
    assert len(client.requests) == 1


def test_memory_is_data_and_previous_summary_is_merged():
    client = FakeClient()
    store = FakeStore([turn(i, "项目资料" * 300) for i in range(2, 12)], "用户预算 500 元", 1)
    messages = ContextManager(client, max_context_bytes=8_000).prepare(
        store, "u", "s", INCOMING, []
    )
    assert messages[1]["role"] == "user"
    assert "历史记忆资料" in messages[1]["content"]
    assert "用户预算 500 元" in client.requests[0][0][1]["content"]


@pytest.mark.parametrize(
    "reply",
    [
        LLMError("模拟服务异常"),
        ModelReply(""),
        ModelReply(tool_calls=[ToolCall("c", "todo", "{}")]),
    ],
)
def test_summary_failure_uses_visible_lossy_fallback(reply):
    client = FakeClient(reply)
    store = FakeStore([turn(i, "用户项目预算 500 元。" * 50) for i in range(1, 15)])
    messages = ContextManager(client, max_context_bytes=9_000).prepare(
        store, "u", "s", INCOMING, []
    )
    assert "有损摘录" in store.saved_summary
    assert "模拟服务异常" not in store.saved_summary
    assert store.events[0][1]["fallback"] is True
    assert payload_size(messages, []) <= 9_000


def test_summary_excludes_reasoning_and_request_is_bounded():
    client = FakeClient()
    records = [turn(i, "资料" * 2_000) for i in range(1, 10)]
    records[0]["messages"][1]["reasoning_content"] = "PRIVATE_REASONING"
    store = FakeStore(records)
    ContextManager(client, max_context_bytes=6_000).prepare(store, "u", "s", INCOMING, [])
    prompt, schemas = client.requests[0]
    assert schemas == []
    assert payload_size(prompt, schemas) <= 6_000
    assert "PRIVATE_REASONING" not in json.dumps(prompt)
    assert "截断" in prompt[1]["content"]
    assert store.events[0][1]["source_truncated"] is True


def test_compaction_never_splits_tool_round_trip():
    client = FakeClient()
    store = FakeStore([turn(i, "旧资料" * 200) for i in range(1, 8)] + [tool_turn(8)])
    messages = ContextManager(client, max_context_bytes=5_000, keep_recent_turns=2).prepare(
        store, "u", "s", INCOMING, []
    )
    assert messages[-5:-1] == store.records[-1]["messages"]
    assert store.through_id < 8


def test_too_large_current_turn_does_not_call_summary_or_advance_checkpoint():
    client = FakeClient()
    store = FakeStore([turn(1)])
    incoming = [{"role": "user", "content": "非常长" * 2_000}]
    with pytest.raises(ContextLimitError):
        ContextManager(client, max_context_bytes=3_000).prepare(store, "u", "s", incoming, [])
    assert client.requests == [] and store.saved == []


def test_current_tool_batch_is_preserved_or_rejected_as_a_whole():
    incoming = tool_turn(1)["messages"][:-1]
    manager = ContextManager(FakeClient(), max_context_bytes=3_000)
    messages = manager.prepare(FakeStore(), "u", "s", incoming, [])
    assert messages[-3:] == incoming
    incoming[-1]["content"] = "庞大的工具结果" * 1_000
    with pytest.raises(ContextLimitError):
        manager.prepare(FakeStore(), "u", "s", incoming, [])


def test_schema_size_counts_towards_budget():
    client = FakeClient()
    schemas = [{"type": "function", "function": {"name": "tool", "description": "描述" * 200}}]
    store = FakeStore([turn(i, "对话" * 60) for i in range(1, 5)])
    plain_size = payload_size(
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + [m for t in store.records for m in t["messages"]]
        + INCOMING,
        [],
    )
    manager = ContextManager(client, max_context_bytes=plain_size + 50)
    manager.prepare(store, "u", "s", INCOMING, [])
    assert client.requests == []
    messages = manager.prepare(store, "u", "s", INCOMING, schemas)
    assert len(client.requests) == 1
    assert payload_size(messages, schemas) <= manager.max_context_bytes


def test_large_existing_memory_is_trimmed_only_for_current_prompt():
    client = FakeClient()
    summary = "事实资料" * 1_000
    store = FakeStore(summary=summary, through_id=30)
    messages = ContextManager(client, max_context_bytes=3_000).prepare(
        store, "u", "s", INCOMING, []
    )
    assert payload_size(messages, []) <= 3_000
    assert store.saved_summary == summary and store.through_id == 30
    assert client.requests == [] and store.saved == []


def test_generated_summary_is_capped_and_truncation_is_visible():
    store = FakeStore([turn(i, "旧对话" * 100) for i in range(1, 10)])
    client = FakeClient(ModelReply("事实" * 2_000))
    ContextManager(client, max_context_bytes=7_000, summary_max_chars=150).prepare(
        store, "u", "s", INCOMING, []
    )
    assert len(store.saved_summary) <= 150
    assert "摘要超长已截断" in store.saved_summary


def test_loop_growth_trims_memory_injection_without_advancing_watermark():
    client = FakeClient()
    summary = "先前事实" * 500
    store = FakeStore([turn(i) for i in range(11, 15)], summary, 10)
    base = [{"role": "system", "content": SYSTEM_PROMPT}]
    base += [message for record in store.records for message in record["messages"]]
    base += INCOMING
    manager = ContextManager(client, max_context_bytes=payload_size(base, []) + 500)
    messages = manager.prepare(store, "u", "s", INCOMING, [], allow_model_summary=False)
    assert payload_size(messages, []) <= manager.max_context_bytes
    assert "截断" in messages[1]["content"]
    assert messages[-9:-1] == [
        message for record in store.records for message in record["messages"]
    ]
    assert store.saved_summary == summary and store.through_id == 10
    assert store.saved == [] and client.requests == []


def test_loop_growth_uses_marked_local_extract_when_old_turns_must_be_compacted():
    client = FakeClient()
    records = [turn(i, "历史资料" * 100) for i in range(11, 25)]
    store = FakeStore(records, "已有项目事实", 10)
    manager = ContextManager(client, max_context_bytes=9_000)
    messages = manager.prepare(store, "u", "s", INCOMING, [], allow_model_summary=False)
    assert client.requests == []
    assert "有损摘录：本轮压缩预算受限" in store.saved_summary
    assert "摘要模型不可用" not in store.saved_summary
    assert 10 < store.through_id < 24
    assert payload_size(messages, []) <= manager.max_context_bytes
    assert store.events[0][1]["fallback"] is True
    assert store.events[0][1]["model_summary_allowed"] is False
    assert store.records == records
