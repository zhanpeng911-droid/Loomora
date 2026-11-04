"""显式启用才访问真实 DeepSeek；自动测试绝不偷偷消耗 API 额度。"""

import copy
import json
import os

import pytest
from dotenv import load_dotenv

from loomora.context import ContextManager
from loomora.llm import DeepSeekClient
from loomora.runtime import Agent
from loomora.store import Store


class RecordingClient(DeepSeekClient):
    """仅捕获测试会话的模型输入，不记录 HTTP 请求头或密钥。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        return super().complete(messages, tools)


@pytest.fixture
def live_client():
    load_dotenv(os.path.join(os.getcwd(), ".env"), override=False)
    if os.getenv("RUN_LIVE_TESTS") != "1":
        pytest.skip("设置 RUN_LIVE_TESTS=1 才运行真实 API 验收")
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    assert key, "真实验收需要本地 DEEPSEEK_API_KEY"
    with RecordingClient(
        key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
    ) as client:
        yield client


def successful_results(messages, tool_name, action=None):
    calls = {}
    for message in messages:
        for call in message.get("tool_calls", []):
            if call["function"]["name"] == tool_name:
                try:
                    args = json.loads(call["function"]["arguments"])
                except ValueError:
                    continue
                if action is None or (isinstance(args, dict) and args.get("action") == action):
                    calls[call["id"]] = call
    return [
        result["data"]
        for message in messages
        if message["role"] == "tool" and message["tool_call_id"] in calls
        for result in [json.loads(message["content"])]
        if result.get("ok") is True
    ]


@pytest.mark.live
def test_real_deepseek_tool_chain_and_followup(tmp_path, live_client):
    store = Store(tmp_path)
    client = live_client
    agent = Agent(client, store)
    result = agent.run(
        "acceptance",
        "window1",
        "请用 calculator 计算 17*23，"
        "用 search 搜索 Agent 循环，并用 todo 添加一条标题严格为‘整理 Agent 笔记’的待办。"
        "请实际调用三个工具，不要仅描述步骤。",
    )
    assert result.status == "completed", result.answer
    messages = store.turns("acceptance", "window1")[0]["messages"]
    assert any(item["result"] == 391 for item in successful_results(messages, "calculator"))
    assert any(
        item["mock"] is True and item["results"] for item in successful_results(messages, "search")
    )
    assert successful_results(messages, "todo", "add")
    todos = store.list_todos("acceptance", "window1")
    assert len(todos) == 1 and todos[0]["title"] == "整理 Agent 笔记"
    followup = agent.run("acceptance", "window1", "把刚才那条待办标记完成，请实际调用工具。")
    assert followup.status == "completed", followup.answer
    assert store.list_todos("acceptance", "window1")[0]["completed"] is True
    assert successful_results(
        store.turns("acceptance", "window1")[-1]["messages"], "todo", "complete"
    )

    request_start = len(client.requests)
    other = agent.run("acceptance", "window2", "请调用 todo 列出当前会话的待办。")
    assert other.status == "completed", other.answer
    assert store.list_todos("acceptance", "window2") == []
    assert successful_results(
        store.turns("acceptance", "window2")[0]["messages"], "todo", "list"
    ) == [{"items": [], "next_cursor": None}]
    assert "整理 Agent 笔记" not in json.dumps(client.requests[request_start:], ensure_ascii=False)


@pytest.mark.live
def test_real_deepseek_plain_followup_after_restart_and_compaction(tmp_path, live_client):
    store = Store(tmp_path)
    first = Agent(live_client, store).run(
        "acceptance", "memory", "本项目代号是北辰-47，预算是730元。请记住并简短确认，无需工具。"
    )
    assert first.status == "completed", first.answer
    store = Store(tmp_path)
    question = "刚才的项目代号是什么、预算多少？请原样写出代号，金额用阿拉伯数字，无需工具。"
    resumed = Agent(live_client, store).run("acceptance", "memory", question)
    assert resumed.status == "completed", resumed.answer
    assert "北辰-47" in resumed.answer and "730" in resumed.answer

    # 合成历史只用于触发压缩，不把它宣称为数十次真实模型聊天。
    with store.session("acceptance", "memory"):
        for index in range(12):
            messages = [
                {"role": "user", "content": f"背景记录{index}：" + "本轮没有变更项目约束。" * 80},
                {"role": "assistant", "content": "收到，项目代号和预算保持原记录。"},
            ]
            turn_id = store.start_turn("acceptance", "memory", messages)
            store.save_turn(turn_id, messages, "completed")
    agent = Agent(
        live_client,
        store,
        context=ContextManager(live_client, max_context_bytes=12000, keep_recent_turns=2),
    )
    compressed = agent.run("acceptance", "memory", question)
    assert compressed.status == "completed", compressed.answer
    assert "北辰-47" in compressed.answer and "730" in compressed.answer
    events = [
        row
        for row in store.traces("acceptance", "memory", 100)
        if row["event"] == "context_compacted"
    ]
    assert events and events[-1]["metadata"]["fallback"] is False
