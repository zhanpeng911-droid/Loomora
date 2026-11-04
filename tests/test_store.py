"""使用临时 SQLite 数据库验证持久化、隔离与真实进程锁。"""

import subprocess
import sys
from pathlib import Path

import pytest

from loomora.store import SessionBusyError, Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "agent-data")


def test_turns_are_isolated_and_watermark_uses_ids_not_list_positions(store):
    first_messages = [{"role": "user", "content": "记住我在广州"}]
    with store.session("alice", "weather"):
        first_id = store.start_turn("alice", "weather", first_messages)
        answer = first_messages + [{"role": "assistant", "content": "记住了"}]
        store.save_turn(first_id, answer, "completed")
    with store.session("alice", "weekly"):
        store.start_turn("alice", "weekly", [{"role": "user", "content": "写周报"}])
    with store.session("bob", "weather"):
        store.start_turn("bob", "weather", [{"role": "user", "content": "厦门天气"}])
    with store.session("alice", "weather"):
        follow_up = [{"role": "user", "content": "我在哪个城市？"}]
        second_id = store.start_turn("alice", "weather", follow_up)

    turns = store.turns("alice", "weather")
    assert [turn["id"] for turn in turns] == [first_id, second_id]
    assert turns[0]["messages"] == answer
    assert turns[0]["status"] == "completed"
    assert turns[1]["status"] == "running"
    assert store.turns("alice", "weather", after_id=first_id) == [turns[1]]
    assert store.turns("alice", "weather", after_id=second_id) == []
    assert store.turns("nobody", "weather") == []
    assert store.turns("alice", "weekly")[0]["messages"][0]["content"] == "写周报"
    assert store.turns("bob", "weather")[0]["messages"][0]["content"] == "厦门天气"


def test_reopened_store_preserves_history_memory_todos_and_traces(store):
    messages = [{"role": "user", "content": "周五交报告"}]
    with store.session("alice", "work"):
        turn_id = store.start_turn("alice", "work", messages)
        store.save_turn(turn_id, messages, "completed")
        todo = store.add_todo("alice", "work", "周五交报告")
        store.save_memory("alice", "work", "报告截止日期：周五。", turn_id)
        store.trace("alice", "work", "tool_completed", tool="todo", ok=True)

    reopened = Store(store.directory)
    with reopened.session("alice", "work"):
        assert reopened.turns("alice", "work")[0]["messages"] == messages
        assert reopened.memory("alice", "work") == ("报告截止日期：周五。", turn_id)
        assert reopened.list_todos("alice", "work") == [todo]
        assert reopened.traces("alice", "work")[0]["metadata"] == {"tool": "todo", "ok": True}


def test_todo_ids_do_not_grant_access_to_another_scope(store):
    original = store.add_todo("alice", "window-1", "准备面试")
    for user_id, session_id in [("alice", "window-2"), ("bob", "window-1")]:
        assert store.list_todos(user_id, session_id) == []
        assert store.complete_todo(user_id, session_id, original["id"]) is None
        independent = store.add_todo(user_id, session_id, "准备面试")
        assert independent["id"] != original["id"]
        assert store.list_todos(user_id, session_id) == [independent]
    assert store.list_todos("alice", "window-1") == [original]


def test_todo_open_title_dedup_and_completion_are_idempotent(store):
    original = store.add_todo("alice", "work", "写周报")
    assert store.add_todo("alice", "work", "写周报") == original
    assert store.list_todos("alice", "work") == [original]
    completed = store.complete_todo("alice", "work", original["id"])
    assert completed == {**original, "completed": True}
    assert store.complete_todo("alice", "work", original["id"]) == completed
    new_todo = store.add_todo("alice", "work", "写周报")
    assert new_todo["id"] != original["id"]
    assert new_todo["completed"] is False
    assert store.list_todos("alice", "work") == [completed, new_todo]


def test_todo_limit_includes_completed_items_and_is_per_session(store):
    todos = [store.add_todo("alice", "work", f"任务 {i}") for i in range(100)]
    # 已有未完成项的重试不会消耗容量，也不应在满额时失败。
    assert store.add_todo("alice", "work", "任务 99") == todos[-1]
    store.complete_todo("alice", "work", todos[0]["id"])
    with pytest.raises(ValueError, match="100"):
        store.add_todo("alice", "work", "新任务")
    assert len(store.list_todos("alice", "work")) == 100
    assert store.add_todo("alice", "personal", "新任务")["title"] == "新任务"
    assert store.add_todo("bob", "work", "新任务")["title"] == "新任务"


def test_memory_and_its_watermark_are_isolated(store):
    scopes = [("alice", "window-1"), ("alice", "window-2"), ("bob", "window-1")]
    for index, scope in enumerate(scopes):
        with store.session(*scope):
            assert store.memory(*scope) == ("", 0)
            store.save_memory(*scope, f"summary-{index}", 10 + index)
    for index, scope in enumerate(scopes):
        assert store.memory(*scope) == (f"summary-{index}", 10 + index)
    with store.session(*scopes[0]):
        store.save_memory(*scopes[0], "updated", 100)
    assert store.memory(*scopes[0]) == ("updated", 100)
    assert store.memory(*scopes[1]) == ("summary-1", 11)
    assert store.memory("unknown", "window-1") == ("", 0)


def test_trace_scope_and_recent_limit_preserve_chronological_order(store):
    for index in range(4):
        store.trace("alice", "work", "step", step=index)
        store.trace("alice", "personal", "other_window", step=index)
        store.trace("bob", "work", "other_user", step=index)
    traces = store.traces("alice", "work", limit=2)
    assert [trace["event"] for trace in traces] == ["step", "step"]
    assert [trace["metadata"]["step"] for trace in traces] == [2, 3]
    assert store.traces("alice", "missing") == []


def test_session_lock_rejects_same_scope_but_allows_other_scopes(store):
    other_store = Store(store.directory)
    with store.session("alice", "work"):
        with pytest.raises(SessionBusyError):
            with other_store.session("alice", "work"):
                pytest.fail("busy session must not enter")
        with other_store.session("alice", "personal"):
            assert other_store.memory("alice", "personal") == ("", 0)
        with other_store.session("bob", "work"):
            assert other_store.memory("bob", "work") == ("", 0)
    with other_store.session("alice", "work"):
        store.save_memory("alice", "work", "lock released", 1)


def test_session_lock_releases_after_an_exception(store):
    with pytest.raises(RuntimeError, match="interrupted"):
        with store.session("alice", "work"):
            raise RuntimeError("interrupted")
    with Store(store.directory).session("alice", "work"):
        assert store.memory("alice", "work") == ("", 0)


@pytest.mark.parametrize("bad_id", ["", " ", "a" * 101, None, 42])
def test_invalid_session_identifiers_are_rejected_without_creating_state(store, bad_id):
    for user_id, session_id in [(bad_id, "work"), ("alice", bad_id)]:
        with pytest.raises(ValueError):
            with store.session(user_id, session_id):
                pytest.fail("invalid identifier must not enter")
    assert store.memory("alice", "work") == ("", 0)


def test_session_lock_is_enforced_between_real_processes(store):
    script = """
import sys
from loomora.store import Store, SessionBusyError
store = Store(sys.argv[1])
try:
    with store.session('alice', 'work'):
        print('unexpected-entry')
except SessionBusyError:
    print('busy')
with store.session('alice', 'personal'):
    print('other-session-entered')
"""
    with store.session("alice", "work"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(store.directory)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    assert result.stdout.splitlines() == ["busy", "other-session-entered"]
    assert result.stderr == ""
    with Store(store.directory).session("alice", "work"):
        assert store.memory("alice", "work") == ("", 0)


def test_parameterized_storage_keeps_sql_like_text_as_data(store):
    user_id = "alice' OR 1=1 --"
    title = "'); DROP TABLE todos; --"
    with store.session(user_id, "work"):
        todo = store.add_todo(user_id, "work", title)
        assert store.list_todos(user_id, "work") == [todo]
        assert store.list_todos("alice", "work") == []
    assert store.add_todo("alice", "work", "正常任务")["title"] == "正常任务"
