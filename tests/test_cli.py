import json

import pytest

from loomora import cli
from loomora.llm import ModelReply


class LocalClient:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def complete(self, messages, tools):
        return ModelReply(content="你好，这是测试回答。")

    def close(self):
        self.closed = True


def test_missing_key_exits_with_actionable_message(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert cli.main(["--message", "你好"]) == 2
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / ".loomora").exists()


def test_cli_single_message_and_safe_trace(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-secret")
    monkeypatch.setattr(cli, "DeepSeekClient", LocalClient)
    assert (
        cli.main(["--user", "alice", "--session", "window1", "--message", "你好", "--trace"]) == 0
    )
    output = capsys.readouterr().out
    assert "你好，这是测试回答。" in output
    assert "run_finished" in output
    assert "not-a-real-secret" not in output


def test_cli_loads_only_local_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=local-test-value\n", encoding="utf-8")
    created = []

    def create_client(key, **kwargs):
        created.append((key, LocalClient()))
        return created[-1][1]

    monkeypatch.setattr(cli, "DeepSeekClient", create_client)
    try:
        assert cli.main(["--message", "hello"]) == 0
        assert created[0][0] == "local-test-value"
        assert created[0][1].closed
    finally:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


@pytest.mark.parametrize("option,value", [("--context-bytes", "1"), ("--max-steps", "0")])
def test_cli_invalid_budget(tmp_path, monkeypatch, option, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setattr(cli, "DeepSeekClient", LocalClient)
    assert cli.main([option, value, "--message", "hello"]) == 2


def test_interactive_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setattr(cli, "DeepSeekClient", LocalClient)
    inputs = iter(["你好", "/history", "/todos", "/trace", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert "user> 你好" in output
    assert json.dumps([], ensure_ascii=False) in output
