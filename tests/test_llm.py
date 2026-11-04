import json

import httpx
import pytest

from loomora.llm import (
    MAX_ARGUMENT_CHARS,
    MAX_RESPONSE_BYTES,
    DeepSeekClient,
    LLMError,
    ModelProtocolError,
    ModelReply,
    ToolCall,
    parse_response,
)


def response(content="你好", calls=None, finish="stop", **extra):
    message = {"role": "assistant", "content": content, **extra}
    if calls is not None:
        message["tool_calls"] = calls
    return {"choices": [{"finish_reason": finish, "message": message}]}


def call(call_id="call_1", name="calculator", arguments='{"expression":"1+1"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def make_client(handler, **kwargs):
    return DeepSeekClient(
        "test-key", client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs
    )


def test_parse_final_and_reasoning_round_trip():
    reply = parse_response(response(reasoning_content="简短的模型决策信息"))
    assert reply == ModelReply("你好", reasoning="简短的模型决策信息")
    assert reply.as_message() == {
        "role": "assistant",
        "content": "你好",
        "reasoning_content": "简短的模型决策信息",
    }


def test_parse_multiple_calls_and_preserve_protocol():
    raw_calls = [call(), call("call_2", "todo", '{"action":"list"}')]
    reply = parse_response(response(None, raw_calls, "tool_calls"))
    assert reply.tool_calls == [
        ToolCall("call_1", "calculator", '{"expression":"1+1"}'),
        ToolCall("call_2", "todo", '{"action":"list"}'),
    ]
    assert reply.as_message() == {"role": "assistant", "content": None, "tool_calls": raw_calls}


def test_invalid_argument_json_is_left_for_tool_error_recovery():
    reply = parse_response(response(None, [call(arguments="{broken")], "tool_calls"))
    assert reply.tool_calls[0].arguments == "{broken"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"choices": []},
        {"choices": [None]},
        response(""),
        response("  "),
        response(None),
        response(123),
        response(finish="length"),
        response(finish="content_filter"),
        response(finish=None),
        response(calls=[], finish="tool_calls"),
        response(calls={}),
        response(reasoning_content=[]),
        response(calls=[call(), call()]),
        response(calls=[call(str(i)) for i in range(9)]),
        response(calls=[call(name="bad.name")]),
        response(calls=[call(call_id="")]),
        response(calls=[call(arguments={})]),
        response(calls=[call(arguments="x" * (MAX_ARGUMENT_CHARS + 1))]),
        response(calls=[{"type": "custom", "id": "c"}]),
        {"choices": [{"finish_reason": "stop", "message": {"role": "user", "content": "bad"}}]},
    ],
)
def test_reject_unusable_responses(payload):
    with pytest.raises(ModelProtocolError):
        parse_response(payload)


def test_http_request_shape_and_defaults():
    received = []

    def handler(request):
        received.append(request)
        return httpx.Response(200, json=response())

    client = make_client(handler)
    messages = [{"role": "user", "content": "你好"}]
    tools = [{"type": "function", "function": {"name": "todo"}}]
    assert client.complete(messages, tools).content == "你好"
    request = received[0]
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body == {
        "model": "deepseek-flash",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 2048,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    assert request.extensions["timeout"]["read"] == 30.0


def test_custom_endpoint_and_empty_tools():
    def handler(request):
        assert str(request.url) == "https://gateway.example/v1/chat/completions"
        assert "tools" not in json.loads(request.content)
        return httpx.Response(200, json=response())

    assert make_client(handler, base_url="https://gateway.example/v1/").complete([], []).content


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_retry_transient_status_then_success(monkeypatch, status):
    attempts, sleeps = [], []
    monkeypatch.setattr("loomora.llm.time.sleep", sleeps.append)

    def handler(request):
        attempts.append(request)
        return (
            httpx.Response(status, text="private failure")
            if len(attempts) < 3
            else httpx.Response(200, json=response())
        )

    assert make_client(handler).complete([], []).content == "你好"
    assert len(attempts) == 3
    assert sleeps == [0.25, 0.5]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 302])
def test_permanent_errors_and_redirects_are_not_retried(status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status, text="test-key SECRET", headers={"Location": "https://attacker.example"}
        )

    with pytest.raises(LLMError) as exc:
        make_client(handler).complete([], [])
    assert len(requests) == 1
    assert str(status) in str(exc.value)
    assert "SECRET" not in str(exc.value) and "test-key" not in str(exc.value)


def test_exhausted_retry_is_bounded_and_sanitized(monkeypatch):
    monkeypatch.setattr("loomora.llm.time.sleep", lambda _: None)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429, text="test-key SECRET")

    with pytest.raises(LLMError, match="429"):
        make_client(handler).complete([], [])
    assert len(requests) == 3


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError])
def test_network_failures_are_bounded_and_sanitized(monkeypatch, error_type):
    monkeypatch.setattr("loomora.llm.time.sleep", lambda _: None)
    requests = []

    def handler(request):
        requests.append(request)
        raise error_type("SECRET test-key", request=request)

    with pytest.raises(LLMError) as exc:
        make_client(handler, max_retries=1).complete([], [])
    assert len(requests) == 2
    assert "SECRET" not in str(exc.value)


@pytest.mark.parametrize(
    "body",
    [b"not-json", b'{"choices": NaN}', b"\xff", b"x" * (MAX_RESPONSE_BYTES + 1)],
    ids=["invalid-json", "nonfinite-number", "invalid-encoding", "oversized-body"],
)
def test_bad_http_body_is_rejected_without_exposure(body):
    with pytest.raises(ModelProtocolError):
        make_client(lambda _: httpx.Response(200, content=body)).complete([], [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": ""},
        {"base_url": "http://api.deepseek.com"},
        {"base_url": "https://user:password@example.com"},
        {"base_url": "https://example.com?key=secret"},
        {"base_url": "https://example.com#fragment"},
        {"base_url": "broken"},
        {"model": ""},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"max_retries": -1},
    ],
)
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        DeepSeekClient(**{"api_key": "test-key", **kwargs})
