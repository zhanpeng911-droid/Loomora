"""DeepSeek HTTP 接入与输出解析；不依赖 Agent 框架或厂商 SDK。"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

MAX_TOOL_CALLS = 8
MAX_ARGUMENT_CHARS = 16_384
MAX_RESPONSE_BYTES = 1_048_576
_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class LLMError(Exception):
    """可展示给用户的安全错误，不包含密钥、请求内容或服务商响应原文。"""


class ModelProtocolError(LLMError):
    """模型响应无法被可靠地解释为最终回答或工具调用。"""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ModelReply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None

    def as_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        # DeepSeek 思考模式要求回传此字段。默认禁用思考，但解析层不丢协议数据。
        if self.reasoning is not None:
            message["reasoning_content"] = self.reasoning
        return message


def parse_response(payload: dict[str, Any]) -> ModelReply:
    """先检查消息结构；参数的 JSON 和 Schema 校验留给工具层，支持模型纠错。"""
    if not isinstance(payload, dict):
        raise ModelProtocolError("LLM 响应必须是 JSON 对象。")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ModelProtocolError("LLM 响应缺少有效的 choices。")
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish == "length":
        raise ModelProtocolError("LLM 输出达到长度上限；本次响应未执行工具，请缩短请求。")
    if finish not in ("stop", "tool_calls"):
        raise ModelProtocolError("LLM 未正常结束响应，无法继续执行。")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ModelProtocolError("LLM 响应缺少 assistant 消息。")
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if content is not None and not isinstance(content, str):
        raise ModelProtocolError("LLM 回答必须是文本。")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ModelProtocolError("LLM 思考字段必须是文本。")
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list) or len(raw_calls) > MAX_TOOL_CALLS:
        raise ModelProtocolError("LLM 工具调用格式错误或超过单步调用上限。")
    calls = []
    seen_ids: set[str] = set()
    for raw in raw_calls:
        if not isinstance(raw, dict) or raw.get("type") != "function":
            raise ModelProtocolError("LLM 返回了不支持的工具调用类型。")
        call_id = raw.get("id")
        function = raw.get("function")
        if (
            not isinstance(call_id, str)
            or not call_id.strip()
            or len(call_id) > 256
            or call_id in seen_ids
            or not isinstance(function, dict)
        ):
            raise ModelProtocolError("LLM 工具调用 ID 缺失、重复或格式错误。")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise ModelProtocolError("LLM 工具名称格式错误。")
        if not isinstance(arguments, str) or len(arguments) > MAX_ARGUMENT_CHARS:
            raise ModelProtocolError("LLM 工具参数必须是有长度限制的 JSON 字符串。")
        seen_ids.add(call_id)
        calls.append(ToolCall(call_id, name, arguments))
    if finish == "tool_calls" and not calls:
        raise ModelProtocolError("LLM 声明了工具调用，但没有提供调用内容。")
    if not calls and not (content and content.strip()):
        raise ModelProtocolError("LLM 返回了空回答。")
    return ModelReply(content, calls, reasoning)


def _reject_constant(value: str) -> None:
    raise ValueError("Non-standard JSON constant")


class DeepSeekClient:
    """同步、非流式客户端；重试只发生在模型请求层，不重放本地工具。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-flash",
        timeout: float = 30.0,
        max_tokens: int = 2048,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("请设置 DEEPSEEK_API_KEY。")
        try:
            url = httpx.URL(base_url)
        except (httpx.InvalidURL, TypeError):
            raise ValueError("DEEPSEEK_BASE_URL 必须是有效 HTTPS 地址。") from None
        if url.scheme != "https" or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("DEEPSEEK_BASE_URL 必须是无凭据、查询参数和片段的 HTTPS 地址。")
        if not isinstance(model, str) or not model.strip() or len(model) > 128:
            raise ValueError("请设置有效的 DEEPSEEK_MODEL。")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (float, int))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("LLM timeout 必须是有限正数。")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32_768:
            raise ValueError("LLM max_tokens 必须在 1 到 32768 之间。")
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("LLM max_retries 必须在 0 到 5 之间。")
        self.endpoint = str(url).rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._api_key = api_key.strip()
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client()

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        if tools:
            body.update(tools=tools, tool_choice="auto")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in range(self.max_retries + 1):
            try:
                with self._client.stream(
                    "POST",
                    self.endpoint,
                    json=body,
                    headers=headers,
                    timeout=self.timeout,
                    follow_redirects=False,
                ) as response:
                    status = response.status_code
                    if status == 429 or 500 <= status < 600:
                        if attempt < self.max_retries:
                            # 小且有限的退避；不信任可能很长的 Retry-After。
                            time.sleep(0.25 * 2**attempt)
                            continue
                        raise LLMError(f"LLM 服务暂时不可用（HTTP {status}），请稍后重试。")
                    if status != 200:
                        raise LLMError(f"LLM 请求失败（HTTP {status}），请检查 API 配置。")
                    chunks = bytearray()
                    for chunk in response.iter_bytes():
                        chunks.extend(chunk)
                        if len(chunks) > MAX_RESPONSE_BYTES:
                            raise ModelProtocolError("LLM 响应超过大小限制。")
                try:
                    payload = json.loads(chunks, parse_constant=_reject_constant)
                except (ValueError, UnicodeDecodeError, RecursionError):
                    raise ModelProtocolError("LLM 响应不是有效 JSON。") from None
                return parse_response(payload)
            except httpx.TimeoutException:
                error = "LLM 请求超时，请稍后重试。"
            except httpx.RequestError:
                error = "LLM 网络请求失败，请检查连接后重试。"
            if attempt < self.max_retries:
                time.sleep(0.25 * 2**attempt)
            else:
                raise LLMError(error) from None
        raise AssertionError("unreachable")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> DeepSeekClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
