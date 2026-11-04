"""核心循环：模型决策 → 工具执行 → 结果回填，直到最终回答或达到上限。"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

from .context import ContextLimitError, ContextManager, payload_size
from .llm import LLMError
from .store import Store
from .tools import ToolContext, ToolRegistry, build_registry


@dataclass(frozen=True)
class RunResult:
    answer: str
    status: str
    steps: int
    run_id: str


def close_pending(messages: list[dict]) -> None:
    """中断后补齐未收到结果的调用；不重放可能已经产生副作用的工具。"""
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    for message in list(messages):
        for call in message.get("tool_calls", []):
            if call["id"] not in answered:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(
                            {
                                "ok": False,
                                "error": {
                                    "code": "interrupted",
                                    "message": "执行被中断，结果未知。修改操作可能已生效，请先查询状态，不要盲目重复。",
                                },
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                answered.add(call["id"])


class Agent:
    def __init__(
        self,
        client,
        store: Store,
        registry: ToolRegistry | None = None,
        max_steps: int = 8,
        context: ContextManager | None = None,
    ):
        if not 1 <= max_steps <= 32:
            raise ValueError("max_steps 必须在 1–32 之间")
        self.client = client
        self.store = store
        self.registry = registry or build_registry()
        self.max_steps = max_steps
        self.context = context or ContextManager(client)

    def _recover(self, user_id: str, session_id: str):
        for turn in self.store.turns(user_id, session_id):
            if turn["status"] == "running":
                messages = turn["messages"]
                close_pending(messages)
                messages.append(
                    {
                        "role": "assistant",
                        "content": "上次执行中断。已保存的工具结果仍有效，未返回的操作结果未知，请先核对待办。",
                    }
                )
                self.store.save_turn(turn["id"], messages, "completed")
                self.store.trace(user_id, session_id, "turn_recovered", turn_id=turn["id"])

    def run(self, user_id: str, session_id: str, text: str) -> RunResult:
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("请输入 1–4000 字符的非空消息")
        # 锁覆盖整次交互，SQLite 事务仅覆盖短暂读写；其他 session 可继续执行。
        with self.store.session(user_id, session_id):
            self._recover(user_id, session_id)
            return self._run_locked(user_id, session_id, text.strip())

    def _run_locked(self, user_id: str, session_id: str, text: str) -> RunResult:
        run_id = uuid.uuid4().hex
        current = [{"role": "user", "content": text}]
        turn_id = self.store.start_turn(user_id, session_id, current)
        steps = 0
        schemas = self.registry.schemas()

        def trace(event: str, **metadata):
            self.store.trace(user_id, session_id, event, run_id=run_id, step=steps, **metadata)

        def finish(answer: str, status: str) -> RunResult:
            close_pending(current)
            current.append({"role": "assistant", "content": answer})
            self.store.save_turn(turn_id, current, "completed")
            trace("run_finished", status=status)
            return RunResult(answer, status, steps, run_id)

        trace("run_started")
        try:
            messages = self.context.prepare(self.store, user_id, session_id, current, schemas)
            prefix = messages[: -len(current)]
            call_ids: set[str] = set()
            tool_context = ToolContext(self.store, user_id, session_id)
            for steps in range(1, self.max_steps + 1):
                messages = prefix + current
                if payload_size(messages, schemas) > self.context.max_context_bytes:
                    # 工具结果会增长当前轮；可继续压旧历史，但每个用户请求只允许
                    # 首次 prepare 调用摘要模型。当前轮的 call/result 始终整体保留。
                    messages = self.context.prepare(
                        self.store,
                        user_id,
                        session_id,
                        current,
                        schemas,
                        allow_model_summary=False,
                    )
                    prefix = messages[: -len(current)]
                started = time.monotonic()
                reply = self.client.complete(messages, schemas)
                trace(
                    "llm_completed",
                    duration_ms=round((time.monotonic() - started) * 1000),
                    tool_count=len(reply.tool_calls),
                    has_reasoning=reply.reasoning is not None,
                )
                if not reply.tool_calls:
                    # 原生协议字段保留在历史中；CLI 和 trace 只显示最终答案/元数据。
                    current.append(reply.as_message())
                    self.store.save_turn(turn_id, current, "completed")
                    trace("run_finished", status="completed")
                    return RunResult(reply.content or "", "completed", steps, run_id)

                if any(call.id in call_ids for call in reply.tool_calls):
                    raise LLMError("模型重复使用了工具调用 ID，本次执行已停止。")
                current.append(reply.as_message())
                self.store.save_turn(turn_id, current)
                for call in reply.tool_calls:
                    call_ids.add(call.id)
                    trace("tool_started", tool=call.name, call_id=call.id)
                    started = time.monotonic()
                    result = self.registry.execute(call.name, call.arguments, tool_context)
                    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
                    # 结果过大时仍回填有效 JSON，避免破坏 tool_call_id 配对。
                    if len(encoded.encode("utf-8")) > 8000:
                        encoded = json.dumps(
                            {
                                "ok": result["ok"],
                                "truncated": True,
                                "preview": encoded[:1500],
                                "note": "工具结果过长，以上仅为预览。",
                            },
                            ensure_ascii=False,
                        )
                    current.append({"role": "tool", "tool_call_id": call.id, "content": encoded})
                    self.store.save_turn(turn_id, current)
                    trace(
                        "tool_finished",
                        tool=call.name,
                        call_id=call.id,
                        ok=result["ok"],
                        duration_ms=round((time.monotonic() - started) * 1000),
                    )
            return finish(
                "已达到本次最大执行轮数，任务可能尚未完成。已执行的工具结果已保存，可继续追问。",
                "max_steps",
            )
        except (LLMError, ContextLimitError) as exc:
            trace("run_error", error_type=type(exc).__name__)
            return finish(f"本次执行未完成：{exc} 已完成的工具操作不会回滚。", "error")
        except KeyboardInterrupt:
            trace("run_cancelled")
            return finish(
                "本次执行已取消。已完成的操作保留，未返回的操作请先查询状态。", "cancelled"
            )
        except Exception as exc:
            # 工具边界已处理预期异常；最后一道边界保证故障后的历史仍可继续。
            trace("run_error", error_type=type(exc).__name__)
            return finish("本次执行发生内部错误，已保存历史。请检查本地存储并重试。", "error")
