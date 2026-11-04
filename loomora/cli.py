"""本地 CLI：每个终端指定不同 session，即可模拟两个独立聊天窗口。"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

from dotenv import load_dotenv

from .context import ContextManager
from .llm import DeepSeekClient
from .runtime import Agent
from .store import SessionBusyError, Store


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Loomora — 自研最小 DeepSeek Agent")
    result.add_argument("--user", default="local", help="本地数据命名空间，不是身份认证")
    result.add_argument("--session", default="default", help="窗口/session ID，相同 ID 可续聊")
    result.add_argument("--data-dir", help="SQLite 和会话锁目录，默认 .loomora")
    result.add_argument("--message", help="执行单条消息后退出；省略则进入交互模式")
    result.add_argument("--max-steps", type=int, default=8, help="单次消息最大模型决策轮数")
    result.add_argument("--context-bytes", type=int, default=32000, help="上下文 UTF-8 字节预算")
    result.add_argument("--trace", action="store_true", help="回答后打印本会话最近执行日志")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # 仅加载当前工作目录的 .env；已设置的进程环境变量优先。
    load_dotenv(os.path.join(os.getcwd(), ".env"), override=False)
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        print(
            "缺少 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并在本地填写密钥。", file=sys.stderr
        )
        return 2
    client = None
    try:
        if not 8000 <= args.context_bytes <= 128000:
            raise ValueError("--context-bytes 必须在 8000–128000 之间")
        store = Store(args.data_dir or os.getenv("LOOMORA_DATA_DIR", ".loomora"))
        client = DeepSeekClient(
            key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        )
        agent = Agent(
            client,
            store,
            max_steps=args.max_steps,
            context=ContextManager(client, max_context_bytes=args.context_bytes),
        )

        def show_trace():
            for row in store.traces(args.user, args.session):
                print(json.dumps(row, ensure_ascii=False))

        def execute(text: str) -> int:
            result = agent.run(args.user, args.session, text)
            print(f"Agent> {result.answer}")
            if args.trace:
                show_trace()
            return 0 if result.status == "completed" else 1

        if args.message is not None:
            return execute(args.message)
        print(f"Loomora | user={args.user} session={args.session}")
        print("/exit 退出 · /history 历史 · /todos 待办 · /trace 日志")
        while True:
            try:
                text = input("You> ").strip()
                if text == "/exit":
                    return 0
                if text == "/history":
                    for turn in store.turns(args.user, args.session):
                        for message in turn["messages"]:
                            if message["role"] in ("user", "assistant") and message.get("content"):
                                print(f"{message['role']}> {message['content']}")
                elif text == "/todos":
                    print(
                        json.dumps(
                            store.list_todos(args.user, args.session), ensure_ascii=False, indent=2
                        )
                    )
                elif text == "/trace":
                    show_trace()
                elif text:
                    execute(text)
            except (SessionBusyError, ValueError) as exc:
                print(str(exc), file=sys.stderr)
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
    except (ValueError, SessionBusyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (sqlite3.Error, OSError):
        print("无法读写本地存储，请检查数据目录权限或磁盘空间。", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
