"""SQLite 保存事实，文件锁保证同一会话一次只执行一个 turn。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock, Timeout


class SessionBusyError(Exception):
    pass


class Store:
    def __init__(self, data_dir: str | Path = ".loomora"):
        self.directory = Path(data_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "locks").mkdir(exist_ok=True)
        self.path = self.directory / "agent.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '', through_id INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(user_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    messages TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS turns_scope ON turns(user_id, session_id, id);
                CREATE TABLE IF NOT EXISTS todos (
                    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    title TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS todos_scope ON todos(user_id, session_id);
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    event TEXT NOT NULL, metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS traces_scope ON traces(user_id, session_id, id);
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def session(self, user_id: str, session_id: str) -> Iterator[None]:
        for value in (user_id, session_id):
            if not isinstance(value, str) or not value.strip() or len(value) > 100:
                raise ValueError("user/session 必须是 1–100 字符的非空字符串")
        key = hashlib.sha256(json.dumps([user_id, session_id]).encode()).hexdigest()
        lock = FileLock(self.directory / "locks" / f"{key}.lock")
        try:
            lock.acquire(timeout=0)
        except Timeout as exc:
            raise SessionBusyError("该 session 正在执行，请完成后重试或切换 session。") from exc
        try:
            with self.connect() as db:
                db.execute(
                    "INSERT OR IGNORE INTO sessions(user_id,session_id) VALUES (?,?)",
                    (user_id, session_id),
                )
            yield
        finally:
            lock.release()

    def memory(self, user_id: str, session_id: str) -> tuple[str, int]:
        with self.connect() as db:
            row = db.execute(
                "SELECT summary,through_id FROM sessions WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).fetchone()
        return (row["summary"], row["through_id"]) if row else ("", 0)

    def save_memory(self, user_id: str, session_id: str, summary: str, through_id: int):
        with self.connect() as db:
            db.execute(
                "UPDATE sessions SET summary=?,through_id=? WHERE user_id=? AND session_id=?",
                (summary, through_id, user_id, session_id),
            )

    def turns(self, user_id: str, session_id: str, after_id: int = 0) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM turns WHERE user_id=? AND session_id=? AND id>? ORDER BY id",
                (user_id, session_id, after_id),
            ).fetchall()
        return [{**dict(row), "messages": json.loads(row["messages"])} for row in rows]

    def start_turn(self, user_id: str, session_id: str, messages: list[dict]) -> int:
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO turns(user_id,session_id,messages,status) VALUES (?,?,?,'running')",
                (user_id, session_id, json.dumps(messages, ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def save_turn(self, turn_id: int, messages: list[dict], status: str = "running"):
        with self.connect() as db:
            db.execute(
                "UPDATE turns SET messages=?,status=? WHERE id=?",
                (json.dumps(messages, ensure_ascii=False), status, turn_id),
            )

    @staticmethod
    def _todo(row: sqlite3.Row) -> dict:
        return {"id": row["id"], "title": row["title"], "completed": bool(row["completed"])}

    def add_todo(self, user_id: str, session_id: str, title: str) -> dict:
        with self.connect() as db:
            # 单事务内查重与新增，避免模型重试造成同名未完成待办重复。
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM todos WHERE user_id=? AND session_id=? AND title=? AND completed=0",
                (user_id, session_id, title),
            ).fetchone()
            if row:
                return self._todo(row)
            count = db.execute(
                "SELECT count(*) FROM todos WHERE user_id=? AND session_id=?", (user_id, session_id)
            ).fetchone()[0]
            if count >= 100:
                raise ValueError("当前 session 最多保存 100 条待办")
            cursor = db.execute(
                "INSERT INTO todos(user_id,session_id,title) VALUES (?,?,?)",
                (user_id, session_id, title),
            )
            return {"id": cursor.lastrowid, "title": title, "completed": False}

    def list_todos(self, user_id: str, session_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM todos WHERE user_id=? AND session_id=? ORDER BY id",
                (user_id, session_id),
            ).fetchall()
        return [self._todo(row) for row in rows]

    def complete_todo(self, user_id: str, session_id: str, todo_id: int) -> dict | None:
        with self.connect() as db:
            db.execute(
                "UPDATE todos SET completed=1 WHERE user_id=? AND session_id=? AND id=?",
                (user_id, session_id, todo_id),
            )
            row = db.execute(
                "SELECT * FROM todos WHERE user_id=? AND session_id=? AND id=?",
                (user_id, session_id, todo_id),
            ).fetchone()
            return self._todo(row) if row else None

    def trace(self, user_id: str, session_id: str, event: str, **metadata):
        # 仅由 runtime 传递白名单元数据；不记录请求正文、工具参数或模型思考。
        with self.connect() as db:
            db.execute(
                "INSERT INTO traces(user_id,session_id,event,metadata) VALUES (?,?,?,?)",
                (user_id, session_id, event, json.dumps(metadata, ensure_ascii=False)),
            )

    def traces(self, user_id: str, session_id: str, limit: int = 20) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT created_at,event,metadata FROM traces WHERE user_id=? AND session_id=? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, session_id, limit),
            ).fetchall()
        return [{**dict(row), "metadata": json.loads(row["metadata"])} for row in reversed(rows)]
