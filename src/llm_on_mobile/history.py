from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import data_dir


HISTORY_DIR = data_dir() / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Message:
    role: str  # system|user|assistant
    content: str
    ts: str


@dataclass
class Chat:
    id: str
    model_id: str
    title: str
    messages: List[Message]

    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "title": self.title,
            "messages": [asdict(m) for m in self.messages],
        }

    @staticmethod
    def from_dict(d: Dict) -> "Chat":
        return Chat(
            id=d["id"],
            model_id=d["model_id"],
            title=d.get("title", ""),
            messages=[Message(**m) for m in d.get("messages", [])],
        )


def _chat_path(chat_id: str) -> Path:
    return HISTORY_DIR / f"{chat_id}.json"


def list_chats(model_id: str | None = None) -> List[Chat]:
    chats: List[Chat] = []
    for p in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            chat = Chat.from_dict(data)
            if not model_id or chat.model_id == model_id:
                chats.append(chat)
        except Exception:
            continue
    chats.sort(key=lambda c: c.messages[0].ts if c.messages else "", reverse=True)
    return chats


def new_chat(model_id: str, title: str, system_prompt: str) -> Chat:
    now = datetime.utcnow().isoformat()
    chat_id = now.replace(":", "").replace("-", "").replace(".", "")
    chat = Chat(id=chat_id, model_id=model_id, title=title, messages=[])
    chat.messages.append(Message(role="system", content=system_prompt, ts=now))
    save_chat(chat)
    return chat


def save_chat(chat: Chat) -> None:
    _chat_path(chat.id).write_text(json.dumps(chat.to_dict(), indent=2), encoding="utf-8")
