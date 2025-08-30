from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


def data_dir() -> Path:
    base = os.environ.get("LOM_HOME") or (Path.home() / ".llmonany")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


CONFIG_PATH = data_dir() / "config.json"


@dataclass
class Settings:
    # Generation defaults
    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    ctx_size: int = 4096
    threads: Optional[int] = None
    system_prompt: str = "You are a helpful assistant."
    # Device preference for Transformers (cpu|cuda|mps|auto)
    device: str = "cpu"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Settings":
        s = Settings()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


class Config:
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data: Dict[str, Any] = data or {}
        # global defaults
        self.data.setdefault("defaults", asdict(Settings()))
        # per-model overrides
        self.data.setdefault("models", {})
        # last active model
        self.data.setdefault("active_model_id", None)

    def get_defaults(self) -> Settings:
        return Settings.from_dict(self.data.get("defaults", {}))

    def set_defaults(self, settings: Settings) -> None:
        self.data["defaults"] = asdict(settings)

    def get_model_settings(self, model_id: str) -> Settings:
        base = self.get_defaults()
        override = self.data.get("models", {}).get(model_id, {})
        merged = asdict(base)
        merged.update(override)
        return Settings.from_dict(merged)

    def set_model_settings(self, model_id: str, settings: Settings) -> None:
        self.data.setdefault("models", {})[model_id] = asdict(settings)

    def get_active_model(self) -> Optional[str]:
        return self.data.get("active_model_id")

    def set_active_model(self, model_id: Optional[str]) -> None:
        self.data["active_model_id"] = model_id

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    @staticmethod
    def load() -> "Config":
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                return Config(data)
            except Exception:
                pass
        return Config()
