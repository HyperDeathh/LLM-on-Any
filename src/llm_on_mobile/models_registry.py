from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import os
from pathlib import Path
try:
    from huggingface_hub import hf_hub_url  # optional
except Exception:  # pragma: no cover
    hf_hub_url = None  # type: ignore

# Simple registry: replace with your curated list when you provide it.
# Each model has a display name, family, quantization, size label, and a download URL.

@dataclass
class Model:
    id: str
    name: str
    size: str
    family: str
    format: str  # gguf, safetensors, etc.
    url: str  # direct URL or blank if gated
    sha256: Optional[str] = None
    # Optional HF info for building a URL
    hf_repo: Optional[str] = None  # e.g., meta-llama/Llama-3.1-8B-Instruct
    hf_filename: Optional[str] = None  # exact file name
    gated: bool = False


DEFAULT_MODELS: List[Model] = [
    # Practical mobile-ready quant (community)
    Model(
        id="llama-3.1-8b-q4",
        name="Llama 3.1 8B Instruct Q4_0",
        size="~4.5 GB",
        family="llama",
        format="gguf",
        url="https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct.Q4_0.gguf?download=true",
        sha256=None,
    ),
    # User-provided hubs (gated or multi-file). Use HF CLI after accepting terms.
    Model(
        id="gpt-oss-20b",
        name="GPT-OSS 20B (original, multi-file)",
        size="varies",
        family="gpt-oss",
        format="safetensors",
        url="",
        sha256=None,
        hf_repo="openai/gpt-oss-20b",
        hf_filename=None,
        gated=False,
    ),
    Model(
        id="gpt-oss-120b",
        name="GPT-OSS 120B (original, multi-file)",
        size=">100 GB",
        family="gpt-oss",
        format="safetensors",
        url="",
        sha256=None,
        hf_repo="openai/gpt-oss-120b",
        hf_filename=None,
        gated=False,
    ),
    Model(
        id="grok-2",
        name="Grok-2 (original, multi-file)",
        size="~500 GB",
        family="grok",
        format="safetensors",
        url="",
        sha256=None,
        hf_repo="xai-org/grok-2",
        hf_filename=None,
        gated=True,
    ),
    Model(
        id="llama-3.3-70b-instruct",
        name="Llama 3.3 70B Instruct (original, multi-file)",
        size=">= 140 GB",
        family="llama",
        format="safetensors",
        url="",
        sha256=None,
        hf_repo="meta-llama/Llama-3.3-70B-Instruct",
        hf_filename=None,
        gated=True,
    ),
    Model(
        id="llama-4-scout-17b-16e-instruct",
        name="Llama 4 Scout 17B 16E Instruct (original)",
        size=">= 30 GB",
        family="llama",
        format="safetensors",
        url="",
        sha256=None,
        hf_repo="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        hf_filename=None,
        gated=True,
    ),
    Model(
        id="llama-4-maverick-17b-128e-instruct",
        name="Llama 4 Maverick 17B 128E Instruct (original)",
        size=">= 30 GB",
        family="llama",
        format="safetensors",
        url="",
        sha256=None,
        hf_repo="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        hf_filename=None,
        gated=True,
    ),
    Model(
        id="llama-3.1-8b-instruct",
        name="Llama 3.1 8B Instruct (original)",
        size=">= 15 GB",
        family="llama",
        format="safetensors",
        url="",
        sha256=None,
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        hf_filename=None,
        gated=True,
    ),
]


def _data_dir() -> Path:
    base = os.environ.get("LOM_HOME") or (Path.home() / ".llmonany")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_models_json_from(path: Path) -> Optional[List[Model]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        models: List[Model] = []
        for item in data:
            models.append(
                Model(
                    id=item["id"],
                    name=item["name"],
                    size=item.get("size", ""),
                    family=item.get("family", ""),
                    format=item.get("format", "gguf"),
                    url=item.get("url", ""),
                    sha256=item.get("sha256"),
                    hf_repo=item.get("hf_repo"),
                    hf_filename=item.get("hf_filename"),
                    gated=item.get("gated", False),
                )
            )
        return models
    except Exception:
        return None


def load_models() -> List[Model]:
    """
    Load models from (priority):
    - $LOM_REGISTRY if set (file path to JSON)
    - $LOM_HOME/models.json
    - ./models.json (current working directory)
    - Built-in DEFAULT_MODELS
    """
    # Env override
    env_path = os.environ.get("LOM_REGISTRY")
    if env_path:
        models = _load_models_json_from(Path(env_path))
        if models:
            return models

    # Data dir
    models = _load_models_json_from(_data_dir() / "models.json")
    if models:
        return models

    # CWD
    models = _load_models_json_from(Path.cwd() / "models.json")
    if models:
        return models

    # Fallback
    return DEFAULT_MODELS


def resolve_model_url(m: Model) -> Tuple[Optional[str], bool]:
    """Return (url, gated). If gated is True, user must accept terms on HF.
    Preference order: direct url -> (hf_repo+hf_filename) -> None.
    """
    if m.url:
        return m.url, m.gated
    if m.hf_repo and m.hf_filename and hf_hub_url is not None:
        try:
            url = hf_hub_url(repo_id=m.hf_repo, filename=m.hf_filename)
            return url, m.gated
        except Exception:
            return None, True
    return None, m.gated or False
