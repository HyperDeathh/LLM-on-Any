from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover
    Llama = None  # type: ignore


class LLMRunner:
    def __init__(self, model_path: Path, ctx_size: int = 4096, n_threads: Optional[int] = None):
        self.model_path = Path(model_path)
        self.ctx_size = ctx_size
        self.n_threads = n_threads
        self._llm = None

    def load(self):
        if Llama is None:
            raise RuntimeError("llama-cpp-python is not installed. Install extra: pip install .[llama]")
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.ctx_size,
            n_threads=self.n_threads or 0,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> str:
        if not self._llm:
            self.load()
        out = self._llm(  # type: ignore
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return out.get("choices", [{}])[0].get("text", "").strip()
