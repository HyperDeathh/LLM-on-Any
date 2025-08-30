from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


class TransformersRunner:
    """
    Minimal runner using Hugging Face Transformers for safetensors/original models.
    Loads from a local directory or directly from a repo ID (HF cache).

    Requirements (install optional extra):
      pip install -e .[hf]
    """

    def __init__(
        self,
        source: Union[str, Path],  # HF repo_id or local directory
        ctx_size: int = 4096,
        n_threads: Optional[int] = None,  # not used directly; HF/torch manages threads
        device: str = "cpu",  # cpu|cuda|mps|auto
    ):
        self.source = source
        self.ctx_size = ctx_size
        self.n_threads = n_threads
        self.device = device
        self._model = None
        self._tokenizer = None

    def load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Transformers is not installed. Install extra: pip install -e .[hf]"
            ) from e

        # Decide device
        import torch
        dev = self.device
        if dev == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"

        # Load tokenizer & model
        self._tokenizer = AutoTokenizer.from_pretrained(self.source, use_fast=True)
        kwargs = {}
        # For explicit single-device placement without accelerate
        if dev in ("cpu", "cuda", "mps"):
            kwargs["device_map"] = {"": dev}
        else:
            # fallback to auto mapping if something else is provided
            kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.source, torch_dtype=None, **kwargs)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> str:
        if self._model is None or self._tokenizer is None:
            self.load()

        tok = self._tokenizer
        model = self._model

        inputs = tok(prompt, return_tensors="pt")

        # Move tensors to model device if not auto-handled
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=getattr(tok, "eos_token_id", None),
            pad_token_id=getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None),
        )
        text = tok.decode(gen_out[0], skip_special_tokens=True)
        # Return only the assistant continuation beyond the original prompt
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
