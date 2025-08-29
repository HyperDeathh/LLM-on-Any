# LLMonAny (LLM on Any)

A modern, open-source CLI to list, download, and run open-weight LLMs on any device, with a mobile-first mindset.

- Android: Termux (recommended)
- Windows/macOS/Linux: Python 3.9+
- iOS: Python-only is limited; a native Swift + llama.cpp (Metal) bridge is planned

Support: https://buymeacoffee.com/hyperdeath — Discord: rhaegarrr — Please open GitHub Issues for questions.

## Features

- Model list and download status (Downloaded/Not downloaded)
- Download by number and start chatting immediately
- Menu flow: New chat, Past chats, Settings
- Per-model settings: max_tokens, temperature, top_k, top_p, ctx_size, threads, system_prompt
- Chat history persisted as JSON

## Install (Desktop)

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -e .
# Local inference (optional):
pip install -e .[llama]
```

## Quickstart

```powershell
lom list
lom download 3
lom chat 3
# or a quick single message
lom chat 3 -p "Hello there!"
```

## Android (Termux)

```bash
pkg update -y && pkg upgrade -y
pkg install -y python clang cmake git
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install llm-on-mobile
# Local inference (may compile):
pip install llama-cpp-python || echo "Skipping"

# Use
lom list
lom download 3
lom chat 3
```

Notes
- On some devices `llama-cpp-python` compiles from source; we’ll add a direct `llama.cpp` binary adapter next.
- Data dir: `$LOM_HOME` (default `~/.llmonany`). Models live under `models/`.

## Commands

- `lom help` — Detailed help and examples
- `lom list` — List models and status
- `lom download <n>` — Download model `n`
- `lom chat <n>` — Menu (New chat / Past chats / Settings)
- `lom chat <n> -p "..."` — Single prompt without menu
- `lom registry` — Registry source priority
- `lom doctor` — Environment check
- `lom paths` — Important paths

## Model Registry

Priority:
1) `LOM_REGISTRY` (file path)
2) `$LOM_HOME/models.json`
3) `./models.json`
4) Built-in defaults

Example `models.json` entry:

```json
[
	{
		"id": "llama-3.1-8b-q4",
		"name": "Llama 3.1 8B Instruct Q4_0",
		"size": "~4.5 GB",
		"family": "llama",
		"format": "gguf",
		"url": "https://…/model.gguf?download=true",
		"sha256": null
	}
]
```

### About the provided models

You shared these hubs:
- xai-org/grok-2
- meta-llama/Llama-3.3-70B-Instruct
- meta-llama/Llama-4-Scout-17B-16E-Instruct
- meta-llama/Llama-4-Maverick-17B-128E-Instruct
- meta-llama/Llama-3.1-8B-Instruct
- openai/gpt-oss-120b
- openai/gpt-oss-20b

Notes:
- Many Meta Llama hubs are gated. You must accept terms and be logged in on HF to download.
- gpt-oss hubs provide “original/*” with Hugging Face CLI; consumer quant GGUFs are often in community repos.
- grok-2 is huge (~500GB) and intended for multi-GPU servers via SGLang.

We will include them in `models.json` as either direct URLs (if available) or as HF repo + filename. If gated, `lom download` will guide you to use `huggingface-cli` after accepting terms.

## Multi-device use

- Point `LOM_HOME` to a shared location to re-use config/history across devices.
- Models are large; consider per-device download or external storage.

## Roadmap

- Android `llama.cpp` binary adapter (no Python binding)
- Minimal HTTP server mode (mobile browser chat)
- Model/device capability hints
- CI/CD and PyPI release workflow

## Support & Contact

- Buy Me a Coffee: https://buymeacoffee.com/hyperdeath
- Discord: rhaegarrr
- GitHub Issues: please open an issue for questions/bugs/ideas.

## License

MIT
