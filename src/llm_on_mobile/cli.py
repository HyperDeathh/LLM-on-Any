from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from datetime import datetime

from .models_registry import load_models, Model, resolve_model_url
from .downloader import download_file
from .runner import LLMRunner
from .config import Config, Settings
from .history import list_chats, new_chat, save_chat, Message

SUPPORT_URL = "https://buymeacoffee.com/hyperdeath"
DISCORD_HANDLE = "rhaegarrr"

app = typer.Typer(
    help=(
        "LLMonAny (LLM on Any): list, download, and run open-weight LLMs.\n"
        "Help: 'lom help' | Support: " + SUPPORT_URL + " | Discord: " + DISCORD_HANDLE
    )
)


def data_dir() -> Path:
    base = os.environ.get("LOM_HOME") or (Path.home() / ".llmonany")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    (p / "models").mkdir(exist_ok=True)
    return p


def models_dir() -> Path:
    return data_dir() / "models"


@app.command()
def list():
    table = Table(title="Available Models")
    table.add_column("#", justify="right")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Size")
    table.add_column("Status")
    table.add_column("Source")

    models = load_models()
    for idx, m in enumerate(models, start=1):
        dest = models_dir() / f"{m.id}.{m.format}"
        status = "downloaded" if dest.exists() else "not downloaded"
        src = m.hf_repo or ("direct" if m.url else "")
        if m.gated:
            src += " (gated)"
        table.add_row(str(idx), m.id, m.name, m.size, status, src)

    print(table)
    print("\nHelp: lom help | Support: ", SUPPORT_URL, "| Discord:", DISCORD_HANDLE)


@app.command()
def download(n: int = typer.Argument(..., help="Model number from the list")):
    models = load_models()
    if n < 1 or n > len(models):
        raise typer.BadParameter("Invalid model number")
    m = models[n - 1]
    dest = models_dir() / f"{m.id}.{m.format}"
    url, gated = resolve_model_url(m)
    if not url:
        print("This model requires gated access or lacks a direct URL.")
        if m.hf_repo:
            print("Please accept terms and/or login on Hugging Face and then use:")
            print(f"  huggingface-cli download {m.hf_repo} --include \"{m.hf_filename or '*'}\" --local-dir {models_dir()}\n")
        raise typer.Exit(1)

    print(f"Downloading {m.name} -> {dest}")
    download_file(url, dest, m.sha256)
    print("Done.")
    print("\nSupport: ", SUPPORT_URL, "| Questions: Discord:", DISCORD_HANDLE)


@app.command()
def chat(
    n: int = typer.Argument(..., help="Model number from the list"),
    prompt: str = typer.Option("", "--prompt", "-p", help="Single-shot quick prompt. If empty, opens interactive menu."),
):
    models = load_models()
    if n < 1 or n > len(models):
        raise typer.BadParameter("Invalid model number")
    m: Model = models[n - 1]
    path = models_dir() / f"{m.id}.{m.format}"
    if not path.exists():
        raise typer.BadParameter("Model not downloaded. Run: lom download <n>")

    cfg = Config.load()
    cfg.set_active_model(m.id)
    cfg.save()

    # Quick mode: if a single prompt is provided, generate once and exit
    if prompt:
        s = cfg.get_model_settings(m.id)
        runner = LLMRunner(path, ctx_size=s.ctx_size, n_threads=s.threads)
        print(f"Loading {m.name}...")
    # Include system prompt in the first turn for a minimal multi-turn effect
        sys = s.system_prompt
        final_prompt = f"[SYSTEM]\n{sys}\n\n[USER]\n{prompt}\n\n[ASSISTANT]"
        text = runner.generate(
            final_prompt,
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
        )
        print("\n[bold]Response:[/bold]\n")
        print(text)
        return

    # Interactive menu
    while True:
        print("\n[bold]LLMonAny[/bold] - Model:", m.name)
        print("1) New chat\n2) Past chats\n3) Settings\n4) Exit")
        print("[dim]Help: lom help | Support: ", SUPPORT_URL, "| Discord:", DISCORD_HANDLE, "[/dim]")
        try:
            choice = IntPrompt.ask("Select", choices=["1", "2", "3", "4"], default=1)
        except Exception:
            choice = 4
        if choice == 4:
            break
        if choice == 3:
            _settings_menu(cfg, m.id)
            continue
        if choice == 2:
            _past_chats_menu(m.id, path)
            continue
        if choice == 1:
            _new_chat_session(m.id, m.name, path)
            continue


def _settings_menu(cfg: Config, model_id: str) -> None:
    s = cfg.get_model_settings(model_id)
    print("\n[bold]Settings[/bold] (leave blank to keep)")
    try:
        max_tokens = Prompt.ask("max_tokens", default=str(s.max_tokens))
        temperature = Prompt.ask("temperature", default=str(s.temperature))
        top_k = Prompt.ask("top_k", default=str(s.top_k))
        top_p = Prompt.ask("top_p", default=str(s.top_p))
        ctx_size = Prompt.ask("ctx_size", default=str(s.ctx_size))
        threads = Prompt.ask("threads (None = auto)", default=str(s.threads) if s.threads is not None else "")
        system_prompt = Prompt.ask("system_prompt", default=s.system_prompt)

        ns = Settings(
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            ctx_size=int(ctx_size),
            threads=int(threads) if threads.strip() else None,
            system_prompt=system_prompt,
        )
        cfg.set_model_settings(model_id, ns)
        cfg.save()
    print("Saved.")
    except Exception as e:
    print(f"Error: {e}")


def _new_chat_session(model_id: str, model_name: str, model_path: Path) -> None:
    cfg = Config.load()
    s = cfg.get_model_settings(model_id)
    chat_title = Prompt.ask("Chat title", default=f"Chat with {model_name}")
    chat = new_chat(model_id, chat_title, s.system_prompt)
    runner = LLMRunner(model_path, ctx_size=s.ctx_size, n_threads=s.threads)
    print(f"Loading {model_name}...")
    while True:
        user = Prompt.ask("You", default="", show_default=False)
        if not user:
            print("(empty line = exit)")
            break
        chat.messages.append(Message(role="user", content=user, ts=datetime.utcnow().isoformat()))
    # Minimal prompt format: accumulate dialogue and send
        conv = []
        for msg in chat.messages:
            conv.append(f"[{msg.role.upper()}]\n{msg.content}")
        conv.append("[ASSISTANT]")
        prompt = "\n\n".join(conv)
        reply = runner.generate(
            prompt,
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
        )
        chat.messages.append(Message(role="assistant", content=reply, ts=datetime.utcnow().isoformat()))
    print(f"\n[bold]Assistant:[/bold] {reply}\n")
        save_chat(chat)


def _past_chats_menu(model_id: str, model_path: Path) -> None:
    chats = list_chats(model_id)
    if not chats:
        print("No saved chats.")
        return
    table = Table(title="Past Chats")
    table.add_column("#")
    table.add_column("Title")
    table.add_column("Messages")
    for i, ch in enumerate(chats, 1):
        table.add_row(str(i), ch.title, str(len(ch.messages)))
    print(table)
    try:
        sel = IntPrompt.ask("Open chat #", default=1)
    except Exception:
        return
    if sel < 1 or sel > len(chats):
        return
    # Resume-able: display and continue chatting
    chat = chats[sel - 1]
    cfg = Config.load()
    s = cfg.get_model_settings(model_id)
    runner = LLMRunner(model_path, ctx_size=s.ctx_size, n_threads=s.threads)
    print(f"\n[bold]{chat.title}[/bold]")
    for msg in chat.messages:
        if msg.role != "system":
            who = "You" if msg.role == "user" else "Assistant"
            print(f"{who}: {msg.content}")
    while True:
    user = Prompt.ask("You", default="", show_default=False)
        if not user:
            break
        chat.messages.append(Message(role="user", content=user, ts=datetime.utcnow().isoformat()))
        conv = []
        for msg in chat.messages:
            conv.append(f"[{msg.role.upper()}]\n{msg.content}")
        conv.append("[ASSISTANT]")
        prompt = "\n\n".join(conv)
        reply = runner.generate(
            prompt,
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
        )
        chat.messages.append(Message(role="assistant", content=reply, ts=datetime.utcnow().isoformat()))
    print(f"\n[bold]Assistant:[/bold] {reply}\n")
        save_chat(chat)


@app.command()
def help():
    """Detailed help and usage examples."""
    print(
        f"""
LLMonAny — Help

Commands:
    lom list                List models and their download status
    lom download <n>        Download model number n from the list
    lom chat <n>            Open menu (New chat / Past chats / Settings)
    lom chat <n> -p "..."   Single-shot prompt without menu
    lom registry            Show registry source priority
    lom doctor              Environment check (Python, data dir, deps)
    lom paths               Show important directories and files

Examples:
    lom list
    lom download 3
    lom chat 3
    lom chat 3 -p "Hello!"

Settings:
    max_tokens, temperature, top_k, top_p, ctx_size, threads, system_prompt
    Per-model overrides are saved. Defaults apply if you don’t change them.

Storage:
    Data dir: $LOM_HOME (default ~/.llmonany)
    Models:   $LOM_HOME/models
    Config:   $LOM_HOME/config.json
    History:  $LOM_HOME/history/*.json

Tips:
    - Use 'lom doctor' for quick diagnostics.
    - Mind disk/RAM for large models.

Support: {SUPPORT_URL}
Discord: {DISCORD_HANDLE}
GitHub: please open an Issue in the repo.
"""
    )


@app.command()
def doctor():
    """Environment check: Python version, data directory, optional deps."""
    import sys
    print("Python:", sys.version)
    print("Data dir:", data_dir())
    try:
        import llama_cpp  # type: ignore
        print("llama-cpp-python: OK")
    except Exception as e:
        print("llama-cpp-python: not installed (optional)")


@app.command()
def paths():
    """Show important paths (data dir, models dir, config, history)."""
    dd = data_dir()
    print("LOM_HOME:", dd)
    print("Models:", models_dir())
    from .config import CONFIG_PATH
    print("Config:", CONFIG_PATH)
    from .history import HISTORY_DIR
    print("History:", HISTORY_DIR)


@app.command("registry")
def registry_info():
    """Show where the model registry is loaded from and how to override."""
    print("Registry sources priority:\n1) $LOM_REGISTRY\n2) $LOM_HOME/models.json\n3) ./models.json\n4) built-in")
    print("\nTip: Set LOM_HOME to choose data dir. Put models.json there to customize.")


if __name__ == "__main__":
    app()
