from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Optional
import fnmatch
import requests

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from datetime import datetime

from .models_registry import load_models, Model, resolve_model_url
from .downloader import download_file
from .runner import LLMRunner
from .transformers_runner import TransformersRunner
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

# Shared Rich console instance
console = Console()

def _render_chat_header(title: str, subtitle: str, device: str, ctx_size: int) -> None:
    """Clear the console and render a modern header for chat sessions."""
    try:
        console.clear()
    except Exception:
        # Fallback for environments without full terminal support
        os.system("cls" if os.name == "nt" else "clear")
    header = Panel(
        f"[bold white]{title}[/bold white]\n[dim]{subtitle}[/dim]\n[dim]Device: {device} • Context: {ctx_size}[/dim]",
        title="LLMonAny",
        border_style="magenta",
        box=box.ROUNDED,
    )
    print(header)
    print("[dim]Press Enter on empty line to exit.[/dim]")


def data_dir() -> Path:
    base = os.environ.get("LOM_HOME") or (Path.home() / ".llmonany")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    (p / "models").mkdir(exist_ok=True)
    return p


def models_dir() -> Path:
    return data_dir() / "models"


def _order_key(m: Model):
    fam_rank = {
        "llama": 0,
        "deepseek": 1,
        "grok": 2,
    }.get(m.family, 9)
    fmt_rank = 0 if m.format.lower() == "gguf" else 1
    # Prefer smaller models first within groups for practicality
    size_hint = 0 if ("~" in m.size or "GB" in m.size) else 1
    return (fmt_rank, fam_rank, size_hint, m.name.lower())


def _sorted_models() -> list[Model]:
    # Kept for future use; default flow uses file order to match models.json
    try:
        return sorted(load_models(), key=_order_key)
    except Exception:
        return load_models()


def _hf_snapshot(repo: str, include: Optional[str], dest_dir: Path, default_exclude: bool = False) -> Path:
    """Download HF repo files into dest_dir/<repo_name> with detailed per-file progress.
    If include is None, try to download all files; otherwise respect glob/comma-separated patterns.
    """
    have_hf = True
    try:
        from huggingface_hub import list_repo_files, hf_hub_url, snapshot_download
        from tqdm.auto import tqdm as _tqdm  # fallback for snapshot
    except Exception:
        have_hf = False

    local = dest_dir / repo.replace("/", "__")
    local.mkdir(parents=True, exist_ok=True)

    # Build include pattern list (None or list of globs)
    include_globs: Optional[list[str]] = None
    if include:
        pats = [p.strip() for p in include.split(",") if p.strip()]
        if any(p in ("*", "**/*") for p in pats):
            include_globs = None  # means all files
        else:
            # dedupe while preserving order
            seen = set(); globs: list[str] = []
            for p in pats:
                if p not in seen:
                    seen.add(p); globs.append(p)
            include_globs = globs

    # Default excludes (applied only when we're doing a full snapshot by default)
    ignore_globs: list[str] = []
    if default_exclude and include_globs is None:
        # Skip common non-essential files by default
        ignore_globs = [
            ".gitattributes",
            "LICENSE",
            "README.md",
            "USAGE_POLICY",
            "metal/**",
            "chat_template.jinja",
        ]

    # Try listing files to do per-file downloads with our progress bars
    all_files = None
    if have_hf:
        try:
            all_files = list_repo_files(repo_id=repo)
        except Exception:
            all_files = None

    # If we were able to list files, conditionally prefer shards over 'original' single-file
    shards_present = False
    if default_exclude and all_files:
        for f in all_files:
            if f.endswith('.safetensors') and '-of-' in f:
                shards_present = True
                break
        if not shards_present:
            # presence of index json also implies sharded setup
            shards_present = any(f.endswith('model.safetensors.index.json') for f in all_files)
        if shards_present:
            ignore_globs.append('original/**')

    if all_files is None:
        if have_hf:
            # Fallback: use snapshot (shows at least a global progress bar)
            try:
                snapshot_download(
                    repo_id=repo,
                    local_dir=str(local),
                    local_dir_use_symlinks=False,
                    allow_patterns=None if include_globs is None else include_globs,
                    repo_type=None,
                    ignore_patterns=ignore_globs or None,
                    cache_dir=str(data_dir() / "hf-cache"),
                    tqdm_class=_tqdm,
                )
            except Exception as e:
                raise typer.BadParameter(
                    f"Failed to fetch snapshot for {repo}. It may be gated, nonexistent, or you may need to login/accept terms on HF. Error: {e}"
                )
            return local
        else:
            # No huggingface_hub available. Allow a limited mode only if include_globs
            # are explicit file paths (no wildcards). We'll construct direct URLs:
            # https://huggingface.co/<repo>/resolve/main/<path>
            if include_globs is None:
                raise typer.BadParameter(
                    "huggingface_hub is not installed. For full snapshots install it, or pass -i with exact filenames (no wildcards)."
                )
            # Validate patterns: must not contain wildcards
            wild = [p for p in include_globs if any(ch in p for ch in "*?[")]
            if wild:
                raise typer.BadParameter(
                    "When huggingface_hub is missing, -i must list exact file paths without wildcards."
                )
            # Proceed to direct downloads for each listed file
            for rel in include_globs:
                url = f"https://huggingface.co/{repo}/resolve/main/{rel}?download=true"
                dest_path = local / rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                download_file(url, dest_path)
            return local

    # Filter by patterns
    if include_globs is None:
        selected = list(all_files)
    else:
        selected = []
        for f in all_files:
            if any(fnmatch.fnmatch(f, pat) for pat in include_globs):
                selected.append(f)
    # Apply ignore filters if any
    if ignore_globs:
        selected = [f for f in selected if not any(fnmatch.fnmatch(f, pat) for pat in ignore_globs)]
    # Remove duplicates, keep order
    seen = set(); filtered: list[str] = []
    for f in selected:
        if f not in seen:
            seen.add(f); filtered.append(f)

    if not filtered:
        raise typer.BadParameter("No files matched the include filters for this repo.")

    # Try to get sizes using repo_info; if missing/zero, fallback to HEAD per file
    sizes: dict[str, int] = {}
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.repo_info(repo_id=repo, files_metadata=True)
        # info.siblings is a list; each has rfilename and size (may be None)
        for s in getattr(info, "siblings", []) or []:
            rfn = getattr(s, "rfilename", None)
            sz = getattr(s, "size", None)
            if isinstance(rfn, str) and isinstance(sz, int):
                sizes[rfn] = sz
    except Exception:
        pass

    # HEAD fallback for missing or zero sizes
    for rel in filtered:
        if rel in sizes and isinstance(sizes[rel], int) and sizes[rel] > 0:
            continue
        try:
            url = hf_hub_url(repo_id=repo, filename=rel)
            resp = requests.head(url, allow_redirects=True, timeout=20)
            clen = resp.headers.get("Content-Length")
            if clen and clen.isdigit():
                sizes[rel] = int(clen)
        except Exception:
            pass

    def _hsize(n: int | None) -> str:
        if not n or n <= 0:
            return "?"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024 or unit == "TB":
                return f"{n:.0f}{unit}" if unit == "B" else f"{n/1024:.1f}{unit}"
            n /= 1024
        return f"{n:.1f}TB"

    # Build plan entries and sort by size asc (unknown sizes last)
    plan = []
    total = 0
    for rel in filtered:
        sz = sizes.get(rel)
        total += sz or 0
        plan.append((rel, sz))
    plan.sort(key=lambda x: (x[1] is None, x[1] or 0))

    # Pretty print the plan (English, single table only)
    try:
        from rich.table import Table as _Table
        tbl = _Table(title=f"Files to download ({len(plan)})")
        tbl.add_column("#", justify="right")
        tbl.add_column("File")
        tbl.add_column("Size", justify="right")
        for i, (rel, sz) in enumerate(plan, 1):
            tbl.add_row(str(i), rel, _hsize(sz))
        print(tbl)
        if total:
            print(f"Estimated total: {_hsize(total)}")
        if ignore_globs:
            print("[dim]Note: Skipped by default:", ", ".join(ignore_globs), "[/dim]")
            if shards_present:
                print("[dim]Reason: Shards detected, so 'original/**' alternative set is omitted.[/dim]")
    except Exception:
        # Suppress secondary list output; keep quiet on failure to render table
        pass

    # Download in the sorted order with per-file progress
    for rel, _sz in plan:
        try:
            url = hf_hub_url(repo_id=repo, filename=rel)
        except Exception as e:
            raise typer.BadParameter(f"Failed to resolve URL for {rel}: {e}")
        dest_path = local / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Use our idempotent downloader (skips existing)
        download_file(url, dest_path)

    return local


def _resolve_local_source(m: Model) -> Path:
    """Return expected local path for the model (file for gguf, folder for repo)."""
    if m.format.lower() == "gguf":
        return models_dir() / f"{m.id}.{m.format}"
    if m.hf_repo:
        return models_dir() / m.hf_repo.replace("/", "__")
    # Fallback: treat as a file
    return models_dir() / f"{m.id}.{m.format}"


def _hf_list_files(repo: str) -> Optional[list[str]]:
    """Return list of files in HF repo or None if unavailable."""
    try:
        from huggingface_hub import list_repo_files
    except Exception:
        return None
    try:
        return list_repo_files(repo_id=repo)
    except Exception:
        return None


def _validate_transformers_local_dir(folder: Path) -> tuple[Optional[str], bool]:
    """Return (message, missing_weights).
    message is None if ok; otherwise a human message listing what's missing.
    We check for: config.json, tokenizer (tokenizer.json or tokenizer.model or vocab+merges), and weights (*.safetensors or pytorch_model.bin).
    """
    if not folder.exists() or not folder.is_dir():
        return "Model folder missing.", True
    files = {p.name for p in folder.glob("*") if p.is_file()}
    # Config
    has_config = "config.json" in files
    # Tokenizer
    has_tok = (
        "tokenizer.json" in files
        or "tokenizer.model" in files
        or ("vocab.json" in files and "merges.txt" in files)
    )
    # Weights
    has_weights = any(p.suffix == ".safetensors" for p in folder.glob("*.safetensors")) or (folder / "pytorch_model.bin").exists()
    missing = []
    if not has_config:
        missing.append("config.json")
    if not has_tok:
        missing.append("tokenizer.json veya tokenizer.model (ya da vocab.json + merges.txt)")
    if not has_weights:
        missing.append("*.safetensors (veya pytorch_model.bin)")
    if missing:
        return (
            "Dizin eksik dosyalar içeriyor: " + ", ".join(missing),
            ("*.safetensors" in missing[0]) or ("*.safetensors" in ", ".join(missing)),
        )
    return None, False


def _missing_report(folder: Path, repo: Optional[str]) -> str:
    """Return a human-readable detailed report of present/missing files.
    If repo is provided, include expected shards from remote listing for comparison.
    """
    lines: list[str] = []
    lines.append("[bold]Eksik dosya teşhisi[/bold]")
    if not folder.exists():
        lines.append(f"Klasör yok: {folder}")
        return "\n".join(lines)

    files = {p.name for p in folder.glob("*") if p.is_file()}
    def has(name: str) -> bool:
        return name in files

    # Config and tokenizer
    lines.append("- Config: " + ("VAR (config.json)" if has("config.json") else "YOK (config.json)"))
    tok_bits = []
    if has("tokenizer.json"):
        tok_bits.append("tokenizer.json")
    if has("tokenizer.model"):
        tok_bits.append("tokenizer.model")
    if has("vocab.json"):
        tok_bits.append("vocab.json")
    if has("merges.txt"):
        tok_bits.append("merges.txt")
    if tok_bits:
        lines.append("- Tokenizer: VAR (" + ", ".join(tok_bits) + ")")
    else:
        lines.append("- Tokenizer: YOK (tokenizer.json veya tokenizer.model ya da vocab.json+merges.txt)")

    # Local weights
    local_weights = [p.name for p in folder.glob("*.safetensors")]
    if (folder / "pytorch_model.bin").exists():
        local_weights.append("pytorch_model.bin")
    if local_weights:
        lines.append("- Ağırlıklar (yerel): " + ", ".join(sorted(local_weights)))
    else:
        lines.append("- Ağırlıklar (yerel): YOK")

    # Remote expectation (if available)
    if repo:
        remote = _hf_list_files(repo) or []
        remote_single = [f for f in remote if f.endswith(".safetensors") and "-of-" not in f]
        remote_shards = [f for f in remote if f.endswith(".safetensors") and "-of-" in f]
        if remote_single:
            lines.append("- Uzaktaki tek-parça aday(lar): " + ", ".join(remote_single))
        if remote_shards:
            # Compare shard coverage
            missing = sorted(set(remote_shards) - set(local_weights))
            present = sorted(set(remote_shards) & set(local_weights))
            lines.append(f"- Uzaktaki shard sayısı: {len(remote_shards)} | Yerelde: {len(present)} mevcut, {len(missing)} eksik")
            # Show up to 10 missing
            if missing:
                show = ", ".join(missing[:10]) + (" …" if len(missing) > 10 else "")
                lines.append("  Eksik shard örnekleri: " + show)

    return "\n".join(lines)


def _download_model(m: Model, include_override: Optional[str] = None) -> Path:
    dest = _resolve_local_source(m)
    url, _ = resolve_model_url(m)
    if url:
        print(f"Downloading {m.name} -> {dest}")
        download_file(url, dest, m.sha256)
        print("Done.")
        return dest
    if m.hf_repo:
        # Essentials we try to always include for Transformers runs
        essentials = [
            "tokenizer.json",
            "config.json",
            "tokenizer.model",
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
        ]
        files = _hf_list_files(m.hf_repo)

        include = include_override if include_override is not None else None

        if include:
            # Merge user include with essentials to avoid missing tokenizer/config
            pats = [p.strip() for p in include.split(",") if p.strip()]
            need_add = []
            has_tok = any(p in ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt") for p in pats)
            has_cfg = any(p == "config.json" for p in pats)
            if not has_tok:
                need_add.append("tokenizer.json")
            if not has_cfg:
                need_add.append("config.json")
            if need_add:
                include = ",".join(pats + need_add)
        else:
            # No include provided: download FULL repository snapshot (all files)
            include = None
            print("[info] Tüm repo indirilecek (full snapshot). Büyük olabilir.")
        print(f"Fetching Hugging Face snapshot: {m.hf_repo} (include={include or '*'})")
        # Apply default excludes for well-known non-essential files/folders
        local = _hf_snapshot(m.hf_repo, include, models_dir(), default_exclude=True)
        print("Downloaded to:", local)
        return local
    raise typer.BadParameter("Model has no direct URL or repo to download from.")


@app.command("list")
def list_models():
    table = Table(title="Available Models", box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("#", justify="right", style="bold")
    table.add_column("ID", max_width=26, overflow="fold")
    table.add_column("Name", max_width=48, overflow="ellipsis")
    table.add_column("Size", justify="right")
    table.add_column("Status")
    table.add_column("Source", max_width=28, overflow="ellipsis")

    models_sorted = load_models()  # file order (models.json)

    for idx, m in enumerate(models_sorted, start=1):
        dest = _resolve_local_source(m)
        status = "downloaded" if dest.exists() else "not downloaded"
        src = m.hf_repo or ("direct" if m.url else "")
        if m.gated:
            src += " (gated)"
        table.add_row(str(idx), m.id, m.name, m.size, status, src)

    print(table)
    print("\nHelp: lom help | Support: ", SUPPORT_URL, "| Discord:", DISCORD_HANDLE)


@app.command()
def registry(
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Merge repo models into data dir registry (adds missing ids)."),
):
    """Show which registry is active and optionally refresh the data dir models.json from repo."""
    # Determine active source similar to models_registry.load_models priority
    env_path = os.environ.get("LOM_REGISTRY")
    dd = data_dir()
    data_models = dd / "models.json"
    cwd_models = Path.cwd() / "models.json"
    cwd_sample = Path.cwd() / "models.sample.json"

    active = None
    if env_path and Path(env_path).exists():
        active = Path(env_path)
    elif data_models.exists():
        active = data_models
    elif cwd_models.exists():
        active = cwd_models
    else:
        active = None  # builtin defaults

    if refresh:
        # Choose repo source to merge from: prefer models.json then models.sample.json
        src = None
        if cwd_models.exists():
            src = cwd_models
        elif cwd_sample.exists():
            src = cwd_sample
        if not src:
            print("No repo models.json or models.sample.json found in current directory.")
            return
        # Ensure data dir file exists and merge non-duplicate ids
        try:
            existing = []
            if data_models.exists():
                existing = json.loads(data_models.read_text(encoding="utf-8"))
            repo = json.loads(src.read_text(encoding="utf-8"))
            by_id = {m.get("id"): m for m in existing if isinstance(m, dict) and m.get("id")}
            added = 0
            for m in repo:
                mid = m.get("id") if isinstance(m, dict) else None
                if mid and mid not in by_id:
                    existing.append(m)
                    by_id[mid] = m
                    added += 1
            if existing and not data_models.exists():
                data_models.parent.mkdir(parents=True, exist_ok=True)
            if existing:
                data_models.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            print(f"Refreshed registry at {data_models} (+{added} new model(s))")
        except Exception as e:
            print("Failed to refresh:", e)
            return

    # Show active source and count
    models = load_models()
    print("Active registry:")
    if env_path and Path(env_path).exists():
        print("  LOM_REGISTRY:", env_path)
    elif active is None:
        print("  Built-in defaults (no models.json found)")
    else:
        print("  ", str(active))
    print("Models:", len(models))
    if len(models) < 3:
        print("Tip: Use 'registry -r' to merge repo models into your data registry, then run 'list' and 'download 3'.")


@app.command()
def download(
    n: int = typer.Argument(0, help="Model number from the list"),
    include: str = typer.Option("", "--include", "-i", help="HF allow_patterns (glob or comma-separated) to limit files"),
    id: str = typer.Option("", "--id", help="Select model by id (overrides number)"),
):
    models = load_models()
    if id:
        matches = [m for m in models if m.id == id]
        if not matches:
            raise typer.BadParameter(f"No model with id '{id}'")
        m = matches[0]
    else:
        if n < 1 or n > len(models):
            raise typer.BadParameter("Invalid model number")
        m = models[n - 1]
    # Non-interactive: do not prompt; auto-pick minimal set (see _download_model)
    _download_model(m, include_override=include.strip() or None)
    print("\nSupport: ", SUPPORT_URL, "| Questions: Discord:", DISCORD_HANDLE)


@app.command()
def delete(n: int = typer.Argument(..., help="Model number from the list to delete")):
    """Delete a downloaded model (file or folder) from local storage."""
    models = load_models()
    if n < 1 or n > len(models):
        raise typer.BadParameter("Invalid model number")
    m = models[n - 1]
    dest = _resolve_local_source(m)
    if not dest.exists():
        print("Nothing to delete: model is not downloaded.")
        return
    try:
        if dest.is_file():
            # Remove main file
            dest.unlink(missing_ok=True)
            # Remove any partial download file leftover
            part = dest.with_suffix(dest.suffix + ".part")
            if part.exists():
                part.unlink(missing_ok=True)
        else:
            # Remove directory (e.g., HF snapshot)
            shutil.rmtree(dest, ignore_errors=True)
        # Clear active model and per-model settings if they match
        cfg = Config.load()
        if cfg.get_active_model() == m.id:
            cfg.set_active_model(None)
        # Remove per-model overrides if present
        try:
            cfg.data.get("models", {}).pop(m.id, None)
        except Exception:
            pass
        cfg.save()
        print(f"Deleted: {m.name}")
    except Exception as e:
        raise typer.BadParameter(f"Failed to delete model: {e}")


@app.command()
def quick(
    n: int = typer.Argument(..., help="Model number from the list"),
    prompt: str = typer.Option("", "--prompt", "-p", help="Optional single-shot prompt before interactive menu."),
):
    """Download if missing and start chat immediately."""
    models = load_models()
    if n < 1 or n > len(models):
        raise typer.BadParameter("Invalid model number")
    m = models[n - 1]
    local = _resolve_local_source(m)
    if not local.exists():
        _download_model(m)
    # Jump straight into chat
    if prompt:
        return chat(n, prompt=prompt)
    return chat(n)


@app.command()
def chat(
    n: int = typer.Argument(..., help="Model number from the list"),
    prompt: str = typer.Option("", "--prompt", "-p", help="Single-shot quick prompt. If empty, opens interactive menu."),
):
    models = load_models()
    if n < 1 or n > len(models):
        raise typer.BadParameter("Invalid model number")
    m: Model = models[n - 1]
    # Determine local source
    if m.format.lower() == "gguf":
        path = models_dir() / f"{m.id}.{m.format}"
        if not path.exists():
            raise typer.BadParameter("Model not downloaded. Run: lom download <n>")
        local_source = path
    else:
        # safetensors/original: expect HF snapshot folder
        if m.hf_repo:
            local_source = models_dir() / m.hf_repo.replace("/", "__")
            if not local_source.exists():
                raise typer.BadParameter("Model repo not downloaded. Run: lom download <n>")
            # Validate required files to avoid runtime errors with partial downloads
            problem, missing_weights = _validate_transformers_local_dir(local_source)
            if problem:
                # Print a detailed diagnosis before raising
                try:
                    print(_missing_report(local_source, m.hf_repo))
                except Exception:
                    pass
                # Tailor suggestion based on repo files if possible
                suggestion = "lom download <n> -i \"model.safetensors,tokenizer.json,config.json\""
                if missing_weights and m.hf_repo:
                    files = _hf_list_files(m.hf_repo) or []
                    single_list = [f for f in files if f.endswith(".safetensors") and "-of-" not in f]
                    # Prefer exact 'model.safetensors' if present, else first single-file path
                    preferred_single = [f for f in single_list if f.endswith("/model.safetensors") or f == "model.safetensors"] or single_list
                    if preferred_single:
                        suggestion = f"lom download <n> -i \"{preferred_single[0]},tokenizer.json,config.json\""
                    else:
                        suggestion = "lom download <n> -i \"*.safetensors,config.json,tokenizer.json\""
                raise typer.BadParameter(problem + "\nÖrnek: " + suggestion)
        else:
            raise typer.BadParameter("Unsupported model format without repo info.")

    cfg = Config.load()
    cfg.set_active_model(m.id)
    cfg.save()

    # Quick mode: if a single prompt is provided, generate once and exit
    if prompt:
        s = cfg.get_model_settings(m.id)
        if m.format.lower() == "gguf":
            runner = LLMRunner(local_source, ctx_size=s.ctx_size, n_threads=s.threads)
        else:
            runner = TransformersRunner(str(local_source), ctx_size=s.ctx_size, n_threads=s.threads, device=s.device)
        _render_chat_header(f"{m.name}", "Single-shot", s.device, s.ctx_size)
        print(f"[bold cyan]Loading[/bold cyan] {m.name}…")
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
        print("\n[bold green]Assistant[/bold green]:\n")
        print(text)
        return

    # Interactive menu
    s = Config.load().get_model_settings(m.id)
    _render_chat_header(m.name, "Interactive Chat", s.device, s.ctx_size)
    while True:
        print("\n[bold]LLMonAny[/bold] - Model:", m.name)
        print("1) New chat\n2) Past chats\n3) Settings\n4) Exit")
        print("[dim]Note: You can change CPU/GPU device under Settings (default: CPU).[/dim]")
        print(f"[dim]Help: lom help | Support: {SUPPORT_URL} | Discord: {DISCORD_HANDLE}[/dim]")
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
            _past_chats_menu(m.id, local_source)
            continue
        if choice == 1:
            _new_chat_session(m.id, m.name, local_source)
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
        device = Prompt.ask("device (cpu|cuda|mps|auto)", default=s.device)

        ns = Settings(
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            ctx_size=int(ctx_size),
            threads=int(threads) if threads.strip() else None,
            system_prompt=system_prompt,
            device=device.strip() or "cpu",
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
    # Decide runner by presence of a gguf file
    if str(model_path).lower().endswith(".gguf"):
        runner = LLMRunner(model_path, ctx_size=s.ctx_size, n_threads=s.threads)
    else:
        runner = TransformersRunner(str(model_path), ctx_size=s.ctx_size, n_threads=s.threads, device=s.device)
    _render_chat_header(model_name, "Interactive Chat", s.device, s.ctx_size)
    print(f"Loading {model_name}...")
    while True:
        user = Prompt.ask("[bold cyan]You[/bold cyan]", default="", show_default=False)
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
        print(f"\n[bold green]Assistant:[/bold green] {reply}\n")
    # Save the chat at the end of the session
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
    if str(model_path).lower().endswith(".gguf"):
        runner = LLMRunner(model_path, ctx_size=s.ctx_size, n_threads=s.threads)
    else:
        runner = TransformersRunner(str(model_path), ctx_size=s.ctx_size, n_threads=s.threads, device=s.device)
    _render_chat_header(chat.title, "Resumed Chat", s.device, s.ctx_size)
    for msg in chat.messages:
        if msg.role != "system":
            who = "You" if msg.role == "user" else "Assistant"
            print(f"{who}: {msg.content}")
    while True:
        user = Prompt.ask("[bold cyan]You[/bold cyan]", default="", show_default=False)
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
        print(f"\n[bold green]Assistant:[/bold green] {reply}\n")
    # Save the chat after resuming a past session
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
    lom download --id ID    Download model by id
    lom download <n> -i "..."  Limit HF snapshot by include pattern(s)
    lom delete <n>          Delete downloaded model number n
    lom chat <n>            Open menu (New chat / Past chats / Settings)
    lom chat <n> -p "..."   Single-shot prompt without menu
    lom registry            Show registry source priority
    lom doctor              Environment check (Python, data dir, deps)
    lom paths               Show important directories and files

Examples:
    lom list
    lom download 3
    lom download 3 -i "model.safetensors,tokenizer.json,config.json"
    lom download --id gpt-oss-20b
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
    - For multi-file Hugging Face hubs, use --include to avoid huge downloads.

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


@app.command("registry-info")
def registry_info():
    """Show where the model registry is loaded from and how to override."""
    print("Registry sources priority:\n1) $LOM_REGISTRY\n2) $LOM_HOME/models.json\n3) ./models.json\n4) built-in")
    print("\nTip: Set LOM_HOME to choose data dir. Put models.json there to customize.")


@app.command()
def clean(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Delete without asking for confirmation"),
    scan: bool = typer.Option(False, "--scan", help="List largest files under caches and models"),
    top: int = typer.Option(20, "--top", help="How many largest files to show when scanning"),
    purge_global: bool = typer.Option(False, "--purge-global", help="Also purge the global Hugging Face cache (~/.cache/huggingface)"),
):
    """Inspect and clean up disk usage (models dir, HF cache, history). Optionally scan largest files."""
    dd = data_dir()
    global_hf = Path.home() / ".cache" / "huggingface"
    paths = {
        "models": models_dir(),
        "hf-cache": dd / "hf-cache",
        "global-hf": global_hf,
        "history": dd / "history",
        "config": dd / "config.json",
    }

    def _dir_size(p: Path) -> int:
        total = 0
        if not p.exists():
            return 0
        for root, dirs, files in os.walk(p):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except Exception:
                    pass
        return total

    def _h(n: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024 or unit == "TB":
                return f"{n:.0f}{unit}" if unit == "B" else f"{n/1024:.1f}{unit}"
            n /= 1024
        return f"{n:.1f}TB"

    table = Table(title="Disk Usage")
    table.add_column("Path")
    table.add_column("Size", justify="right")
    sizes = {}
    for name, p in paths.items():
        size = _dir_size(p) if p.is_dir() else (p.stat().st_size if p.exists() else 0)
        sizes[name] = size
        table.add_row(str(p), _h(size))
    print(table)

    if scan:
        roots = [paths["models"], paths["hf-cache"], paths["global-hf"]]
        big: list[tuple[int, Path]] = []
        for r in roots:
            if not r.exists():
                continue
            for root, dirs, files in os.walk(r):
                for f in files:
                    fp = Path(root) / f
                    try:
                        big.append((fp.stat().st_size, fp))
                    except Exception:
                        pass
        big.sort(key=lambda x: x[0], reverse=True)
        from rich.table import Table as _Table
        t2 = _Table(title=f"Top {min(top, len(big))} largest files")
        t2.add_column("Size", justify="right")
        t2.add_column("File")
        for sz, fp in big[:top]:
            t2.add_row(_h(sz), str(fp))
        print(t2)

    if not confirm:
        print("Tip: Run with --yes to purge models and local hf-cache. Add --purge-global to also delete ~/.cache/huggingface.")
        return

    # Purge models dir and hf-cache (keeps config/history)
    try:
        if paths["models"].exists():
            shutil.rmtree(paths["models"], ignore_errors=True)
        (dd / "models").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        if paths["hf-cache"].exists():
            shutil.rmtree(paths["hf-cache"], ignore_errors=True)
    except Exception:
        pass
    if purge_global:
        try:
            if paths["global-hf"].exists():
                shutil.rmtree(paths["global-hf"], ignore_errors=True)
        except Exception:
            pass
    print("Cleaned: models, hf-cache" + (", global-hf" if purge_global else ""))

if __name__ == "__main__":
    app()
