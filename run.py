from __future__ import annotations

import json
import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

# Ensure src/ is importable when running from repo without installing
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def _ensure_registry() -> None:
    from llm_on_mobile.config import data_dir

    dd = data_dir()
    (dd / "models").mkdir(parents=True, exist_ok=True)
    target = dd / "models.json"
    if target.exists():
        # Merge new repo models non-destructively
        repo_models_path = None
        for candidate in (ROOT / "models.json", ROOT / "models.sample.json"):
            if candidate.exists():
                repo_models_path = candidate
                break
        if repo_models_path:
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
                repo = json.loads(repo_models_path.read_text(encoding="utf-8"))
                by_id = {m.get("id"): m for m in existing if isinstance(m, dict) and m.get("id")}
                added = 0
                for m in repo:
                    mid = m.get("id")
                    if mid and mid not in by_id:
                        existing.append(m)
                        by_id[mid] = m
                        added += 1
                if added:
                    target.write_text(json.dumps(existing, indent=2), encoding="utf-8")
                    print(f"[setup] Merged {added} new model(s) into {target.name} from repo")
            except Exception:
                pass
        return

    # Prefer repo models.json or models.sample.json if available
    for candidate in (ROOT / "models.json", ROOT / "models.sample.json"):
        if candidate.exists():
            target.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[setup] Created {target} from {candidate.name}")
            return

    # Fallback: minimal working defaults
    default_models = [
        {
            "id": "llama-3.1-8b-q4",
            "name": "Llama 3.1 8B Instruct Q4_0",
            "size": "~4.5 GB",
            "family": "llama",
            "format": "gguf",
            "url": "https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct.Q4_0.gguf?download=true",
            "sha256": None,
        },
        {
            "id": "tinyllama-1.1b-chat",
            "name": "TinyLlama 1.1B Chat v1.0",
            "size": "~2.0 GB",
            "family": "tinyllama",
            "format": "safetensors",
            "url": "",
            "hf_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "hf_filename": None,
            "gated": False,
        },
    ]
    target.write_text(json.dumps(default_models, indent=2), encoding="utf-8")
    print(f"[setup] Created default registry at {target}")


def _is_termux() -> bool:
    pref = os.environ.get("PREFIX", "")
    return "com.termux" in sys.prefix or pref.startswith("/data/data/com.termux")


def _run(cmd: list[str]) -> int:
    print(f"[exec] {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    except Exception as e:
        print("[exec] failed:", e)
        return 1


def _pip_install(args: list[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"] + args
    return _run(cmd) == 0


def _ensure_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def _bootstrap_env() -> None:
    """Best-effort auto-install of project and optional backends.
    - Installs this project (editable when running from repo).
    - Installs core deps (handled by pyproject on project install).
    - Tries llama-cpp-python (GGUF) and HF stack (transformers/accelerate/torch CPU).
    - On Termux, still attempts installs but skips fatal errors.
    """
    print("[bootstrap] Starting environment setup…")

    # Termux system packages (best-effort)
    if _is_termux():
        pkg_path = shutil.which("pkg")
        if pkg_path:
            print("[bootstrap] Termux detected. Installing system packages…")
            _run([pkg_path, "update", "-y"])  # ignore failures
            _run([pkg_path, "upgrade", "-y"])  # ignore failures
            _run([pkg_path, "install", "-y", "clang", "cmake", "git"])  # toolchain for optional builds
        else:
            print("[bootstrap] Termux detected but 'pkg' not found; skipping system packages.")

    # Upgrade pip tooling (lightweight, safe)
    _pip_install(["--upgrade", "pip", "setuptools", "wheel"])  # ignore result

    # Install this project if running from repo; otherwise skip
    if (ROOT / "pyproject.toml").exists():
        print("[bootstrap] Installing project (editable)…")
        _pip_install(["-e", "."])  # core deps

    # Optional backends
    # 1) llama-cpp-python (for GGUF). May build from source; can fail on Termux w/o toolchain.
    if not _ensure_import("llama_cpp"):
        print("[bootstrap] Installing llama-cpp-python (optional)…")
        _pip_install(["llama-cpp-python"])  # tolerate failure
    else:
        print("[bootstrap] llama-cpp-python already present.")

    # 2) HF stack for safetensors: transformers + accelerate
    hf_missing = not _ensure_import("transformers")
    if hf_missing:
        print("[bootstrap] Installing Transformers + Accelerate (optional)…")
        _pip_install(["transformers>=4.41.0", "accelerate>=0.30.0"])  # tolerate failure
    else:
        print("[bootstrap] Transformers already present.")

    # 3) Torch CPU for desktop platforms when Transformers is used
    # Try only if transformers is importable but torch is missing
    if _ensure_import("transformers") and not _ensure_import("torch"):
        print("[bootstrap] Installing PyTorch (CPU)…")
        # Prefer official CPU index for mainstream OSes
        idx = "--index-url=https://download.pytorch.org/whl/cpu"
        # Some environments (like Termux/Android) won't have wheels; tolerate failure
        _pip_install([idx, "torch"]) or _pip_install(["torch"])  # second attempt from PyPI

    print("[bootstrap] Done.")


def _deps_check() -> None:
    print("[doctor] Python:", sys.version.split(" ")[0], platform.platform())
    core_ok = True
    try:
        import requests  # noqa: F401
        import rich  # noqa: F401
        import typer  # noqa: F401
        print("[doctor] Core deps: OK")
    except Exception as e:
        core_ok = False
        print("[doctor] Core deps missing. Try: pip install -e .")
        print("          Error:", e)

    if _ensure_import("llama_cpp"):
        print("[doctor] llama-cpp-python: OK (GGUF)")
    else:
        print("[doctor] llama-cpp-python: not installed (optional for GGUF)")

    if _ensure_import("transformers"):
        print("[doctor] transformers: OK (safetensors)")
    else:
        print("[doctor] transformers: not installed (optional for safetensors)")

    from llm_on_mobile.cli import data_dir, models_dir
    print("[doctor] Data dir:", data_dir())
    print("[doctor] Models dir:", models_dir())

    if not core_ok:
        print("\nHint:\n  pip install -e . ; pip install -e .[llama] ; pip install -e .[hf]")
        print("  CPU Torch: pip install torch --index-url https://download.pytorch.org/whl/cpu")


def main() -> None:
    # Ensure registry & print diagnostics
    _ensure_registry()
    _bootstrap_env()
    _deps_check()

    # Show model list via CLI function
    from llm_on_mobile import cli as lom_cli

    print("\n[info] Available Models\n")
    # call the Typer command function directly
    lom_cli.list_models()


if __name__ == "__main__":
    main()
