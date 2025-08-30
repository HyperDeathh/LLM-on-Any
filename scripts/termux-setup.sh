#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

# Termux bootstrap for LLM-on-Any (English-only, working defaults)
echo "[setup] Updating Termux packages…"
pkg update -y && pkg upgrade -y
pkg install -y python clang make cmake git

echo "[setup] Creating Python virtual environment (.venv)…"
if [ ! -d .venv ]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] Upgrading pip tooling…"
pip install --upgrade pip setuptools wheel

echo "[setup] Installing project (editable if repo present)…"
if [ -f "pyproject.toml" ]; then
  pip install -e .
else
  pip install llm-on-any
fi

echo "[setup] Installing core runtime deps (HF pin for Termux)…"
pip install "huggingface_hub<0.33" tqdm rich typer requests

echo "[setup] Installing optional backends (best-effort)…"
pip install "transformers>=4.41.0" "accelerate>=0.30.0" || true
pip install llama-cpp-python || echo "[setup] Skipping llama-cpp-python (build failed)."

cat <<'EOF'

[done] LLM-on-Any is set up.

How to use:
  1) Activate venv (each new session):
    source .venv/bin/activate
  2) List models:
    lom list
  3) Download a model:
    lom download 3
  4) Chat:
    lom chat 3

Notes:
  - If 'lom' is not found in PATH, run the REPL instead:
    python fastStart.py
  - For gated Hugging Face models, accept the license on the model page and login:
    python -m huggingface_hub login
  - On Termux, PyTorch wheels are not available; Transformers may be limited without torch.
  - Hugging Face cache is stored under ~/.llmonany/hf-cache by default.
EOF
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

# Termux bootstrap for LLMonAny (LLM on Any)
pkg update -y && pkg upgrade -y
pkg install -y python clang make cmake git

# Python venv
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip setuptools wheel

# Install from local directory if present, else from PyPI
if [ -f "pyproject.toml" ]; then
  pip install -e .
else
  # Project package name in pyproject is 'llm-on-any'
  pip install llm-on-any
fi

# Optional: local inference via llama-cpp-python (may compile; often fails on Termux without extra toolchain)
# If this fails, you can skip and use remote/back-end adapters instead.
pip install llama-cpp-python || echo "Skipping llama-cpp-python (build failed). You can use remote adapters later."

echo "\nDone. Try: lom list (LLMonAny)"
