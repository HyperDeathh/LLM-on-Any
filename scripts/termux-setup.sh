#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

# Termux bootstrap for LLMonAny (LLM on Any)
pkg update -y && pkg upgrade -y
pkg install -y python clang cmake git

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
  pip install llm-on-mobile
fi

# Optional: local inference via llama-cpp-python (may compile, can be slow)
pip install llama-cpp-python || echo "Skipping llama-cpp-python (build failed). You can use remote adapters later."

echo "\nDone. Try: lom list (LLMonAny)"
