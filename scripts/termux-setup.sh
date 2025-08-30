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
