from __future__ import annotations

import sys
from pathlib import Path
import shlex

# Make src/ importable when running from repo
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def _ensure_min_registry() -> None:
    """Create minimal registry/config dirs without heavy bootstrap (fast start)."""
    try:
        from llm_on_mobile.config import data_dir
        dd = data_dir()
        (dd / "models").mkdir(parents=True, exist_ok=True)
        target = dd / "models.json"
        if not target.exists():
            for cand in (ROOT / "models.json", ROOT / "models.sample.json"):
                if cand.exists():
                    target.write_text(cand.read_text(encoding="utf-8"), encoding="utf-8")
                    break
    except Exception:
        # Best-effort; keep startup fast
        pass


def main() -> None:
    _ensure_min_registry()

    # Import CLI after sys.path tweak
    from llm_on_mobile import cli as lom_cli

    print("\nKomutları doğrudan yazın (ör: 'list', 'download 3', 'chat 3'). Çıkış: Ctrl-D/Ctrl-C.\n")
    # Optionally show models once for convenience
    try:
        lom_cli.list_models()
    except Exception:
        pass

    while True:
        try:
            line = input("lom> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break
        if not line:
            continue
        try:
            args = shlex.split(line)
            # Invoke Typer app directly with provided args (no 'lom' prefix needed)
            lom_cli.app(args=args, prog_name="lom", standalone_mode=False)
        except SystemExit:
            # Typer/Click may raise SystemExit on command completion
            pass
        except Exception as e:
            print("[shell] error:", e)


if __name__ == "__main__":
    main()
