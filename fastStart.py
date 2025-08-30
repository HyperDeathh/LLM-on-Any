from __future__ import annotations

import sys
from pathlib import Path
import shlex
from typing import List

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

    def _dispatch(tokens: List[str]) -> None:
        if not tokens:
            return
        cmd = tokens[0].lower()
        rest = tokens[1:]
        try:
            if cmd in {"list", "ls"}:
                lom_cli.list_models()
                return
            if cmd == "help":
                lom_cli.help()
                return
            if cmd == "registry":
                refresh = False
                for t in rest:
                    if t in {"-r", "--refresh"}:
                        refresh = True
                lom_cli.registry(refresh=refresh)
                return
            if cmd == "download":
                n = 0
                include = ""
                mid = ""
                i = 0
                while i < len(rest):
                    t = rest[i]
                    if t in {"-i", "--include"} and i + 1 < len(rest):
                        include = rest[i + 1]
                        i += 2
                        continue
                    if t == "--id" and i + 1 < len(rest):
                        mid = rest[i + 1]
                        i += 2
                        continue
                    if t.lstrip("-+").isdigit() and n == 0:
                        try:
                            n = int(t)
                        except Exception:
                            n = 0
                    i += 1
                lom_cli.download(n=n, include=include, id=mid)
                return
            if cmd == "delete":
                if rest and rest[0].lstrip("-+").isdigit():
                    n = int(rest[0])
                    lom_cli.delete(n)  # type: ignore[arg-type]
                    return
            if cmd == "chat":
                n = None
                prompt = ""
                i = 0
                while i < len(rest):
                    t = rest[i]
                    if t in {"-p", "--prompt"} and i + 1 < len(rest):
                        prompt = rest[i + 1]
                        i += 2
                        continue
                    if n is None and t.lstrip("-+").isdigit():
                        n = int(t)
                    i += 1
                if n is None:
                    print("Usage: chat <n> [-p PROMPT]")
                    return
                lom_cli.chat(n=n, prompt=prompt)
                return
            if cmd == "quick":
                n = None
                prompt = ""
                i = 0
                while i < len(rest):
                    t = rest[i]
                    if t in {"-p", "--prompt"} and i + 1 < len(rest):
                        prompt = rest[i + 1]
                        i += 2
                        continue
                    if n is None and t.lstrip("-+").isdigit():
                        n = int(t)
                    i += 1
                if n is None:
                    print("Usage: quick <n> [-p PROMPT]")
                    return
                lom_cli.quick(n=n, prompt=prompt)
                return
        except Exception as e:
            print("[dispatch] error:", e)

        # Fallback to Typer parsing if manual dispatch didn't handle it
        try:
            lom_cli.app(args=tokens, prog_name="lom", standalone_mode=False)
        except SystemExit:
            pass

    while True:
        try:
            line = input("lom> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break
        if not line:
            continue
        args = shlex.split(line)
        _dispatch(args)


if __name__ == "__main__":
    main()
