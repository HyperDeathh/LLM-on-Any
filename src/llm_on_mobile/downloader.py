from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


CHUNK = 1024 * 1024


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_sha256: Optional[str] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    headers = {}
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        mode = "ab" if "Range" in headers else "wb"
        with tmp.open(mode) as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    tmp.rename(dest)

    if expected_sha256:
        actual = _sha256(dest)
        if expected_sha256.lower() != actual.lower():
            dest.unlink(missing_ok=True)
            raise ValueError(f"SHA256 mismatch: expected {expected_sha256}, got {actual}")

    return dest
