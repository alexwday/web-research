"""File helpers for output artifacts."""
import csv
import json
import re
from pathlib import Path
from typing import Iterable


def ensure_directory(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_filename(name: str, max_length: int = 120) -> str:
    cleaned = name.replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", cleaned)
    cleaned = cleaned[:max_length]
    return cleaned or "artifact"


def write_json(path: str | Path, payload: dict | list) -> Path:
    out = Path(path)
    ensure_directory(out.parent)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return out


def write_markdown(path: str | Path, content: str) -> Path:
    out = Path(path)
    ensure_directory(out.parent)
    with open(out, "w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")
    return out


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[dict]) -> Path:
    out = Path(path)
    ensure_directory(out.parent)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out
