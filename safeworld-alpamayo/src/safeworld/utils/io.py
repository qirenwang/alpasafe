from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def ensure_parent(path: str | Path) -> Path:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = project_path(path)
    if not resolved.exists():
        return []
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return resolved


def write_markdown(path: str | Path, text: str) -> Path:
    resolved = ensure_parent(path)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def append_markdown(path: str | Path, text: str) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(text)
    return resolved

