"""Small, strict I/O helpers; never discover assets or overwrite old runs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence(value, name="sequence"):
    if not isinstance(value, str) or not value or set(value) - AA:
        raise ValueError(f"{name} must be a nonempty uppercase standard-AA sequence")
    return value


def finite(value, *, probability=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not probability or 0 <= value <= 1))


def relative_file(root, value, expected_sha256=None):
    root = Path(root).resolve(strict=True)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Paths must be relative to the explicit project root")
    path = root / relative
    if not path.resolve(strict=True).is_relative_to(root) or path.is_symlink():
        raise ValueError("Escaping paths and symlink inputs are not accepted")
    if not path.is_file():
        raise ValueError("Expected a regular input file")
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
            raise ValueError("An explicit SHA256 identity is required")
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input SHA256 differs: {relative}")
    return path


def new_output(root, relative):
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Output must be a project-relative new subdirectory")
    if any(re.search(r"(?:19|20)\d{2}[-_]?\d{2}[-_]?\d{2}", part) for part in relative.parts):
        raise ValueError("Use a descriptive output name; put dates in metadata")
    output = root / relative
    if not output.resolve().is_relative_to(root):
        raise ValueError("Output escapes the project root")
    output.mkdir(parents=True, exist_ok=False)
    return output


def read_jsonl(path):
    rows, ids = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Each JSONL row must be an object")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError("Every row must have a unique nonempty string id")
            ids.add(identifier)
            rows.append(row)
    if not rows:
        raise ValueError("Input has no records")
    return rows


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
