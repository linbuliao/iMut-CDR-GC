"""Explicit, project-contained, hash-pinned local assets; never downloads."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Mapping


class AssetError(ValueError):
    """Missing, changed, unpinned or outside-project input asset."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def project_path(project_root, value, *, must_exist=True):
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise AssetError('Explicit project_root must be an existing directory')
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise AssetError('Asset path must be nonempty')
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if path == root or root not in path.parents:
        raise AssetError('Asset path escapes the explicit project root')
    if must_exist and not path.is_file():
        raise AssetError('Required local asset is missing: ' + str(path.relative_to(root)))
    return path


@dataclass(frozen=True)
class AssetSpec:
    path: str
    sha256: str


def verified_asset(project_root, spec, *, role='asset'):
    """Require exact content identity before any parser or model sees the file."""
    if isinstance(spec, AssetSpec):
        spec = dict(path=spec.path, sha256=spec.sha256)
    if not isinstance(spec, Mapping):
        raise AssetError(role + ' requires {path, sha256}, not an implicit default')
    expected = spec.get('sha256')
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise AssetError(role + ' requires a lowercase 64-character SHA256')
    path = project_path(project_root, spec.get('path'))
    observed = file_sha256(path)
    if observed != expected:
        raise AssetError(role + ' content hash differs from its declared identity')
    return path


def asset_receipt(project_root, path):
    path = project_path(project_root, path)
    return dict(path=str(path.relative_to(Path(project_root).resolve())), sha256=file_sha256(path))
