"""Frozen contact-graph convention for the released 30-feature scorer.

Edges are the historical single directed edge per CA pair (<5 Angstrom), not
silently made bidirectional. The model's weights were fitted with that convention.
CDR context uses Chothia numbering and includes the documented flanking residues.
"""
from pathlib import Path
import json
import numpy as np

CDR_CONTEXT = {'H': ((26, 35), (50, 65), (95, 102)),
               'L': ((24, 34), (50, 56), (89, 97))}


def load_features(path):
    """Load explicit JSON vectors; no pickle deserialization for public assets."""
    values = json.loads(Path(path).read_text())
    vectors = {str(k).upper(): np.asarray(v, dtype=np.float32) for k, v in values.items()}
    if not vectors or any(v.shape != (30,) or not np.isfinite(v).all() for v in vectors.values()):
        raise ValueError('Expected finite 30-dimensional amino-acid vectors')
    vectors.setdefault('UNK', np.zeros(30, dtype=np.float32))
    return vectors


def contact_graph(pdb_path, features, *, cdr_context=False):
    """Return (features, edges), preserving the campaign PDB parser convention.

No files are created beside the input PDB. cdr_context=True requires H/L chains
already Chothia-numbered; native training CDR-only PDBs use False.
"""
    residues, seen = [], set()
    with Path(pdb_path).open() as handle:
        for line in handle:
            if not line.startswith('ATOM') or len(line) < 54 or line[16] not in ' A':
                continue
            if line[12:16].strip() != 'CA' or line[17:20].strip() == 'UNK':
                continue
            try:
                chain, number = line[21], int(line[22:26])
                coord = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
            except ValueError:
                continue
            if not np.isfinite(coord).all():
                raise ValueError(f'Nonfinite CA coordinate in {pdb_path}')
            if cdr_context and not any(lo <= number <= hi for lo, hi in CDR_CONTEXT.get(chain, ())):
                continue
            key = (chain, number, line[26])
            if key in seen:
                continue
            seen.add(key)
            residues.append((line[17:26], line[17:20].strip(), coord))
    if len(residues) < 2:
        raise ValueError(f'Fewer than two eligible CA residues in {pdb_path}')
    xyz = np.stack([r[2] for r in residues])
    dist = np.sqrt(np.maximum(((xyz[:, None] - xyz[None, :]) ** 2).sum(-1), 1e-12))
    edges = [(i, j) for i in range(len(residues)) for j in range(i + 1, len(residues)) if dist[i, j] < 5.0]
    used = {i for e in edges for i in e}
    if not used:
        raise ValueError(f'No <5 Angstrom CA contacts in {pdb_path}')
    # Historical residue labels exclude insertion code; preserve for compatibility.
    labels, x = {}, []
    for i, (label, aa, _) in enumerate(residues):
        if i in used and label not in labels:
            labels[label] = len(x)
            x.append(features.get(aa, features['UNK']))
    mapped = [(labels[residues[i][0]], labels[residues[j][0]]) for i, j in edges]
    return np.asarray(x, dtype=np.float32), np.asarray(mapped, dtype=np.int64).T
