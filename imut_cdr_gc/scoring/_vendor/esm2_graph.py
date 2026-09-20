"""Pinned PDB/sequence input math from deepcdr_esm2.interface, no scoring."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

MAX_SEQ_LEN = 95

GRAPH_CUTOFF = 5.0

AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X",
}

def encode_cdr_sequence(seq: str, aa_dict: Dict[str, np.ndarray], max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    out = np.zeros((max_len, 20), dtype=np.float32)
    for idx, aa in enumerate(seq[:max_len]):
        out[idx] = aa_dict.get(aa, aa_dict["X"])
    return out

def _graph_labels_from_pdb(pdb_path: str) -> Tuple[List[str], np.ndarray]:
    labels: List[str] = []
    coords: List[np.ndarray] = []
    for raw in open(pdb_path, "r", encoding="utf-8"):
        tem_b = raw[16] if len(raw) > 16 else " "
        line = raw[:16] + " " + raw[17:] if len(raw) > 16 else raw
        arr = line.split()
        if not arr:
            continue
        if arr[0] == "ATOM" and tem_b != "B" and " HOH " not in line:
            if arr[2] == "CA" and arr[3] != "UNK":
                labels.append(line[17:26])
                coords.append(np.asarray([line[30:38], line[38:46], line[46:54]], dtype=np.float32))
    return labels, np.asarray(coords, dtype=np.float32)

def build_graph_from_pdb(
    pdb_path: str,
    aa_dict: Dict[str, np.ndarray],
    cutoff: float = GRAPH_CUTOFF,
    graph_mode: str = "train_like",
) -> Tuple[np.ndarray, np.ndarray]:
    labels, coords_arr = _graph_labels_from_pdb(pdb_path)
    if len(labels) < 2:
        return np.zeros((0, 20), dtype=np.float32), np.zeros((2, 0), dtype=np.int64)

    dmat = np.sqrt(np.maximum(((coords_arr[:, None, :] - coords_arr[None, :, :]) ** 2).sum(-1), 1e-12))
    np.fill_diagonal(dmat, np.inf)

    edge_pairs: List[Tuple[int, int]] = []
    used_mask = np.zeros((len(labels),), dtype=bool)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if float(dmat[i, j]) < float(cutoff):
                edge_pairs.append((i, j))
                used_mask[i] = True
                used_mask[j] = True

    used_node_idx = np.where(used_mask)[0].tolist()
    if not used_node_idx:
        return np.zeros((0, 20), dtype=np.float32), np.zeros((2, 0), dtype=np.int64)

    ordered_labels = list(set([labels[idx] for idx in used_node_idx]))
    label_to_idx = {label: idx for idx, label in enumerate(ordered_labels)}

    feats_arr = []
    for label in ordered_labels:
        aa3 = label[:3].strip().upper()
        aa1 = AA_3TO1.get(aa3, "X")
        feats_arr.append(aa_dict.get(aa1, aa_dict["X"]))
    feats_arr = np.asarray(feats_arr, dtype=np.float32)

    src: List[int] = []
    dst: List[int] = []
    for i, j in edge_pairs:
        src.append(label_to_idx[labels[i]])
        dst.append(label_to_idx[labels[j]])
        if graph_mode == "bidirectional":
            src.append(label_to_idx[labels[j]])
            dst.append(label_to_idx[labels[i]])

    edge_index = np.stack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)], axis=0) if src else np.zeros((2, 0), dtype=np.int64)
    return feats_arr, edge_index
