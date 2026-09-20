"""Safe numerical successor of the executed JM-Epi residue-graph builder.

Input is an explicitly prepared antigen pocket, not a whole-complex pocket
selector. This preserves native node order and duplicated sequence/contact edges.
Ambiguous insertion codes and pickle-backed atom arrays must be resolved upstream.
"""
from __future__ import annotations

from collections import defaultdict
import json

import numpy as np

from .complexes import RESIDUES
from .records import finite, new_output, relative_file, sha256, write_json

NATIVE_TRAINER_SHA256 = "95562e5d5f443c4d3ae96c47b4c80f0704b5f01c8c1a0f0e08b10e7646f4cd95"


def residue_graph(atoms, features, *, cutoff=8.0):
    """Return four numeric arrays, with the same arithmetic/order as native.

    `atoms` has pos[N,3], res_name[N], res_id[N], chain_id[N], atom_name[N].
    Unicode strings, integral IDs and finite coordinates are mandatory. Features
    are an explicitly hash-bound JSON mapping, never an implicit scoring-model
    table. Every used residue requires exactly 30 finite float32 features.
    """
    if not finite(cutoff) or cutoff <= 0:
        raise ValueError("A finite positive distance cutoff is required")
    required = {"pos", "res_name", "res_id", "chain_id", "atom_name"}
    if set(atoms) - (required | {"insertion_code"}) or not required <= set(atoms):
        raise ValueError("Expected explicit native atom columns only")
    a = {key: np.asarray(value) for key, value in atoms.items()}
    if any(value.dtype.hasobject for value in a.values()):
        raise ValueError("Object/pickle-backed atom arrays are not accepted")
    pos = a["pos"]
    if pos.ndim != 2 or pos.shape[1] != 3 or not len(pos) or pos.dtype.kind != "f" or not np.isfinite(pos).all():
        raise ValueError("Coordinates must be a nonempty finite floating-point N-by-3 array")
    n = len(pos)
    for key in required - {"pos"}:
        if a[key].shape != (n,):
            raise ValueError("Atom columns must have equal length")
        if key != "res_id" and a[key].dtype.kind != "U":
            raise ValueError("Atom labels must be Unicode strings, not bytes or objects")
    if a["res_id"].dtype.kind not in "iu":
        raise ValueError("Residue identifiers must be integral, not rounded")
    if "insertion_code" in a:
        if a["insertion_code"].shape != (n,) or a["insertion_code"].dtype.kind != "U":
            raise ValueError("Invalid insertion-code column")
        if any(str(value).strip() for value in a["insertion_code"]):
            raise ValueError("Native graph ignores insertion codes; explicitly renumber and retain a mapping first")
    groups, residue_names, seen_atoms = {}, {}, set()
    for i in range(n):
        chain, rid, name, atom = str(a["chain_id"][i]), int(a["res_id"][i]), str(a["res_name"][i]), str(a["atom_name"][i])
        if not chain.strip() or name not in RESIDUES or not atom.strip():
            raise ValueError("Use explicit chains and standard protein residues/atom names")
        if residue_names.setdefault((chain, rid), name) != name:
            raise ValueError("Native residue key conflates distinct residue identities")
        atom_key = (chain, rid, atom.strip().upper())
        if atom_key in seen_atoms:
            raise ValueError("Duplicate atom or unresolved alternate conformer")
        seen_atoms.add(atom_key)
        groups.setdefault((chain, rid, name), []).append(i)
    nodes, coords, vectors = [], [], []
    for (chain, rid, name), indices in groups.items():
        ca = next((i for i in indices if str(a["atom_name"][i]).strip().upper() == "CA"), None)
        point = pos[ca] if ca is not None else pos[indices].mean(axis=0)
        coords.append(point.astype(np.float32))
        nodes.append((chain, rid, name))
        key = name if name in features else RESIDUES[name]
        if key not in features:
            raise ValueError("Missing explicit residue feature; no unknown/zero-vector fallback")
        vector = np.asarray(features[key], dtype=np.float32)
        if vector.shape != (30,) or not np.isfinite(vector).all():
            raise ValueError("Every used residue must have exactly 30 finite features")
        vectors.append(vector)
    coordinates = np.stack(coords).astype(np.float32)
    x = np.stack(vectors).astype(np.float32)
    chains = defaultdict(dict)
    for i, (chain, rid, _) in enumerate(nodes):
        chains[chain][rid] = i
    src, dst, is_seq = [], [], []
    for residue_to_node in chains.values():
        order = sorted(residue_to_node)
        for a_id, b_id in zip(order[:-1], order[1:]):
            i, j = residue_to_node[a_id], residue_to_node[b_id]
            src.extend([i, j]); dst.extend([j, i]); is_seq.extend([1, 1])
    if len(nodes) >= 2:
        distances = np.sqrt(np.maximum(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(-1), 1e-12))
        np.fill_diagonal(distances, np.inf)
        contact_src, contact_dst = np.where(distances <= float(cutoff))
        for i, j in zip(contact_src.tolist(), contact_dst.tolist()):
            src.append(i); dst.append(j); is_seq.append(0)
    if src:
        src_array, dst_array = np.asarray(src, np.int64), np.asarray(dst, np.int64)
        distance = np.linalg.norm(coordinates[src_array] - coordinates[dst_array], axis=1).astype(np.float32)
        edge_index = np.stack([src_array, dst_array])
        edge_attr = np.stack([distance, np.asarray(is_seq, np.float32)], axis=1)
    else:
        edge_index, edge_attr = np.zeros((2, 0), np.int64), np.zeros((0, 2), np.float32)
    return dict(x=x, pos=coordinates, edge_index=edge_index, edge_attr=edge_attr)


def prepare_antigen_graph(*, project_root, atom_npz, atom_sha256, feature_json,
                          feature_sha256, antigen_id, output_dir, cutoff=8.0):
    if not isinstance(antigen_id, str) or not antigen_id:
        raise ValueError("Explicit antigen identity is required")
    source = relative_file(project_root, atom_npz, atom_sha256)
    feature_file = relative_file(project_root, feature_json, feature_sha256)
    features = json.loads(feature_file.read_text())
    if not isinstance(features, dict):
        raise ValueError("Feature JSON must map residue names to numeric vectors")
    with np.load(source, allow_pickle=False) as data:
        atoms = {key: data[key] for key in data.files}
    graph = residue_graph(atoms, features, cutoff=cutoff)
    if sha256(source) != atom_sha256 or sha256(feature_file) != feature_sha256:
        raise ValueError("Antigen graph inputs changed during preparation")
    output = new_output(project_root, output_dir)
    np.savez(output / "antigen_graph.npz", **graph)
    receipt = dict(status="prepared_numeric_graph_not_model_inference", antigen_id=antigen_id,
        input_sha256=atom_sha256, feature_sha256=feature_sha256, cutoff_angstrom=float(cutoff),
        graph_sha256=sha256(output / "antigen_graph.npz"), nodes=len(graph["x"]),
        directed_edges=graph["edge_index"].shape[1], node_order="first atom occurrence",
        native_trainer_sha256=NATIVE_TRAINER_SHA256,
        sequence_edges_connect_sorted_adjacent_observed_residue_ids=True,
        duplicate_sequence_contact_edges_preserved=True,
        pocket_selection_performed=False, protein_unknown_or_insertion_collision_fallback=False,
        scientific_acceptance=False)
    write_json(output / "preparation.json", receipt)
    return receipt
