"""Load an already materialized native antigen graph without pickle or remapping."""
from __future__ import annotations
from pathlib import Path
import re
from .loading import sha256


def load_antigen_graph(path, expected_sha256, *, node_dim=30):
    """Read safe NPZ x/pos/edge_index/edge_attr, preserving every node and edge.

    This is NOT an atom-level NPZ converter. Historical atom-level NPZ files
    require separately verified native graph materialization and feature
    identities; object-array files are refused rather than loading pickle.
    """
    if not re.fullmatch('[0-9a-f]{64}', str(expected_sha256)):
        raise ValueError('An explicit antigen graph SHA-256 is required')
    source = Path(path).resolve()
    if source.suffix.lower() != '.npz' or not source.is_file():
        raise FileNotFoundError('Provide an existing safe native-graph NPZ')
    if sha256(source) != expected_sha256:
        raise ValueError('Antigen graph hash mismatch')
    import numpy as np
    import torch
    fields = {'x', 'pos', 'edge_index', 'edge_attr'}
    with np.load(source, allow_pickle=False) as loaded:
        if set(loaded.files) != fields:
            raise ValueError('Expected materialized native graph x/pos/edge_index/edge_attr, not atom-level NPZ')
        arrays = {name:loaded[name].copy() for name in sorted(fields)}
    if arrays['edge_index'].dtype != np.int64:
        raise ValueError('Native edge_index must be int64; no implicit index conversion')
    for name in ('x', 'pos', 'edge_attr'):
        if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError('Native graph arrays must be finite float32: ' + name)
    graph = {name:torch.from_numpy(values) for name, values in arrays.items()}
    from ..generation import _validate_graph
    _validate_graph(graph, node_dim)
    if graph['pos'].shape != (graph['x'].shape[0], 3):
        raise ValueError('Native graph coordinates must be N by 3')
    flags = graph['edge_attr'][:, 1]
    if not torch.all((flags == 0) | (flags == 1)):
        raise ValueError('Native sequence-edge flags must be zero or one')
    if sha256(source) != expected_sha256:
        raise ValueError('Antigen graph changed during loading')
    graph['_source_identity'] = dict(sha256=expected_sha256,
        representation='materialized_native_jm_epi_graph_v1',
        nodes=int(graph['x'].shape[0]), edges=int(graph['edge_index'].shape[1]),
        no_node_or_edge_recomputation=True, pickle_loaded=False)
    return graph
