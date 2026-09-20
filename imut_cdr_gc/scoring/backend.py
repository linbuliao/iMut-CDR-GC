"""Lazy CPU backend. No untrusted pickle, historical score caches or downloads."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from ..assets import AssetError
from .common import AA, AA3, InputError, is_resource_failure


def feature_json(path, dimension):
    import numpy as np
    if dimension not in (20, 30):
        raise AssetError('Only the native 20- and 30-dimensional features are supported')
    with Path(path).open() as stream:
        source = json.load(stream)
    if not isinstance(source, dict):
        raise AssetError('AA features must be a JSON mapping')
    expected = set(AA3) if dimension == 30 else set(AA)
    padding = 'UNK' if dimension == 30 else 'X'
    if set(source) - expected - {padding} or not expected <= set(source):
        raise AssetError('AA feature keys must include every canonical residue and no unknown aliases')
    vectors = {key: np.asarray(value, dtype=np.float32) for key, value in source.items()}
    if any(value.shape != (dimension,) or not np.isfinite(value).all() for value in vectors.values()):
        raise AssetError('Expected finite ' + str(dimension) + '-dimensional AA features')
    if padding in vectors and np.count_nonzero(vectors[padding]):
        raise AssetError('Unknown/padding feature must be exactly zero')
    vectors.setdefault(padding, np.zeros(dimension, dtype=np.float32))
    # A caller-supplied SHA binds identity, but does not establish that an
    # arbitrary feature table is the table expected by the native weights.
    native_path = Path(__file__).parent / '_vendor' / ('amino_acid_vectors_' + str(dimension) + 'dim.json')
    native = json.loads(native_path.read_text())
    for key, value in vectors.items():
        expected_value = np.asarray(native.get(key, [0.] * dimension), dtype=np.float32)
        if not np.array_equal(value, expected_value):
            raise AssetError('AA features differ from the pinned native numeric vectors: ' + key)
    return vectors


def configure_cpu(threads):
    import torch
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise InputError('CPU thread count must be a positive integer')
    torch.set_num_threads(threads)


@lru_cache(maxsize=2)
def load_model(model_name, weights_path, weights_sha256, threads):
    """Cache model objects only by exact asset identity; never cache scores."""
    import torch
    configure_cpu(threads)
    if model_name == 'DeepCDR-3D':
        from ._vendor.deepcdr3d_model import DeepCDR3DNet
        model = DeepCDR3DNet()
    elif model_name == 'DeepCDR-ESM2':
        from ._vendor.esm2_model import GCNNet
        model = GCNNet()
    else:
        raise InputError('Unsupported scoring model')
    # Never load author full-object models. Caller must provide the pinned state
    # dictionary; unsafe pickle fallback and changed epoch substitution are absent.
    try:
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
    except Exception as exc:
        if is_resource_failure(exc):
            raise
        raise AssetError('Checkpoint could not be read with safe weights_only loading') from exc
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        if is_resource_failure(exc):
            raise
        raise AssetError('Checkpoint is not an exact compatible ' + model_name + ' state dictionary') from exc
    model.cpu().eval()
    return model


def runtime_identity():
    import platform
    import numpy
    import torch
    import torch_geometric
    return dict(python=platform.python_version(), numpy=numpy.__version__, torch=str(torch.__version__),
        torch_geometric=str(torch_geometric.__version__), device='cpu', dtype='float32',
        threads=int(torch.get_num_threads()), weights_only=True,
        historical_runtime_reproduced=False, model_cache='Exact state identity only; no score cache')


def _forward(model, call):
    import torch
    logits = []
    hook = model.out.register_forward_hook(lambda module, args, value: logits.extend(value.detach().cpu().flatten().tolist()))
    try:
        model.eval()
        with torch.inference_mode():
            scores = call().detach().cpu().flatten().tolist()
    finally:
        hook.remove()
    if len(scores) != 1 or len(logits) != 1:
        raise ValueError('One requested record did not produce exactly one score/logit')
    return float(scores[0]), float(logits[0]), runtime_identity()


def predict_esm2(prepared, antigen_path, weights_path, features_path, weights_sha256, threads=1):
    import numpy as np
    import torch
    from torch_geometric.data import Data, Batch
    from ._vendor.esm2_graph import build_graph_from_pdb, encode_cdr_sequence
    configure_cpu(threads)
    features = feature_json(features_path, 20)
    x, edges = build_graph_from_pdb(str(antigen_path), features, cutoff=5.0, graph_mode='train_like')
    if len(x) < 2 or edges.size == 0:
        raise InputError('Antigen has no eligible native <5 Å CA graph')
    target = encode_cdr_sequence(prepared['scorer_cdr_seq'], features)
    if target.shape != (95, 20) or not np.isfinite(target).all():
        raise InputError('Invalid native 95×20 input tensor')
    graph = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edges), target=torch.from_numpy(target))
    batch = Batch.from_data_list([graph])
    model = load_model('DeepCDR-ESM2', str(weights_path), weights_sha256, threads)
    return _forward(model, lambda: model(batch))


def predict_3d(fold_path, antigen_path, weights_path, features_path, weights_sha256, threads=1):
    import torch
    from torch_geometric.data import Data, Batch
    from ._vendor.deepcdr3d_graph import contact_graph
    configure_cpu(threads)
    features = feature_json(features_path, 30)
    try:
        ax, ae = contact_graph(antigen_path, features, cdr_context=False)
        bx, be = contact_graph(fold_path, features, cdr_context=True)
    except ValueError as exc:
        raise InputError('Invalid native antigen/antibody graph: ' + str(exc)) from exc
    antigen = Batch.from_data_list([Data(x=torch.from_numpy(ax), edge_index=torch.from_numpy(ae))])
    antibody = Batch.from_data_list([Data(x=torch.from_numpy(bx), edge_index=torch.from_numpy(be))])
    model = load_model('DeepCDR-3D', str(weights_path), weights_sha256, threads)
    # Critical: first branch is antigen, second branch is actual antibody CDR.
    return _forward(model, lambda: model(antigen=antigen, antibody=antibody))
