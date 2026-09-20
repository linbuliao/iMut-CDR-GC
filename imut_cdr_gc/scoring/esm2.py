"""Actual mutant H/L -> strict native95 -> explicit-antigen ESM2 measurement."""
from pathlib import Path

from ..assets import AssetError, file_sha256
from .common import (canonical_record, error_result, result, verify_antigen,
                     verify_model_assets, verify_unchanged)
from .native_cdr95 import PROTOCOL, canonical_esm_input

MODEL = 'DeepCDR-ESM2'
REFERENCE_WEIGHTS_SHA256 = '28250f0b24e713e4b73f13fdcfb1a440db4a5473ea3804c5dd4e29dcb77c2387'


def score_esm2(record, *, project_root, assets, antigen, numberer=None, threads=1):
    identity = inputs = None
    try:
        row = canonical_record(record)
        ag_path, antigen_identity = verify_antigen(row, antigen, project_root)
        weights, features, asset_identity = verify_model_assets(assets, project_root, MODEL)
        reference = asset_identity['weights']['sha256'] == REFERENCE_WEIGHTS_SHA256
        if not reference and assets.get('allow_nonreference_checkpoint') is not True:
            raise AssetError('Not the audited ESM2 checkpoint. A user-trained/fixture checkpoint requires explicit allow_nonreference_checkpoint=true')
        inputs = canonical_esm_input(row, project_root=project_root, numberer=numberer)
        inputs['antigen'] = antigen_identity
        paths = {ag_path: antigen_identity['pocket_pdb']['sha256'], weights: asset_identity['weights']['sha256'],
                 features: asset_identity['features']['sha256']}
        here = Path(__file__).resolve().parent
        code = {name: file_sha256(here / name) for name in ['esm2.py', 'common.py', 'native_cdr95.py', 'backend.py',
                        '_vendor/esm2_model.py', '_vendor/esm2_graph.py']}
        code['../assets.py'] = file_sha256(here.parent / 'assets.py')
        code['_vendor/amino_acid_vectors_20dim.json'] = file_sha256(here / '_vendor/amino_acid_vectors_20dim.json')
        identity = dict(model=MODEL, protocol=PROTOCOL, assets=asset_identity, code_sha256=code,
            reference_checkpoint=reference, graph_mode='train_like', graph_cutoff_angstrom=5.0,
            source_layout='Actual mutant heavy/light; ANARCI list indices; heavy-first 95; not generator cdr_seq',
            sequence_backend=inputs['backend'], old_score_cache_used=False)
        from .backend import predict_esm2
        score, logit, runtime = predict_esm2(inputs, ag_path, weights, features,
                                          asset_identity['weights']['sha256'], threads=threads)
        verify_unchanged(paths)
        identity['runtime'] = runtime
        return result(row, MODEL, PROTOCOL, 'ok', score=score, logit=logit, identity=identity, inputs=inputs)
    except Exception as exc:
        return error_result(record, MODEL, PROTOCOL, exc, identity=identity, inputs=inputs)
