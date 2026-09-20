"""Strict actual-fold identity -> correct antigen-first 30-feature 3D scoring."""
from collections.abc import Mapping
from pathlib import Path

from ..assets import AssetError, asset_receipt, file_sha256, verified_asset
from .common import (InputError, atomic_chain_sequences, canonical_record, error_result, result,
                     sequence_sha, verify_antigen, verify_model_assets, verify_unchanged)

MODEL = 'DeepCDR-3D'
PROTOCOL = 'deepcdr_3d_antigen_first_v1'
REFERENCE_WEIGHTS_SHA256 = '35dd2b41dc4765ec08fd2cb09006811c6078c389845205b7772a76030c262672'


def verify_mutant_fold(record, project_root):
    row = canonical_record(record)
    provenance = row.get('fold_provenance')
    if not isinstance(provenance, Mapping) or provenance.get('kind') != 'antibody_fold':
        raise InputError('3D scoring requires actual antibody_fold provenance; founder-sidechain reconstructions are not folds')
    if row.get('numbering') != 'chothia' or provenance.get('numbering') != 'chothia':
        raise InputError('Actual H/L fold must have explicitly recorded Chothia numbering')
    if not isinstance(provenance.get('method'), str) or not provenance['method'].strip():
        raise InputError('Fold provenance requires its method name')
    for role in ('heavy', 'light'):
        if provenance.get(role + '_sha256') != sequence_sha(row[role]):
            raise InputError('Fold provenance is for a different mutant ' + role + ' chain')
    path = verified_asset(project_root, dict(path=row.get('pdb_path'), sha256=row.get('pdb_sha256')), role='actual mutant fold')
    receipt = asset_receipt(project_root, path)
    if provenance.get('pdb_sha256') != receipt['sha256']:
        raise InputError('Fold provenance PDB identity differs from candidate PDB')
    if atomic_chain_sequences(path) != {'H': row['heavy'], 'L': row['light']}:
        raise InputError('Actual fold full H/L atomic sequences do not match the requested mutant')
    return path, dict(fold=receipt, heavy_sha256=sequence_sha(row['heavy']), light_sha256=sequence_sha(row['light']),
                      exact_full_heavy_light_sequence_match=True, fold_provenance=dict(provenance))


def score_3d(record, *, project_root, assets, antigen, threads=1):
    identity = inputs = None
    try:
        row = canonical_record(record)
        fold_path, inputs = verify_mutant_fold(row, project_root)
        ag_path, antigen_identity = verify_antigen(row, antigen, project_root)
        inputs['antigen'] = antigen_identity
        weights, features, asset_identity = verify_model_assets(assets, project_root, MODEL)
        reference = asset_identity['weights']['sha256'] == REFERENCE_WEIGHTS_SHA256
        if not reference and assets.get('allow_nonreference_checkpoint') is not True:
            raise AssetError('Not the audited 3D checkpoint. A user-trained/fixture checkpoint requires explicit allow_nonreference_checkpoint=true')
        paths = {fold_path: inputs['fold']['sha256'], ag_path: antigen_identity['pocket_pdb']['sha256'],
                 weights: asset_identity['weights']['sha256'], features: asset_identity['features']['sha256']}
        here = Path(__file__).resolve().parent
        code = {name: file_sha256(here / name) for name in ['native_structure.py', 'common.py', 'backend.py',
                              '_vendor/deepcdr3d_model.py', '_vendor/deepcdr3d_graph.py']}
        code['../assets.py'] = file_sha256(here.parent / 'assets.py')
        code['_vendor/amino_acid_vectors_30dim.json'] = file_sha256(here / '_vendor/amino_acid_vectors_30dim.json')
        identity = dict(model=MODEL, protocol=PROTOCOL, assets=asset_identity, code_sha256=code,
            reference_checkpoint=reference, graph_roles=dict(first_GCN='antigen_pocket', second_GCN='antibody_CDR'),
            feature_dimension=30, antibody_numbering='chothia', graph_cutoff_angstrom=5.0,
            edges='Historical single directed CA edge, strictly less than cutoff', old_score_cache_used=False)
        from .backend import predict_3d
        score, logit, runtime = predict_3d(fold_path, ag_path, weights, features,
                                        asset_identity['weights']['sha256'], threads=threads)
        verify_unchanged(paths)
        identity['runtime'] = runtime
        return result(row, MODEL, PROTOCOL, 'ok', score=score, logit=logit, identity=identity, inputs=inputs)
    except Exception as exc:
        return error_result(record, MODEL, PROTOCOL, exc, identity=identity, inputs=inputs)
