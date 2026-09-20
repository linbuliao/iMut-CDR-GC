"""Input identity and measurement statuses, separate from selection decisions."""
from __future__ import annotations

from collections import defaultdict
import errno
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from ..assets import AssetError, asset_receipt, file_sha256, verified_asset

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA3 = 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()
AA1 = 'ARNDCQEGHILKMFPSTWYV'
THREE_TO_ONE = dict(zip(AA3, AA1))


class InputError(ValueError):
    """Input is absent, ambiguous or inconsistent with its provenance."""


def is_resource_failure(error):
    """Resource exhaustion stops a batch; it is not a defective input/asset."""
    if isinstance(error, MemoryError) or (isinstance(error, OSError) and error.errno == errno.ENOMEM):
        return True
    if any(cls.__name__ == 'OutOfMemoryError' for cls in type(error).__mro__):
        return True
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        return any(text in message for text in ('out of memory', "can't allocate memory", 'cannot allocate memory', 'std::bad_alloc'))
    return False


def sequence_sha(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def identity_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def canonical_record(record):
    if not isinstance(record, Mapping):
        raise InputError('Candidate must be a mapping')
    identifier = record.get('id', record.get('mutant_id'))
    if not isinstance(identifier, (str, int)) or isinstance(identifier, bool) or not str(identifier).strip():
        raise InputError('Candidate requires a nonempty id or mutant_id')
    if 'id' in record and 'mutant_id' in record and str(record['id']) != str(record['mutant_id']):
        raise InputError('id and mutant_id disagree')
    output = dict(record, id=str(identifier))
    for role in ('heavy', 'light'):
        alias = role + '_full'
        if role in record and alias in record and record[role] != record[alias]:
            raise InputError(role + ' and ' + alias + ' disagree')
        sequence = record.get(role, record.get(alias))
        if not isinstance(sequence, str) or not sequence or set(sequence) - set(AA):
            raise InputError('Actual mutant ' + role + ' must be nonempty ungapped standard amino acids')
        output[role] = sequence
    return output


def atomic_chain_sequences(path):
    """Single-model standard ATOM CA identity; ambiguity fails, never normalized."""
    chains = defaultdict(list)
    seen = set()
    models = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith('MODEL '):
            models += 1
            if models > 1:
                raise InputError('Multiple PDB models are not accepted as one input')
        if not line.startswith(('ATOM  ', 'HETATM')) or len(line) < 54 or line[12:16].strip() != 'CA':
            continue
        if not line.startswith('ATOM  '):
            raise InputError('Non-ATOM CA residue is not part of the strict scoring contract')
        if line[16] not in ' A':
            raise InputError('Ambiguous alternate-location CA input')
        key = line[21], line[22:27]
        if key in seen:
            raise InputError('Duplicate CA residue identity')
        seen.add(key)
        try:
            int(line[22:26])
            coordinates = [float(line[k:k+8]) for k in (30, 38, 46)]
            aa = THREE_TO_ONE[line[17:20]]
        except (ValueError, KeyError) as exc:
            raise InputError('Nonstandard CA residue or invalid PDB coordinates') from exc
        if not all(math.isfinite(v) for v in coordinates):
            raise InputError('Nonfinite PDB coordinates')
        chains[key[0]].append(aa)
    if not chains:
        raise InputError('No canonical atomic chain identity found')
    return {chain: ''.join(values) for chain, values in chains.items()}


def verify_antigen(record, antigen, project_root):
    if not isinstance(antigen, Mapping) or not isinstance(antigen.get('id'), str) or not antigen['id']:
        raise InputError('Explicit antigen identity is required')
    if record.get('antigen_id') != antigen['id']:
        raise InputError('Candidate antigen_id does not match the requested antigen')
    path = verified_asset(project_root, antigen.get('pocket_pdb'), role='antigen pocket')
    chains = antigen.get('chains')
    if not isinstance(chains, Mapping) or not chains or any(not isinstance(v, str) or not v or set(v)-set(AA) for v in chains.values()):
        raise InputError('Antigen requires exact per-chain pocket sequences, not an unverified name')
    if atomic_chain_sequences(path) != dict(chains):
        raise InputError('Antigen pocket atomic sequences differ from the declared antigen identity')
    receipt = asset_receipt(project_root, path)
    if record.get('antigen_pocket_sha256', receipt['sha256']) != receipt['sha256']:
        raise InputError('Candidate is bound to a different antigen pocket')
    return path, dict(id=antigen['id'], pocket_pdb=receipt, chains=dict(chains))


def verify_model_assets(assets, project_root, model):
    if not isinstance(assets, Mapping):
        raise AssetError('Explicit weights/features asset mapping is required')
    if assets.get('model', model) != model:
        raise AssetError('Asset declaration belongs to a different model')
    weights = verified_asset(project_root, assets.get('weights'), role=model + ' weights')
    features = verified_asset(project_root, assets.get('features'), role=model + ' features')
    if features.suffix.lower() != '.json':
        raise AssetError('Scoring accepts explicit JSON features only; never unpickles feature dictionaries')
    return weights, features, dict(model=model, weights=asset_receipt(project_root, weights),
                                   features=asset_receipt(project_root, features))


def verify_unchanged(paths):
    for path, expected in paths.items():
        if file_sha256(path) != expected:
            raise AssetError('Pinned input or asset changed during scoring')


def result(record, model, protocol, status, *, score=None, logit=None, identity=None, inputs=None, error=None):
    identifier = record.get('id', record.get('mutant_id')) if isinstance(record, Mapping) else None
    if status == 'ok':
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Only finite [0,1] measurements may have status ok')
        if isinstance(logit, bool) or not isinstance(logit, (int, float)) or not math.isfinite(logit):
            raise ValueError('Model logit must be finite')
    elif score is not None or logit is not None:
        raise ValueError('Errors and not-computed measurements must not have numeric scores')
    name = 'binding' if model == 'DeepCDR-ESM2' else 'structure'
    return dict(id=identifier, model=model, protocol=protocol, status=status, score=score, logit=logit,
        **{name + '_score': score, name + '_logit': logit, name + '_status': status, name + '_score_protocol': protocol},
        identity=identity, identity_sha256=identity_sha(identity) if identity is not None else None,
        inputs=inputs, error=error, selection_decision=None,
        interpretation='Computational interaction score, not measured affinity or specificity')


def error_result(record, model, protocol, error, *, identity=None, inputs=None):
    if is_resource_failure(error):
        status = 'resource_error'
    elif isinstance(error, AssetError):
        status = 'asset_error'
    elif isinstance(error, InputError):
        status = 'invalid_input'
    elif isinstance(error, (ModuleNotFoundError, ImportError)):
        status = 'dependency_error'
    else:
        status = 'inference_error'
    return result(record, model, protocol, status, identity=identity, inputs=inputs,
                  error=dict(type=type(error).__name__, message=str(error)))
