"""Explicit-asset native AB2 antibody folds, separate from final complexes.

No upstream package initializer, refinement, download or implicit GPU choice.
Tests exercise CPU contracts, not real learned-weight predictor parity.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping
import warnings

from .assets import AssetError, asset_receipt, file_sha256, verified_asset
from .records import new_output, write_json, write_jsonl
from .scoring.common import (InputError, THREE_TO_ONE, atomic_chain_sequences,
    canonical_record, identity_sha, is_resource_failure, sequence_sha, verify_unchanged)

PROFILE = 'native_ab2_123_chothia_unrefined_v1'
SOURCE_SHA256 = {
    'constants.py': 'f93048f70edb72550cda87412bfb0379baca705f63c67d3c6bd9f2196d2d4429',
    'rigids.py': '8b149407fe20c03ad249901a56d0e78c40d1645eb1fcb3b4628a83476f77b7a2',
    'models.py': '69367b5ea0b4fa822daf6308c4480cd3b40ff2c9b3cbe69511a2d520748126d1',
    'util.py': '803b3488269165a47fdb26396121b943a458a7d2ea1adbd9fd62b8729b04d525',
    'ABodyBuilder2.py': '654dbd44ec5a445b0f8120f2905d32ed5f8434c626154f9afe581420166a3881',
}
WEIGHT_SHA256 = {
    'antibody_model_1': 'ecefff604457a105fe405dcca44a76b046609ff2ac3c355329974659f6881e09',
    'antibody_model_2': '91dd9f5ebca8cc68f292d3a40ccdbc8a86974e7cebc4853e19565ecb8a8d508b',
    'antibody_model_3': '33085a728cfb1059d9e817b785a21f1dda74eff8f083a0c2a033ae88bdbdcd45',
}
NUMBERING_SOURCE_SHA256 = {
    '__init__.py': '2b3033bfadabd97d4843bee2a8befc624373893490884741d707ce3c35f4ec31',
    'anarci.py': 'f42311674d73ae45a4553aa13dbe618a3efad527ef9d8bd6135d57dff1eb4b45',
    'germlines.py': '95a13463e2620b8037bf7ea67ecdf9ec6f8a8f973c5658f3c54126e9e36f3452',
    'schemes.py': '856f4c942c226cbeabfabb69f8bfd4004f00f9b0aa6ce1e296636d2ce88f4b4b',
}
HMM_SHA256 = {
    'ALL.hmm': '60051f5a838704eb97fe39488f23894aa335ddadd24ad72b241f6e1cea6829ee',
    'ALL.hmm.h3f': 'ae7b65199339c633096b2106448a29b35cafd3cbfb9868550d103049a43ad7a8',
    'ALL.hmm.h3i': 'bfd9b15a083a5e7af186e7f5be28b6fab6b8ccf123b62043d26086d9d4ae1be8',
    'ALL.hmm.h3m': '51635b8c6bae5f7a43effbb943ef29c1fefb7641c766ce95cf281ad18bc65ab0',
    'ALL.hmm.h3p': '4f66bde5bdd1766ae0d880ca5aa1b50b5c09c58a3b6c34a6a9ca8f0ddea63c07',
}
HMMSCAN_SHA256 = 'a00e919b362e074603e4bfbf3adc27ab20095636f7966597e07c62fe8cac1be9'
FOLD_FIELDS = frozenset(('pdb_path', 'pdb_sha256', 'fold_provenance', 'fold_status',
                          'numbering', 'fold_receipt', 'antibody_pdb'))
PARAMETERS = dict(model_ids=[1, 2, 3], embed_dims=[128, 256, 256], rel_pos_dim=64,
    numbering='chothia', allowed_species=['human', 'mouse'], refined=False,
    precision='float32', ensemble_ranking='native mean ensemble CA squared deviation; ascending',
    save_method='save_single_unrefined(index=ranking[0])', chain_order=['H', 'L'],
    antigen_coordinates_used=False, learned_weight_predictor_parity_verified=False)


def _exact_keys(value, keys, name):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise AssetError(name + ' requires exactly: ' + ', '.join(keys))


def _verify_group(root, specs, expected, role):
    _exact_keys(specs, expected, role)
    paths, receipts = {}, {}
    for name, digest in expected.items():
        spec = specs[name]
        if not isinstance(spec, Mapping) or spec.get('sha256') != digest:
            raise AssetError(role + ' ' + name + ' is not the frozen native identity')
        value = spec.get('path')
        if not isinstance(value, str) or Path(value).is_absolute() or '..' in Path(value).parts:
            raise AssetError('Folding assets must be explicit project-relative paths')
        paths[name] = verified_asset(root, spec, role=role + ' ' + name)
        receipts[name] = asset_receipt(root, paths[name])
    return paths, receipts


def _verify_assets(root, assets):
    _exact_keys(assets, ('profile', 'sources', 'weights', 'numbering'), 'fold assets')
    if assets['profile'] != PROFILE:
        raise AssetError('Unsupported native fold profile; no implicit model conversion')
    sources, sr = _verify_group(root, assets['sources'], SOURCE_SHA256, 'AB2 source')
    weights, wr = _verify_group(root, assets['weights'], WEIGHT_SHA256, 'AB2 weight')
    numbering = assets['numbering']
    _exact_keys(numbering, ('sources', 'hmms', 'hmmscan'), 'numbering assets')
    ns, nr = _verify_group(root, numbering['sources'], NUMBERING_SOURCE_SHA256, 'ANARCI source')
    hmms, hr = _verify_group(root, numbering['hmms'], HMM_SHA256, 'ANARCI HMM')
    scan, er = _verify_group(root, {'hmmscan': numbering['hmmscan']},
                             {'hmmscan': HMMSCAN_SHA256}, 'HMMER executable')
    package = ns['__init__.py'].parent
    if any(path != package / name for name, path in ns.items()):
        raise AssetError('ANARCI sources must describe one installed native package')
    if any(path != package / 'dat/HMMs' / name for name, path in hmms.items()):
        raise AssetError('ANARCI HMM files are not colocated with the declared native package')
    if not os.access(scan['hmmscan'], os.X_OK):
        raise AssetError('Declared hmmscan is not executable')
    pins = {}
    for group, rec in ((sources, sr), (weights, wr), (ns, nr), (hmms, hr), (scan, er)):
        pins.update({path: rec[name]['sha256'] for name, path in group.items()})
    return dict(sources=sources, weights=weights, numbering_sources=ns, hmms=hmms,
        hmmscan=scan['hmmscan'], pins=pins, receipt=dict(profile=PROFILE,
        sources=sr, weights=wr, numbering=dict(sources=nr, hmms=hr, hmmscan=er['hmmscan'])))


def _number_chain(sequence, role, anarci):
    """Same recognition/scheme calls as native, with full-domain coverage guards."""
    allow = {'H'} if role == 'H' else {'L', 'K'}
    final = None
    for scheme in ('imgt', 'chothia'):
        numbered, details, _ = anarci([('sequence', sequence)], scheme=scheme,
            output=False, allow=allow, allowed_species=['human', 'mouse'])
        if not numbered or len(numbered) != 1 or not numbered[0] or len(numbered[0]) != 1:
            raise InputError('Exactly one complete ' + role + ' variable domain is required')
        if not details or len(details) != 1 or not details[0] or len(details[0]) != 1:
            raise InputError('ANARCI chain-role evidence is absent or ambiguous')
        if details[0][0].get('chain_type') not in allow:
            raise InputError('ANARCI chain role differs from requested H/L')
        output = [x for x in numbered[0][0][0] if x[1] != '-']
        if ''.join(x[1] for x in output) != sequence:
            raise InputError('Numbering would truncate or change full H/L; provide the complete variable domain only')
        keys = []
        for pair, aa in output:
            number, insertion = pair
            if type(number) is not int or not 1 <= number <= 9999 or not isinstance(insertion, str) or len(insertion) != 1 or (insertion != ' ' and not insertion.isascii()):
                raise InputError('Invalid residue number or insertion code')
            keys.append((number, insertion))
        if len(keys) != len(set(keys)):
            raise InputError('Duplicate numbered residue identity')
        final = output
    return final


def _number_worker(spec):
    root = Path(spec['project_root']).resolve(strict=True)
    bundle = _verify_assets(root, spec['assets'])
    expected = bundle['numbering_sources']['__init__.py']
    found = importlib.util.find_spec('anarci')
    if found is None or not found.origin or Path(found.origin).resolve() != expected:
        raise AssetError('Installed ANARCI is not the explicitly pinned package')
    executable = shutil.which('hmmscan')
    if executable is None or Path(executable).resolve() != bundle['hmmscan']:
        raise AssetError('PATH hmmscan differs from the explicitly pinned executable')
    package = importlib.import_module('anarci')
    core = importlib.import_module('anarci.anarci')
    if Path(core.HMM_path).resolve() != bundle['hmms']['ALL.hmm'].parent:
        raise AssetError('Loaded ANARCI resolves a different HMM database')
    for name, path in bundle['numbering_sources'].items():
        module_name = 'anarci' if name == '__init__.py' else 'anarci.' + name[:-3]
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != path:
            raise AssetError('Loaded numbering source differs from the pinned module')
    rows = []
    for record in spec['records']:
        numbered = {role: _number_chain(record[key], role, package.anarci)
                    for role, key in (('H', 'heavy'), ('L', 'light'))}
        rows.append(dict(id=record['id'], numbered=numbered))
    verify_unchanged(bundle['pins'])
    return rows


def _number_all(records, root, assets, output):
    # Native ANARCI uses temporary HMMER input files. Isolate their location in
    # a child, without changing caller environment or monkeypatching ANARCI.
    with tempfile.TemporaryDirectory(prefix='numbering_', dir=output) as scratch:
        scratch = Path(scratch)
        request, response = scratch / 'request.json', scratch / 'response.json'
        write_json(request, dict(project_root=str(root), assets=assets, records=records))
        env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
                   PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        command = [sys.executable, '-B', '-m', 'imut_cdr_gc.folding',
                   '--number-worker', str(request), str(response)]
        run = subprocess.run(command, env=env, cwd=root, text=True, capture_output=True,
                             timeout=max(120, len(records) * 60), check=False)
        if run.returncode != 0:
            raise InputError('Isolated numbering failed: ' + (run.stderr or run.stdout)[-4000:])
        result = json.loads(response.read_text())
    if [r.get('id') for r in result] != [r['id'] for r in records]:
        raise InputError('Numberer did not preserve the complete ordered candidate identities')
    return [row['numbered'] for row in result]


def _tree(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise AssetError('Native source changed before compilation')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        return ast.parse(data, filename=str(path))


def _native_namespace(sources, *, include_models=True):
    import numpy as np
    import torch
    namespace = {'__name__': 'imut_cdr_gc._isolated_native_ab2', 'np': np, 'torch': torch}
    names = ['constants.py']
    if include_models:
        from einops import rearrange
        namespace['rearrange'] = rearrange
        names += ['rigids.py', 'models.py']
    names += ['util.py', 'ABodyBuilder2.py']
    for name in names:
        tree = _tree(sources[name], SOURCE_SHA256[name])
        if name == 'util.py':
            kept = {'get_one_hot', 'get_encoding', 'find_alignment_transform', 'to_pdb'}
            body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in kept]
            if {n.name for n in body} != kept:
                raise AssetError('Native utility definitions absent')
        elif name == 'ABodyBuilder2.py':
            body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Antibody']
            if len(body) != 1:
                raise AssetError('Native Antibody definition absent')
            body[0].body = [n for n in body[0].body if isinstance(n, ast.FunctionDef)
                           and n.name in ('__init__', 'save_single_unrefined')]
        else:
            body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(sources[name]), 'exec'), namespace)
    return namespace


class _NativePredictor:
    def __init__(self, bundle, device):
        import torch
        if torch.get_default_dtype() != torch.float32:
            raise InputError('Native fold profile requires the existing default dtype float32')
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise InputError('Explicit CUDA device is unavailable; no CPU fallback')
        self.native = _native_namespace(bundle['sources'])
        self.models = []
        for name, dimension in zip(WEIGHT_SHA256, PARAMETERS['embed_dims']):
            model = self.native['StructureModule'](rel_pos_dim=64, embed_dim=dimension)
            weights = torch.load(bundle['weights'][name], map_location='cpu', weights_only=True)
            model.load_state_dict(weights, strict=True)
            self.models.append(model.to(self.device).eval())
        self.runtime = dict(torch=torch.__version__, numpy=importlib.import_module('numpy').__version__,
                            einops=importlib.import_module('einops').__version__, device=str(self.device))

    def predict(self, record, numbered):
        import torch
        sequences = {'H': record['heavy'], 'L': record['light']}
        with torch.no_grad():
            encoding = torch.tensor(self.native['get_encoding'](sequences),
                                    device=self.device, dtype=torch.float32)
            outputs = [model(encoding, sequences['H'] + sequences['L']) for model in self.models]
        prediction = self.native['Antibody'](numbered, outputs)
        if not torch.isfinite(prediction.error_estimates).all():
            raise InputError('Nonfinite native ensemble dispersion; no ranking fallback')
        return prediction


def _load_predictor(bundle, device):
    return _NativePredictor(bundle, device)


def _validate_numbered(record, numbered):
    if not isinstance(numbered, Mapping) or set(numbered) != {'H', 'L'}:
        raise InputError('Numberer must return exactly H and L')
    normalized = {}
    for role, key in (('H', 'heavy'), ('L', 'light')):
        values = numbered[role]
        if ''.join(x[1] for x in values) != record[key]:
            raise InputError('Numbered residues no longer equal the full input chain')
        normalized[role] = [((x[0][0], x[0][1]), x[1]) for x in values]
        keys = [x[0] for x in normalized[role]]
        if (len(keys) != len(set(keys)) or any(type(n) is not int or not 1 <= n <= 9999
            or not isinstance(i, str) or len(i) != 1 or (i != ' ' and not re.fullmatch('[A-Z]', i))
            for n, i in keys)):
            raise InputError('Invalid or duplicate Chothia residue identity')
    return normalized


def _verify_fold_pdb(path, record, numbered):
    from .complexes import inspect_pdb
    if atomic_chain_sequences(path) != {'H': record['heavy'], 'L': record['light']}:
        raise InputError('Predicted PDB full H/L differs from its candidate')
    try:
        inspect_pdb(path, require_complete_atoms=True)
    except ValueError as error:
        raise InputError('Predicted PDB atom completeness failed: ' + str(error)) from error
    expected = {role: [(n, ins, aa) for (n, ins), aa in numbered[role]] for role in ('H', 'L')}
    observed = {'H': [], 'L': []}
    atoms = set()
    for line in path.read_text().splitlines():
        if not line.startswith('ATOM  '):
            raise InputError('Native unrefined PDB contains unexpected non-ATOM records')
        try:
            role, number, insertion = line[21], int(line[22:26]), line[26]
            aa = THREE_TO_ONE[line[17:20]]
            xyz = [float(line[i:i+8]) for i in (30, 38, 46)]
        except (IndexError, ValueError, KeyError) as exc:
            raise InputError('Malformed predicted atom identity') from exc
        if role not in observed or line[16] != ' ' or not all(math.isfinite(v) for v in xyz):
            raise InputError('Unexpected chain, alternate location or nonfinite predicted atom')
        atom = line[12:16].strip()
        key = role, number, insertion, atom
        if key in atoms:
            raise InputError('Duplicate predicted atom')
        atoms.add(key)
        if atom == 'CA':
            observed[role].append((number, insertion, aa))
    if observed != expected:
        raise InputError('PDB atom correspondence differs from Chothia numbering')
    for role, residues in expected.items():
        for number, insertion, _ in residues:
            if any((role, number, insertion, atom) not in atoms for atom in ('N', 'CA', 'C', 'O')):
                raise InputError('Predicted residue lacks a backbone atom')
    return dict(atom_count=len(atoms), residue_counts={k: len(v) for k, v in expected.items()},
                chothia_residues={k: [[n, i, aa] for n, i, aa in v] for k, v in expected.items()})


def fold_antibodies(records, *, project_root, output_dir, assets, device='cpu'):
    """True antibody-only folds; all-or-failed attempt, no automatic resumption.

    Returns (rows, receipt); writes fold_manifest.jsonl/receipt.json only after
    every record succeeds. A failed attempt retains partial.jsonl/failure.json
    and its own PDBs as incomplete evidence, never as a completed delivery.
    """
    if device != 'cpu' and (not isinstance(device, str) or re.fullmatch(r'cuda:\d+', device) is None):
        raise InputError('Device must be cpu or an explicit cuda:N; no automatic GPU choice')
    root = Path(project_root).resolve(strict=True)
    rows = [canonical_record(deepcopy(row)) for row in records]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise InputError('Nonempty records with unique candidate IDs are required')
    for row in rows:
        if FOLD_FIELDS.intersection(row):
            raise InputError('Existing fold evidence must not be silently replaced')
        if not isinstance(row.get('antigen_id'), str) or not row['antigen_id'].strip():
            raise InputError('Every candidate requires its explicit antigen_id')
        if row.get('full_chain_reconstruction_required') or any(len(row[k]) <= 70 for k in ('heavy', 'light')):
            raise InputError('Supply complete reconstructed variable H/L domains, not FR277 or fragments')
    frozen_rows = identity_sha(rows)  # JSON-safe finite metadata; not a sequence-only fingerprint.
    bundle = _verify_assets(root, assets)  # All sources/weights before model imports.
    output = new_output(root, output_dir)
    completed, current = [], None
    common = dict(profile=PROFILE, parameters=deepcopy(PARAMETERS), assets=bundle['receipt'],
        candidate_records_sha256=frozen_rows, device=device, requested=len(rows),
        adapter_sha256=file_sha256(Path(__file__)),
        scientific_selection_performed=False, complex_pdbs_created=0,
        learned_weight_predictor_parity_verified=False)
    try:
        numbered = _number_all(rows, root, assets, output)
        if len(numbered) != len(rows):
            raise InputError('Incomplete numbering batch')
        numbered = [_validate_numbered(row, num) for row, num in zip(rows, numbered)]
        predictor = _load_predictor(bundle, device)
        for index, (row, num) in enumerate(zip(rows, numbered), 1):
            current = row['id']
            prediction = predictor.predict(row, num)
            ranking = list(prediction.ranking)
            if len(ranking) != 3 or set(ranking) != {0, 1, 2} or any(type(i) is not int for i in ranking):
                raise InputError('Native three-model ranking is invalid')
            path = output / f'antibody_{index:06d}.pdb'
            if path.exists():
                raise InputError('Refusing to overwrite a predicted PDB')
            prediction.save_single_unrefined(str(path), index=ranking[0])
            atom_identity = _verify_fold_pdb(path, row, num)
            digest = file_sha256(path)
            provenance = dict(kind='antibody_fold', method='ABodyBuilder2', profile=PROFILE,
                numbering='chothia', refined=False, heavy_sha256=sequence_sha(row['heavy']),
                light_sha256=sequence_sha(row['light']), antigen_id=row['antigen_id'],
                pdb_sha256=digest, model_ids=[1, 2, 3], model_ranking=[i + 1 for i in ranking],
                saved_model_id=ranking[0] + 1, save_method=PARAMETERS['save_method'],
                asset_identity_sha256=identity_sha(bundle['receipt']), **atom_identity)
            completed.append(dict(row, pdb_path=str(path.relative_to(root)), pdb_sha256=digest,
                                  numbering='chothia', fold_status='ok', fold_provenance=provenance))
        verify_unchanged(bundle['pins'])
        if identity_sha(rows) != frozen_rows:
            raise InputError('Candidate metadata changed during prediction')
        manifest = output / 'fold_manifest.jsonl'
        write_jsonl(manifest, completed)
        receipt = dict(common, status='complete_antibody_folds_not_complexes', folded=len(completed),
            runtime=dict(predictor.runtime), output_manifest=asset_receipt(root, manifest),
            pdbs=[dict(id=r['id'], path=r['pdb_path'], sha256=r['pdb_sha256']) for r in completed])
        write_json(output / 'receipt.json', receipt)
        return completed, receipt
    except BaseException as error:
        # No success marker after failed inference, including OOM/interruption.
        write_jsonl(output / 'partial.jsonl', completed)
        write_json(output / 'failure.json', dict(common, status='failed', completed_before_failure=len(completed),
            failed_candidate=current, error_type=type(error).__name__, error=str(error),
            resource_error=is_resource_failure(error), partial_outputs_are_not_complete=True))
        raise


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--number-worker':
        raise SystemExit('Internal isolated numbering worker only; use fold_antibodies API')
    write_json(Path(sys.argv[3]), _number_worker(json.loads(Path(sys.argv[2]).read_text())))
