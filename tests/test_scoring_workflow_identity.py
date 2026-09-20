"""Independent regression cases for scorer/lineage/export identity boundaries.

All sequences, coordinates and scores are synthetic engineering fixtures.
Nothing in this file runs a learned model or certifies a scientific library.
"""
from copy import deepcopy
import hashlib

import pytest

from imut_cdr_gc.preparation import prepare_founder, reconstruct
from imut_cdr_gc.selection import measured_score, select_records
from imut_cdr_gc.workflow import check_stepwise_trace


def _sha(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def _founder():
    fr = ['X'] * 277
    for start, motif in [(26, 'ACD'), (55, 'EFG'), (102, 'HIK'),
                          (165, 'LMN'), (194, 'PQR'), (241, 'STV')]:
        fr[start:start+3] = motif
    return prepare_founder(dict(id='independent_fixture_founder', design='fixture',
        antigen_id='fixture_antigen', heavy='WLMNYPQRWSTVY', light='WACDYEFGWHIKY',
        fr_cdr_seq=''.join(fr)))


def _measurement(identifier, heavy, light):
    protocol = 'deepcdr_esm2_hfirst_anarci_list_index95_v1'
    return dict(id=identifier, model='DeepCDR-ESM2', protocol=protocol,
        identity=dict(protocol=protocol), status='ok', score=0.8, inputs=dict(
            heavy_sha256=_sha(heavy), light_sha256=_sha(light), antigen={'id': 'fixture_antigen'}))


def _policy(version='v1'):
    return dict(version=version, route='esm2_only', target=1, common_deepcdr_floor=0.5,
        likelihood_calibration=dict(lower_quantile=0.05, lower_cutoff=-1.9,
            upper_quantile=None, upper_cutoff=None, metric_identity='fixture_only',
            source_identity={'fixture': 'not_learned'}, reference_sha256='a'*64,
            reference_not_candidates=True))


def _row(k=3, version='v1'):
    founder = _founder()
    mutant = list(founder['fr_cdr_seq'])
    for position in list(founder['cdr_position_map'])[:k]:
        p = int(position)
        mutant[p] = 'A' if mutant[p] != 'A' else 'C'
    rebuilt = reconstruct(founder, ''.join(mutant))
    row = dict(rebuilt, id='independent_fixture', model='iMut-CDR-JM-Epi',
        version=version, design=founder['design'], antigen_id=founder['antigen_id'], founder=founder)
    row['scores'] = {'esm2': _measurement(row['id'], row['heavy'], row['light'])}
    row['likelihood'] = dict(id=row['id'], heavy_sha256=_sha(row['heavy']),
        light_sha256=_sha(row['light']), status='ok', score=-1., metric_identity='fixture_only',
        source_identity={'fixture': 'not_learned'})
    return row


def test_same_sequence_and_checkpoint_name_do_not_authorize_old_input_protocol():
    row = _row()
    assert measured_score(row, 'esm2') == 0.8
    for place in ('top', 'identity'):
        altered = deepcopy(row)
        target = altered['scores']['esm2']
        if place == 'identity': target = target['identity']
        target['protocol'] = 'historical_generator_cdr_layout'
        assert measured_score(altered, 'esm2') is None


def test_likelihood_cannot_be_reused_for_another_candidate_or_changed_chain():
    row = _row()
    assert len(select_records([row], _policy())[0]) == 1
    for field, value in [('id', 'other_candidate'), ('heavy_sha256', 'b'*64), ('light_sha256', 'c'*64)]:
        altered = deepcopy(row)
        altered['likelihood'][field] = value
        chosen, summary = select_records([altered], _policy())
        assert not chosen
        assert summary['counts']['likelihood_unmeasured_error_or_identity_mismatch'] == 1


def _trace_row():
    row = _row(k=11, version='v3')
    parent = {role: row['founder'][role] for role in ('heavy', 'light')}
    row['stepwise_trace'] = []
    for round_id, changes in enumerate((row['mutations'][:10], row['mutations'][10:]), 1):
        child = dict(parent)
        for change in changes:
            sequence = list(child[change['chain']])
            sequence[change['chain_index']] = change['to']
            child[change['chain']] = ''.join(sequence)
        identifier = f'independent_proposal_{round_id}'
        row['stepwise_trace'].append(dict(round=round_id, proposal_id=identifier,
            parent=parent, child=child,
            requested_sites=[dict(chain=m['chain'], index=m['chain_index']) for m in changes],
            scores={'esm2': _measurement(identifier, child['heavy'], child['light'])}))
        parent = dict(child)
    return row


def test_v3_cannot_request_unchanged_framework_sites_or_repeat_proposal_identity():
    row = _trace_row()
    check_stepwise_trace(row, _policy('v3'))
    altered = deepcopy(row)
    # Framework residue 0 remains unchanged: change-subset checking alone
    # cannot detect this forbidden request, so the actual founder map is needed.
    altered['stepwise_trace'][1]['requested_sites'].append(dict(chain='heavy', index=0))
    with pytest.raises(ValueError):
        check_stepwise_trace(altered, _policy('v3'))
    altered = deepcopy(row)
    duplicate = altered['stepwise_trace'][0]['proposal_id']
    altered['stepwise_trace'][1]['proposal_id'] = duplicate
    altered['stepwise_trace'][1]['scores']['esm2']['id'] = duplicate
    with pytest.raises(ValueError):
        check_stepwise_trace(altered, _policy('v3'))


def _pdb(chains):
    from imut_cdr_gc.complexes import RESIDUES, SIDECHAINS
    inverse = {aa: name for name, aa in RESIDUES.items()}
    lines, serial = [], 0
    for chain, sequence in chains.items():
        for number, aa in enumerate(sequence, 1):
            for atom in ['N', 'CA', 'C', 'O', *SIDECHAINS[aa].split()]:
                serial += 1
                lines.append(f'ATOM  {serial:5d} {atom:^4s} {inverse[aa]:3s} {chain}{number:4d}    '
                    f'{float(number):8.3f}{float(serial % 5):8.3f}{0.:8.3f}  1.00 20.00          {atom[0]:>2s}\n')
    return ''.join(lines) + 'END\n'


def test_export_cannot_replace_antigen_and_its_declaration_together(tmp_path):
    from imut_cdr_gc.complexes import export_library
    from imut_cdr_gc.records import sha256
    row = _row()
    founder = deepcopy(row['founder'])
    reference = tmp_path/'founder.pdb'
    reference.write_text(_pdb({'H': founder['heavy'], 'L': founder['light'], 'A': 'GG'}))
    founder['complex_reference'] = dict(path='founder.pdb', sha256=sha256(reference),
        antigen_id=founder['antigen_id'], heavy_chain='H', light_chain='L', antigen_chains={'A': 'GG'})
    row['founder'] = deepcopy(founder)
    mutant = tmp_path/'mutant.pdb'
    mutant.write_text(_pdb({'H': row['heavy'], 'L': row['light'], 'A': 'GG'}))
    row['complex'] = dict(path='mutant.pdb', sha256=sha256(mutant), antigen_id=founder['antigen_id'],
        heavy_chain='H', light_chain='L', antigen_chains={'A': 'GG'},
        provenance=dict(kind='founder_sidechain_reconstruction', synthetic_fixture_not_physical_model=True))
    success = export_library([row], project_root=tmp_path, output_dir='valid_fixture', prepared_founder=founder)
    assert success['count'] == 1 and success['scientific_protocol_acceptance'] is False
    mutant.write_text(_pdb({'H': row['heavy'], 'L': row['light'], 'A': 'GA'}))
    row['complex'].update(sha256=sha256(mutant), antigen_chains={'A': 'GA'})
    with pytest.raises(ValueError):
        export_library([row], project_root=tmp_path, output_dir='wrong_antigen_fixture', prepared_founder=founder)
    assert not (tmp_path/'wrong_antigen_fixture').exists()
