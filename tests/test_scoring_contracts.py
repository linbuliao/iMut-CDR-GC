"""Small CPU contract fixtures; no learned-quality or performance benchmark."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from imut_cdr_gc.assets import AssetError, file_sha256, project_path, verified_asset
from imut_cdr_gc.scoring import score_3d, score_esm2
from imut_cdr_gc.scoring.common import canonical_record, sequence_sha
from imut_cdr_gc.scoring.native_cdr95 import canonical_esm_input
from imut_cdr_gc.scoring.native_structure import verify_mutant_fold

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'imut_cdr_gc/scoring/_vendor'


def atom(serial, number, x, *, chain='H', aa='ALA'):
    return f'ATOM  {serial:5d}  CA  {aa:3s} {chain}{number:4d}    {x:8.3f}{0.:8.3f}{0.:8.3f}  1.00 20.00           C  \n'


def spec(path, root):
    return dict(path=str(path.relative_to(root)), sha256=file_sha256(path))


def numbered(pairs, **kwargs):
    assert kwargs == dict(scheme='imgt', output=False, ncpu=1, allow={'H','K','L'})
    numbering = [[([((7000+i, ' '), aa) for i, aa in enumerate(sequence)], 0, len(sequence)-1)] for role,sequence in pairs]
    details = [[dict(chain_type='H')], [dict(chain_type='K')]]
    return numbering, details, None


@pytest.fixture
def payload(tmp_path):
    antigen_path = tmp_path / 'antigen.pdb'
    antigen_path.write_text(atom(1, 1, 0, chain='A', aa='GLY') + atom(2, 2, 3, chain='A', aa='GLY'))
    antigen = dict(id='fixture-antigen', pocket_pdb=spec(antigen_path, tmp_path), chains={'A': 'GG'})
    record = dict(id='fixture', antigen_id=antigen['id'], heavy='A'*128, light='C'*128,
                  cdr_seq='GENERATOR_LAYOUT_MUST_NOT_BE_USED')
    weights = tmp_path / 'fixture_weights.pt'
    weights.write_bytes(b'Explicit non-model fixture; no inference allowed')
    features = tmp_path / 'features.json'
    features.write_text('{}')
    assets = dict(weights=spec(weights, tmp_path), features=spec(features, tmp_path), allow_nonreference_checkpoint=True)
    return record, antigen, assets


@pytest.fixture
def fold_payload(tmp_path, payload):
    row, antigen, assets = payload
    row = dict(row, heavy='AA', light='CC', numbering='chothia')
    fold = tmp_path / 'fold.pdb'
    fold.write_text(atom(1, 26, 0) + atom(2, 27, 3) + atom(3, 24, 0, chain='L', aa='CYS') + atom(4, 25, 3, chain='L', aa='CYS'))
    row.update(pdb_path='fold.pdb', pdb_sha256=file_sha256(fold), fold_provenance=dict(
        kind='antibody_fold', method='explicit synthetic coordinate fixture, not real folding', numbering='chothia',
        heavy_sha256=sequence_sha(row['heavy']), light_sha256=sequence_sha(row['light']), pdb_sha256=file_sha256(fold)))
    return row, antigen, assets


def test_aliases_are_accepted_but_conflicts_never_silently_chosen():
    row = canonical_record(dict(id='a', heavy_full='AC', light_full='DE'))
    assert row['heavy'] == 'AC' and row['light'] == 'DE'
    with pytest.raises(ValueError, match='disagree'):
        canonical_record(dict(id='a', heavy='AC', heavy_full='AA', light='DE'))
    with pytest.raises(ValueError, match='disagree'):
        canonical_record(dict(id='a', mutant_id='b', heavy='AC', light='DE'))


def test_native95_uses_list_indices_actual_chains_and_heavy_first(tmp_path, payload):
    row, _, _ = payload
    prepared = canonical_esm_input(row, project_root=tmp_path, numberer=numbered)
    expected = 'X'.join(['A'*10, 'A'*10, 'A'*13+'X'*12, 'C'*10, 'C'*10, 'C'*13+'X'*12])
    assert prepared['scorer_cdr_seq'] == expected and len(expected) == 95
    assert not prepared['generation_cdr_seq_used']
    assert all(s['imgt_number'] >= 7000 for s in prepared['slot_mapping'] if s['kind'] == 'residue')
    assert prepared['domains']['heavy']['input_sha256'] == sequence_sha(row['heavy'])
    assert not list((tmp_path/'.cache/tmp').iterdir())


@pytest.mark.parametrize('failure', ['swapped', 'missing', 'multiple', 'truncated', 'sequence_mismatch', 'empty_window'])
def test_native95_rejects_bad_numbering_without_zero_padding_missing_domains(tmp_path, payload, failure):
    row, _, _ = payload
    def bad(pairs, **kwargs):
        n,d,h = numbered(pairs, **kwargs)
        if failure == 'swapped': d[0][0]['chain_type'] = 'K'
        elif failure == 'missing': n[0] = None
        elif failure == 'multiple': n[0] = n[0] * 2
        elif failure == 'truncated': n[0] = [(n[0][0][0][:117], 0, 116)]
        elif failure == 'sequence_mismatch': n[0][0][0][30] = ((7030,' '),'W')
        elif failure == 'empty_window':
            n[0][0][0][27:39] = [((7000+i,' '),'-') for i in range(27,39)]
        return n,d,h
    with pytest.raises(ValueError):
        canonical_esm_input(row, project_root=tmp_path, numberer=bad)


def test_explicit_assets_missing_changed_and_escape(tmp_path):
    path=tmp_path/'model.pt'; path.write_bytes(b'model fixture')
    declared=spec(path,tmp_path)
    assert verified_asset(tmp_path, declared) == path
    path.write_bytes(b'changed')
    with pytest.raises(AssetError, match='hash'): verified_asset(tmp_path,declared)
    with pytest.raises(AssetError, match='missing'): verified_asset(tmp_path,dict(path='absent',sha256='a'*64))
    with pytest.raises(AssetError, match='escapes'): project_path(tmp_path,'../outside')
    with pytest.raises(AssetError, match='SHA256'): verified_asset(tmp_path,dict(path='model.pt',sha256=None))


def test_esm_routing_error_and_below_threshold_are_distinct(tmp_path, payload, monkeypatch):
    row, antigen, assets = payload
    import imut_cdr_gc.scoring.backend as backend
    observed = []
    def mocked(prepared, antigen_path, *args, **kwargs):
        observed.append(prepared)
        return 0.2, -1.386, {'fixture': True, 'device': 'cpu'}
    monkeypatch.setattr(backend, 'predict_esm2', mocked)
    ok=score_esm2(row,project_root=tmp_path,assets=assets,antigen=antigen,numberer=numbered)
    assert ok['status']=='ok' and ok['score']==0.2 and ok['selection_decision'] is None
    assert not ok['identity']['reference_checkpoint']
    assert observed[0]['scorer_cdr_seq'] != row['cdr_seq']
    bad=score_esm2(dict(row,antigen_id='wrong'),project_root=tmp_path,assets=assets,antigen=antigen,numberer=numbered)
    assert bad['status']=='invalid_input' and bad['score'] is None and len(observed)==1
    strict=score_esm2(row,project_root=tmp_path,assets=dict(assets,allow_nonreference_checkpoint=False),antigen=antigen,numberer=numbered)
    assert strict['status']=='asset_error' and strict['score'] is None and len(observed)==1


def test_esm_dependency_and_nonfinite_failures_are_not_scores(tmp_path,payload,monkeypatch):
    row,antigen,assets=payload
    import imut_cdr_gc.scoring.backend as backend
    def missing(*args,**kwargs): raise ModuleNotFoundError('Explicit fixture dependency')
    monkeypatch.setattr(backend,'predict_esm2',missing)
    failed=score_esm2(row,project_root=tmp_path,assets=assets,antigen=antigen,numberer=numbered)
    assert failed['status']=='dependency_error' and failed['binding_score'] is None
    monkeypatch.setattr(backend,'predict_esm2',lambda *a,**k:(float('nan'),0.0,{'fixture':True}))
    failed=score_esm2(row,project_root=tmp_path,assets=assets,antigen=antigen,numberer=numbered)
    assert failed['status']=='inference_error' and failed['score'] is None


@pytest.mark.parametrize('failure', [MemoryError('Fixture memory exhaustion'), RuntimeError("DefaultCPUAllocator: can't allocate memory")])
def test_resource_exhaustion_is_not_an_asset_or_cutoff_failure(tmp_path, payload, monkeypatch, failure):
    row, antigen, assets = payload
    import imut_cdr_gc.scoring.backend as backend
    def exhausted(*args, **kwargs): raise failure
    monkeypatch.setattr(backend, 'predict_esm2', exhausted)
    failed = score_esm2(row, project_root=tmp_path, assets=assets, antigen=antigen, numberer=numbered)
    assert failed['status'] == 'resource_error' and failed['score'] is None
    assert failed['identity']['assets']['weights']['sha256'] == assets['weights']['sha256']


@pytest.mark.parametrize('change',['wrong_chain','sidechain','numbering','source_sequence','antigen_sequence','antigen_hash'])
def test_3d_rejects_fold_and_antigen_identity_errors_before_backend(tmp_path,fold_payload,monkeypatch,change):
    row,antigen,assets=copy.deepcopy(fold_payload)
    import imut_cdr_gc.scoring.backend as backend
    monkeypatch.setattr(backend,'predict_3d',lambda *a,**k:pytest.fail('Invalid identity reached inference'))
    if change=='wrong_chain': row['light']='DD'; row['fold_provenance']['light_sha256']=sequence_sha('DD')
    elif change=='sidechain': row['fold_provenance']['kind']='founder_sidechain_reconstruction'
    elif change=='numbering': row['numbering']='imgt'
    elif change=='source_sequence': row['fold_provenance']['heavy_sha256']='0'*64
    elif change=='antigen_sequence': antigen['chains']={'A':'AA'}
    elif change=='antigen_hash': row['antigen_pocket_sha256']='0'*64
    measured=score_3d(row,project_root=tmp_path,assets=assets,antigen=antigen)
    assert measured['status']=='invalid_input' and measured['score'] is None


def test_3d_routes_actual_fold_and_antigen_explicitly(tmp_path,fold_payload,monkeypatch):
    row,antigen,assets=fold_payload
    import imut_cdr_gc.scoring.backend as backend
    def mocked(fold,ag,*args,**kwargs):
        assert fold.name=='fold.pdb' and ag.name=='antigen.pdb'
        return 0.4,-0.405,{'fixture':True,'device':'cpu'}
    monkeypatch.setattr(backend,'predict_3d',mocked)
    measured=score_3d(row,project_root=tmp_path,assets=assets,antigen=antigen)
    assert measured['status']=='ok' and measured['structure_score']==0.4
    assert measured['identity']['graph_roles']==dict(first_GCN='antigen_pocket',second_GCN='antibody_CDR')
    assert measured['inputs']['exact_full_heavy_light_sequence_match']


def test_changed_asset_during_mock_inference_is_invalidated(tmp_path,fold_payload,monkeypatch):
    row,antigen,assets=fold_payload
    import imut_cdr_gc.scoring.backend as backend
    def change(fold,*args,**kwargs):
        fold.write_text(fold.read_text()+'REMARK altered\n')
        return 0.9,2.2,{'fixture':True}
    monkeypatch.setattr(backend,'predict_3d',change)
    measured=score_3d(row,project_root=tmp_path,assets=assets,antigen=antigen)
    assert measured['status']=='asset_error' and measured['score'] is None


def test_import_is_lazy_and_does_not_discover_research_root():
    code="import sys; import imut_cdr_gc.scoring; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert 'anarci' not in sys.modules"
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_cpu_synthetic_state_roundtrip_both_models(tmp_path,fold_payload,payload):
    """Actual CPU forwards with random fixture states; scores have no quality meaning."""
    torch=pytest.importorskip('torch')
    pytest.importorskip('torch_geometric')
    from imut_cdr_gc.scoring._vendor.deepcdr3d_model import DeepCDR3DNet
    from imut_cdr_gc.scoring._vendor.esm2_model import GCNNet
    from imut_cdr_gc.scoring.backend import load_model
    torch.set_num_threads(1)
    torch.manual_seed(1921)
    row3d,antigen,_=fold_payload
    rowesm,_,_=payload
    for name,model,dimension,row,score_fn in [
        ('3d',DeepCDR3DNet(),30,row3d,score_3d),
        ('esm2',GCNNet(),20,rowesm,score_esm2)]:
        weights=tmp_path/(name+'_synthetic.pt');torch.save(model.state_dict(),weights)
        features=tmp_path/(name+'_features.json');features.write_bytes((VENDOR/f'amino_acid_vectors_{dimension}dim.json').read_bytes())
        assets=dict(weights=spec(weights,tmp_path),features=spec(features,tmp_path),allow_nonreference_checkpoint=True)
        kwargs=dict(project_root=tmp_path,assets=assets,antigen=antigen,threads=1)
        if name=='esm2':kwargs['numberer']=numbered
        output=score_fn(row,**kwargs)
        assert output['status']=='ok',output
        assert math.isfinite(output['score']) and 0<=output['score']<=1
        assert output['identity']['runtime']['device']=='cpu'
        assert output['identity']['runtime']['weights_only'] is True
        assert not output['identity']['reference_checkpoint']
    load_model.cache_clear()
