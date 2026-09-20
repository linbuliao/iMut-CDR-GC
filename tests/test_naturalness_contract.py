"""Isolated CPU fixtures; no model import, private weights or GPU execution."""
import ast
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from imut_cdr_gc import naturalness as nat
from imut_cdr_gc import _likelihood_native as loader
from imut_cdr_gc.assets import AssetError, file_sha256

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH_CHECKOUT = Path(__file__).resolve().parents[3]


def records():
    return [dict(id='one', model=nat.CAMPAIGN_MODEL, heavy='A'*70, light='A'*70),
            dict(id='two', model=nat.CAMPAIGN_MODEL, heavy='C'*70, light='C'*70),
            dict(id='same_sequence_new_id', model=nat.CAMPAIGN_MODEL, heavy='A'*70, light='A'*70)]


class Numberer:
    def __init__(self, mode='ok'):
        self.mode, self.calls = mode, []

    def __call__(self, pairs, **kwargs):
        self.calls.append((deepcopy(pairs), kwargs))
        chain = 'H' if len(self.calls) % 2 else 'K'
        numbering, details = [], []
        for key, sequence in pairs:
            numbered = [((low+i, ' '), sequence[0]) for low, high in nat.RANGES for i in range(3)]
            domain = (numbered, 0, 20)
            if self.mode == 'missing' and key == '1':
                numbering.append(None); details.append(None)
            elif self.mode == 'multiple' and key == '1':
                numbering.append([domain, domain]); details.append([{'chain_type': chain}]*2)
            else:
                numbering.append([domain]); details.append([{'chain_type': chain}])
        if self.mode == 'short':
            numbering = numbering[:-1]
        if self.mode == 'oom':
            raise MemoryError('synthetic CPU allocation failure')
        return numbering, details, []


class Scorer:
    def __init__(self, mode='ok'):
        self.mode, self.calls = mode, []

    def score_many(self, sequences, **kwargs):
        self.calls.append((list(sequences), kwargs))
        if self.mode == 'oom':
            raise RuntimeError('CUDA out of memory (synthetic fixture)')
        if self.mode == 'unknown':
            raise RuntimeError('synthetic unknown native execution failure')
        if self.mode == 'interrupt':
            raise KeyboardInterrupt('synthetic interrupt')
        output = []
        for sequence in sequences:
            positions = nat.native_positions(sequence)
            value = -2.0 if sequence[0] == 'A' else -3.0
            output.append(dict(seq_frcdr=sequence, sequence_length=298, positions_scored=positions,
                n_positions=len(positions), average_log_likelihood=value,
                log_likelihood=value*len(positions), temperature=1.0, skipped_positions=[],
                per_position=[dict(position=p, aa=sequence[p], log_prob=value) for p in positions]))
        if self.mode == 'short':
            return output[:-1]
        if self.mode == 'wrong_order':
            return output[1:]+output[:1]
        if self.mode == 'nan':
            output[0]['average_log_likelihood'] = float('nan')
        if self.mode == 'wrong_mask':
            output[0]['positions_scored'] = output[0]['positions_scored'][:-1]
        if self.mode == 'wrong_reduction':
            output[0]['average_log_likelihood'] = -1
        return output


class TempCase(unittest.TestCase):
    def setUp(self):
        parent = PROJECT/'.cache/tmp'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='naturalness_fixture_', dir=parent)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def asset(self, relative, content):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return dict(path=relative, sha256=file_sha256(path))

    def manifest(self):
        # Non-checkpoint bytes: no private or executable tensor asset is inspected.
        model = self.asset('native/model.py', 'MODEL_FIXTURE = True\n')
        scorer = self.asset('native/scorer.py', 'SCORER_FIXTURE = True\n')
        weights = self.asset('assets/weights.fixture', 'synthetic bytes, not model weights')
        config = self.asset('assets/config/config.json', json.dumps(dict(model_type='esm', max_position_embeddings=512)))
        vocab = self.asset('assets/config/vocab.txt', 'A\nC\nX\n')
        return dict(schema='campaign-likelihood-assets.v1', model_source=model,
            scorer_source=scorer, weights=weights, local_model_dir='assets/config',
            config_tokenizer_sha256={'config.json': config['sha256'], 'vocab.txt': vocab['sha256']})


class CalibrationTests(unittest.TestCase):
    def calibration(self):
        source = dict(source='fixture', runtime='cpu-fixture')
        cal = nat.calibrate(list(range(101)), 'fixture-id', .05, None, 'separate_reference', source_identity=source)
        cal.update(reference_sha256='a'*64, reference_not_candidates=True)
        return cal, source

    def test_campaign_p5_inclusive_no_upper(self):
        cal, source = self.calibration()
        nat.validate_campaign_calibration(cal, 'fixture-id', source)
        labels, counts = nat.apply_calibration([None, 4.99, 5, 10000], cal, 'fixture-id', source_identity=source)
        self.assertEqual(labels, ['missing', 'below_lower', 'pass', 'pass'])

    def test_campaign_rejects_upper(self):
        cal, source = self.calibration()
        for field, value in [('upper_cutoff', 99), ('upper_quantile', .975), ('lower_quantile', .025)]:
            changed = dict(cal, **{field: value})
            with self.assertRaises(ValueError):
                nat.validate_campaign_calibration(changed, 'fixture-id', source)

    def test_campaign_requires_reference(self):
        cal, source = self.calibration()
        for field in ('reference_sha256', 'reference_not_candidates', 'reference_name', 'source_identity'):
            changed = dict(cal); changed.pop(field)
            with self.assertRaises(ValueError):
                nat.validate_campaign_calibration(changed, 'fixture-id', source)

    def test_source_runtime_mismatch(self):
        cal, source = self.calibration()
        with self.assertRaises(ValueError):
            nat.validate_campaign_calibration(cal, 'fixture-id', dict(source, runtime='other-device'))

    def test_metric_mismatch(self):
        cal, source = self.calibration()
        with self.assertRaises(ValueError):
            nat.validate_campaign_calibration(cal, 'another-id', source)

    def test_bool_and_nonfinite_not_measured(self):
        self.assertEqual(nat.score_qc([True, float('inf'), 'not-a-score', -1])['finite'], 1)

    def test_empty_qc(self):
        self.assertEqual(nat.score_qc([])['finite'], 0)
        self.assertTrue(nat.score_qc([])['degenerate'])


class ConversionTests(unittest.TestCase):
    def test_order_and_equal_sequence_ids_retained(self):
        numberer = Numberer()
        result = nat.convert_pairs(records(), numberer)
        self.assertEqual([r['id'] for r in result], [r['id'] for r in records()])
        self.assertEqual(result[0]['seq298'], result[2]['seq298'])
        self.assertEqual(len(numberer.calls), 2)
        self.assertEqual(numberer.calls[0][1], dict(scheme='imgt', output=False))
        self.assertEqual(numberer.calls[0][0], [('0', 'A'*70), ('1', 'C'*70), ('2', 'A'*70)])

    def test_duplicate_id_rejected(self):
        rows = records(); rows[2]['id'] = 'one'
        with self.assertRaises(ValueError):
            nat.convert_pairs(rows, Numberer())

    def test_both_chain_errors_retained(self):
        result = nat.convert_pairs(records(), Numberer('missing'))
        self.assertIsNone(result[1]['seq298'])
        self.assertEqual([e['chain'] for e in result[1]['conversion_errors']], ['heavy', 'light'])

    def test_ambiguous_domains_rejected(self):
        result = nat.convert_pairs(records(), Numberer('multiple'))
        self.assertIsNone(result[1]['seq298'])

    def test_incomplete_numberer_result_aborts(self):
        with self.assertRaises(nat.MeasurementError):
            nat.convert_pairs(records(), Numberer('short'))

    def test_numberer_oom_propagates(self):
        with self.assertRaises(MemoryError):
            nat.convert_pairs(records(), Numberer('oom'))

    def test_never_infer_founder(self):
        row = dict(id='one', heavy_full='A'*70, light_full='A'*70)
        result = nat.convert_pairs([row], Numberer())
        self.assertIsNone(result[0]['seq298'])
        self.assertEqual(len(result[0]['conversion_errors']), 2)

    def test_length_padding_alphabet(self):
        sequence = nat.convert_pairs(records(), Numberer())[0]['seq298']
        bad = [sequence[:-1], '?' + sequence[1:], 'X' + sequence[1:], sequence[:26]+'A'+sequence[27:]]
        for value in bad:
            with self.subTest(value=value[:30]):
                with self.assertRaises(ValueError):
                    nat.native_positions(value)

    def test_exact_native_layout_source_oracle(self):
        # Compile only original pure layout functions, never its torch import/model.
        path = RESEARCH_CHECKOUT/'model_cache/likelihood_native/cdr_likelihood.py'
        if not path.is_file():
            self.skipTest('Optional source-pinned oracle: authorized research source is not shipped in this package')
        self.assertEqual(file_sha256(path), loader.SOURCE_SHA256['scorer_source'])
        source = ast.parse(path.read_text())
        names = {'FR_CDR_LAYOUT', 'CDR_SEGMENTS', 'SEQUENCE_LENGTH', '_normalize_segment',
                 'build_frcdr298_sequence', 'validate_frcdr298_sequence'}
        chosen = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names
                  or isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in names for x in n.targets)]
        scope = dict(AA20_SET=nat.AA)
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]+chosen, type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(path), 'exec'), scope)
        light = {k: 'A'*(i+1) for i, k in enumerate(nat.SEGMENTS)}
        heavy = {k: 'C'*(i+1) for i, k in enumerate(nat.SEGMENTS)}
        arguments = {name+'_'+chain: parts[name] for chain, parts in [('light', light), ('heavy', heavy)] for name in nat.SEGMENTS}
        self.assertEqual(nat.build_layout298(heavy, light), scope['build_frcdr298_sequence'](**arguments))


class CampaignTests(TempCase):
    def run_fixture(self, name='attempt', *, scorer=None, numberer=None, calibration=None, rows=None):
        self.scorer = scorer or Scorer()
        with patch.object(nat, '_numbering_backend', return_value=(numberer or Numberer(), {'fixture': True, 'ranges': [(1, 26)]})), \
                patch.object(nat, 'load_native_scorer', return_value=(self.scorer, {'runtime': 'cpu-fixture'})):
            return nat.score_campaign_likelihood_pairs(rows or records(), name,
                project_root=self.root, assets={'fixture': True}, calibration=calibration)

    def test_complete_native_contract(self):
        rows, meta = self.run_fixture()
        self.assertEqual(meta['counts'], dict(input=3, scored=3, conversion_errors=0))
        self.assertFalse(meta['scientific_clearance'])
        self.assertEqual(self.scorer.calls[0][1], dict(batch_size=64, strict=True, return_per_position=True))
        self.assertEqual(len(self.scorer.calls), 1)
        self.assertEqual(rows[0]['average_log_likelihood'], -2)
        self.assertEqual(rows[0]['likelihood']['score'], -2)
        self.assertEqual(rows[0]['likelihood']['id'], 'one')
        self.assertEqual(rows[0]['likelihood']['heavy_sha256'], hashlib.sha256(('A'*70).encode()).hexdigest())
        self.assertEqual(rows[0]['likelihood']['light_sha256'], hashlib.sha256(('A'*70).encode()).hexdigest())
        self.assertEqual(rows[0]['likelihood']['native298_sha256'], hashlib.sha256(rows[0]['model_result']['seq_frcdr'].encode()).hexdigest())
        self.assertIsNone(rows[0]['p5_eligibility'])
        self.assertIsNone(rows[0]['gc_selected'])
        self.assertEqual(rows[0]['likelihood']['source_identity'], meta['source_identity'])

    def test_json_roundtrip_calibration_and_exact_p5(self):
        _, meta = self.run_fixture('identity')
        cal = dict(metric_identity=meta['metric_identity'], source_identity=meta['source_identity'],
            lower_quantile=.05, lower_cutoff=-2, upper_quantile=None, upper_cutoff=None,
            bounds='inclusive', reference_name='separate', reference_sha256='b'*64, reference_not_candidates=True)
        cal = json.loads(json.dumps(cal))
        rows, final = self.run_fixture('with_calibration', calibration=cal)
        self.assertEqual([r['p5_eligibility'] for r in rows], ['pass', 'below_reference_p5', 'pass'])
        self.assertEqual(final['counts']['below_reference_p5'], 1)

    def test_conversion_error_not_below_p5(self):
        rows, meta = self.run_fixture(numberer=Numberer('missing'))
        self.assertEqual(meta['counts'], dict(input=3, scored=2, conversion_errors=1))
        self.assertIsNone(rows[1]['average_log_likelihood'])
        self.assertEqual(rows[1]['status'], 'conversion_error')

    def test_conversion_failure_with_calibration_not_threshold_failure(self):
        _, identity = self.run_fixture('identity')
        calibration = dict(metric_identity=identity['metric_identity'], source_identity=identity['source_identity'],
            lower_quantile=.05, lower_cutoff=-2, upper_quantile=None, upper_cutoff=None,
            bounds='inclusive', reference_name='separate', reference_sha256='a'*64, reference_not_candidates=True)
        rows, meta = self.run_fixture(numberer=Numberer('missing'), calibration=calibration)
        self.assertEqual([r['p5_eligibility'] for r in rows], ['pass', 'missing', 'pass'])
        self.assertEqual(meta['counts']['below_reference_p5'], 0)
        self.assertEqual(meta['counts']['missing'], 1)

    def test_scoring_data_error_stops_without_retry(self):
        class RecordFailure(Scorer):
            def score_many(self, sequences, **kwargs):
                self.calls.append((list(sequences), kwargs))
                raise ValueError('synthetic per-record failure in whole native call')
        scorer = RecordFailure()
        with self.assertRaises(ValueError):
            self.run_fixture(scorer=scorer)
        self.assertEqual(len(scorer.calls), 1)
        self.assertFalse((self.root/'attempt/scores.jsonl').exists())

    def test_external_arm_rejected_before_loading_or_output(self):
        rows = records(); rows[1]['model'] = 'AntiBERTy'
        with patch.object(nat, 'load_native_scorer') as mocked:
            with self.assertRaises(ValueError):
                self.run_fixture(rows=rows)
            mocked.assert_not_called()
        self.assertFalse((self.root/'attempt').exists())

    def test_oom_aborts_without_low_score(self):
        with self.assertRaises(RuntimeError):
            self.run_fixture(scorer=Scorer('oom'))
        meta = json.loads((self.root/'attempt/metadata.json').read_text())
        self.assertEqual(meta['status'], 'failed')
        self.assertTrue(meta['error']['resource_error'])
        self.assertEqual(meta['scores_committed'], 0)
        self.assertFalse(meta['error_is_p5_failure'])
        self.assertFalse((self.root/'attempt/scores.jsonl').exists())
        self.assertEqual(len(self.scorer.calls), 1)

    def test_unknown_execution_failure_not_filtered(self):
        with self.assertRaises(RuntimeError):
            self.run_fixture(scorer=Scorer('unknown'))
        meta = json.loads((self.root/'attempt/metadata.json').read_text())
        self.assertEqual(meta['failed_stage'], 'scoring')
        self.assertFalse(meta['error']['resource_error'])
        self.assertEqual(meta['scores_committed'], 0)

    def test_interrupt_receipt(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(scorer=Scorer('interrupt'))
        meta = json.loads((self.root/'attempt/metadata.json').read_text())
        self.assertEqual(meta['status'], 'interrupted')
        self.assertFalse((self.root/'attempt/scores.jsonl').exists())

    def test_malformed_results_abort(self):
        for mode in ('short', 'wrong_order', 'nan', 'wrong_mask', 'wrong_reduction'):
            with self.subTest(mode=mode):
                with self.assertRaises(nat.MeasurementError):
                    self.run_fixture(mode, scorer=Scorer(mode))
                self.assertFalse((self.root/mode/'scores.jsonl').exists())

    def test_calibration_mismatch_no_scoring(self):
        scorer = Scorer()
        with self.assertRaises(ValueError):
            self.run_fixture(scorer=scorer, calibration={})
        self.assertEqual(scorer.calls, [])

    def test_duplicate_ids_no_output(self):
        rows = records(); rows[2]['id'] = rows[0]['id']
        with self.assertRaises(ValueError):
            self.run_fixture(rows=rows)
        self.assertFalse((self.root/'attempt').exists())

    def test_existing_likelihood_fields_cannot_be_overwritten(self):
        for field in ('likelihood', 'average_log_likelihood'):
            for value in (None, -2.0, {'status': 'ok', 'score': -2.0}):
                with self.subTest(field=field, value=value):
                    rows = records()
                    rows[0][field] = value
                    original = deepcopy(rows)
                    with self.assertRaisesRegex(ValueError, 'fresh candidate'):
                        self.run_fixture(rows=rows)
                    self.assertEqual(rows, original)
                    self.assertFalse((self.root/'attempt').exists())

    def test_persisted_score_output_integrates_with_real_workflow_selection(self):
        # Synthetic numberer/native-scorer values only. Conversion, output I/O,
        # ID/chain hashing, workflow selection and deduplication are real code.
        from imut_cdr_gc.workflow import select_run
        from imut_cdr_gc.selection import PROTOCOLS
        founder = dict(id='synthetic_founder', heavy='A'*70, light='A'*70)
        candidates = []
        for ident, heavy, light, probability in (
                ('pass', 'CCC'+'A'*67, 'A'*70, 0.9),
                ('below_p5', 'DDD'+'A'*67, 'CCC'+'A'*67, 0.95),
                ('duplicate_pass', 'CCC'+'A'*67, 'A'*70, 0.8)):
            mutations = [dict(chain=chain, chain_index=i, source=old, target=new)
                         for chain, sequence in (('heavy', heavy), ('light', light))
                         for i, (old, new) in enumerate(zip(founder[chain], sequence)) if old != new]
            item = dict(id=ident, model=nat.CAMPAIGN_MODEL, design='synthetic_design', version='v1',
                antigen_id='synthetic_antigen', heavy=heavy, light=light, founder=deepcopy(founder),
                mutations=mutations, mutation_count=len(mutations),
                lineage={'generation_protocol': 'explicit_synthetic_fixture', 'round': 1},
                scores={})
            for name, label in (('esm2', 'DeepCDR-ESM2'), ('3d', 'DeepCDR-3D')):
                item['scores'][name] = dict(id=ident, model=label, protocol=PROTOCOLS[name],
                    identity={'protocol': PROTOCOLS[name], 'fixture': True}, status='ok', score=probability,
                    inputs=dict(heavy_sha256=hashlib.sha256(heavy.encode()).hexdigest(),
                        light_sha256=hashlib.sha256(light.encode()).hexdigest(),
                        antigen={'id': item['antigen_id']}))
            candidates.append(item)
        original = deepcopy(candidates)
        _, identity = self.run_fixture('identity')
        calibration = dict(metric_identity=identity['metric_identity'], source_identity=identity['source_identity'],
            lower_quantile=.05, lower_cutoff=-2, upper_quantile=None, upper_cutoff=None,
            bounds='inclusive', reference_name='separate_synthetic_reference',
            reference_sha256='a'*64, reference_not_candidates=True)
        measured, _ = self.run_fixture('measurements', rows=candidates, calibration=calibration)
        self.assertEqual(candidates, original)
        self.assertEqual([r['id'] for r in measured], [r['id'] for r in candidates])
        for before, after in zip(candidates, measured):
            self.assertTrue(all(after[key] == value for key, value in before.items()))
        policy = dict(version='v1', route='esm2_OR_native3d', target=1,
                      common_deepcdr_floor=.5, likelihood_calibration=calibration)
        (self.root/'policy.json').write_text(json.dumps(policy))
        summary = select_run(self.root, 'measurements/scores.jsonl', 'policy.json', 'selected')
        delivered = nat.read_score_rows(self.root/'selected/selected.jsonl')
        self.assertEqual([r['id'] for r in delivered], ['pass'])
        self.assertEqual(summary['counts']['below_reference_p5'], 1)
        self.assertEqual(summary['counts']['duplicate_eligible_sequence_pair'], 1)
        self.assertEqual(summary['counts']['selected'], 1)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(delivered[0]['founder'], founder)
        self.assertEqual(delivered[0]['mutations'], candidates[0]['mutations'])
        self.assertEqual(delivered[0]['scores'], candidates[0]['scores'])
        self.assertEqual(delivered[0]['lineage'], candidates[0]['lineage'])
        self.assertEqual(summary['input_sha256'], file_sha256(self.root/'measurements/scores.jsonl'))
        self.assertFalse(summary['corrected_production_accepted'])

    def test_no_overwrite_or_escape(self):
        self.run_fixture()
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        with self.assertRaises(ValueError):
            self.run_fixture('../outside')

    def test_atomic_json_validation_no_partial(self):
        path = self.root/'result.jsonl'
        with self.assertRaises(ValueError):
            nat._write_rows(path, [{'score': 1}, {'score': float('nan')}])
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_temporary_numbering_restores(self):
        previous = tempfile.tempdir
        with self.assertRaises(MemoryError):
            self.run_fixture(numberer=Numberer('oom'))
        self.assertEqual(tempfile.tempdir, previous)
        self.assertFalse(list((self.root/'attempt').glob('numbering_*')))


class AssetTests(TempCase):
    def verify(self, manifest):
        pins = {k: manifest[k]['sha256'] for k in loader.SOURCE_SHA256}
        with patch.object(loader, 'SOURCE_SHA256', pins):
            return loader.verify_native_assets(self.root, manifest)

    def test_explicit_manifest(self):
        manifest = self.manifest()
        paths, directory, hashes, identity = self.verify(manifest)
        self.assertEqual(directory, self.root/'assets/config')
        self.assertEqual(identity['checkpoint_sha256'], manifest['weights']['sha256'])
        self.assertTrue(identity['local_files_only'])

    def test_missing_local_model_dir(self):
        manifest = self.manifest(); del manifest['local_model_dir']
        with self.assertRaises(AssetError):
            self.verify(manifest)

    def test_stale_weights(self):
        manifest = self.manifest()
        (self.root/manifest['weights']['path']).write_text('changed synthetic fixture')
        with self.assertRaises(AssetError):
            self.verify(manifest)

    def test_unpinned_config(self):
        manifest = self.manifest()
        self.asset('assets/config/tokenizer_config.json', '{}')
        with self.assertRaises(AssetError):
            self.verify(manifest)

    def test_stale_config(self):
        manifest = self.manifest()
        (self.root/'assets/config/vocab.txt').write_text('other-vocabulary')
        with self.assertRaises(AssetError):
            self.verify(manifest)

    def test_unknown_source_revision(self):
        with self.assertRaises(AssetError):
            loader.verify_native_assets(self.root, self.manifest())

    def test_outside_asset(self):
        manifest = self.manifest()
        manifest['weights']['path'] = '../not_in_project'
        with self.assertRaises(AssetError):
            self.verify(manifest)

    def test_remote_code_rejected(self):
        manifest = self.manifest()
        extra = self.asset('assets/config/tokenizer_config.json', '{"auto_map":{"AutoTokenizer":"remote-code"}}')
        manifest['config_tokenizer_sha256']['tokenizer_config.json'] = extra['sha256']
        with self.assertRaises(AssetError):
            self.verify(manifest)

    def test_safe_local_import_no_global_alias(self):
        source = self.asset('native/isolated.py', 'from imut_cdr_jm_model import MARKER\nRESULT = MARKER\n')
        model = ModuleType('private'); model.MARKER = 'synthetic'
        before_path = list(sys.path)
        before_alias = sys.modules.get('imut_cdr_jm_model')
        result = loader._execute_source(self.root/source['path'], expected_sha256=source['sha256'],
                                       imports={'imut_cdr_jm_model': model})
        self.assertEqual(result.RESULT, 'synthetic')
        self.assertEqual(sys.path, before_path)
        self.assertIs(sys.modules.get('imut_cdr_jm_model'), before_alias)

    def test_source_mutation_before_execution(self):
        source = self.asset('native/isolated.py', 'MARKER = True\n')
        (self.root/source['path']).write_text('MARKER = False\n')
        with self.assertRaises(AssetError):
            loader._execute_source(self.root/source['path'], expected_sha256=source['sha256'])

    def test_unsafe_pickle_fallback_disabled(self):
        calls = []
        guard = loader._SafeTorch(SimpleNamespace(load=lambda *a, **kw: calls.append((a, kw))))
        with self.assertRaises(AssetError):
            guard.load('fixture')
        self.assertEqual(calls, [])
        guard.load('fixture', weights_only=True)
        self.assertEqual(len(calls), 1)

    def test_explicit_native_constructor_wiring(self):
        manifest = self.manifest()
        pins = {k: manifest[k]['sha256'] for k in loader.SOURCE_SHA256}
        calls = []
        torch = SimpleNamespace(__version__='cpu-fixture', version=SimpleNamespace(cuda=None),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))))
        model = SimpleNamespace(**loader.ARCHITECTURE, torch=torch)
        scorer = SimpleNamespace(IMutCDRJMLikelihood=lambda **kw: calls.append(kw) or Scorer())
        with patch.object(loader, 'SOURCE_SHA256', pins), \
                patch.object(loader, '_execute_source', side_effect=[model, scorer]), \
                patch.dict(sys.modules, {'torch': torch, 'transformers': SimpleNamespace(__version__='fixture')}):
            _, identity = loader.load_native_scorer(self.root, manifest, device='cpu')
        self.assertEqual(calls, [dict(weights_path=str(self.root/'assets/weights.fixture'),
            local_model_dir=str(self.root/'assets/config'), device='cpu')])
        self.assertEqual(identity['runtime']['torch'], 'cpu-fixture')

    def test_no_implicit_gpu_selection(self):
        for value in (None, 'cuda', 'auto'):
            with self.assertRaises(ValueError):
                loader.load_native_scorer(self.root, {}, device=value)


class AbNatiVTests(TempCase):
    def raw(self, mode='ok'):
        output = self.root/'analysis'
        (output/'raw').mkdir(parents=True)
        (output/'metadata.json').write_text(json.dumps(dict(status='running', subprocess_returncode=0)))
        with (output/'raw/paired_abnativ_seq_scores.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['ID', 'input_seq_vh', 'input_seq_vl', nat.METRIC])
            writer.writeheader()
            for row in records():
                writer.writerow(dict(ID=row['id'], input_seq_vh='G'*70 if mode == 'wrong_chain' else row['heavy'],
                    input_seq_vl=row['light'], **{nat.METRIC: 0.8}))
        return output

    def test_external_evaluation_has_no_campaign_gate(self):
        output = self.raw()
        rows = records()
        for row in rows:
            row['model'] = 'AntiBERTy'
        results, meta = nat.finish_abnativ_pairs(rows, output)
        self.assertEqual([r['abnativ_paired_score'] for r in results], [0.8]*3)
        self.assertFalse(any('likelihood' in r or 'p5_eligibility' in r for r in results))

    def test_mutant_chain_mismatch_fails(self):
        with self.assertRaises(nat.MeasurementError):
            nat.finish_abnativ_pairs(records(), self.raw('wrong_chain'))

    def test_failed_process_not_completed(self):
        output = self.raw()
        (output/'metadata.json').write_text(json.dumps(dict(status='running', subprocess_returncode=1)))
        with self.assertRaises(nat.MeasurementError):
            nat.finish_abnativ_pairs(records(), output)

    def test_generic_analysis_retains_both_bounds(self):
        cal = nat.calibrate(list(range(101)), 'AbNatiV-fixture', .025, .975, 'reference')
        self.assertEqual((cal['lower_cutoff'], cal['upper_cutoff']), (2.5, 97.5))


if __name__ == '__main__':
    unittest.main()
