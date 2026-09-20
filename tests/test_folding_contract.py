"""Synthetic CPU safety/interface tests, not learned AB2 or GPU validation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from imut_cdr_gc import folding as f
from imut_cdr_gc.assets import AssetError, file_sha256
from imut_cdr_gc.scoring.common import InputError
from imut_cdr_gc.scoring.native_structure import verify_mutant_fold


def numbered(row):
    return {chain: [((i + 1, ' '), aa) for i, aa in enumerate(row[key])]
            for chain, key in (('H', 'heavy'), ('L', 'light'))}


class FakePrediction:
    ranking = [1, 2, 0]

    def __init__(self, row, numbering, *, missing_cb=False, wrong_number=False):
        self.row, self.numbering = row, numbering
        self.missing_cb, self.wrong_number = missing_cb, wrong_number
        self.saved = None

    def save_single_unrefined(self, filename, index=0):
        self.saved = index
        lines = []
        for chain in ('H', 'L'):
            for (number, insertion), aa in self.numbering[chain]:
                name = 'ALA' if aa == 'A' else 'GLY'
                names = ['N', 'CA', 'C', 'O'] + (['CB'] if aa == 'A' and not self.missing_cb else [])
                for atom in names:
                    n = number + 1 if self.wrong_number else number
                    lines.append(f'ATOM  {len(lines):5d} {" " + atom:<4} {name:3s} {chain}'
                        f'{n:4d}{insertion}   {float(number):8.3f}{float(index):8.3f}{0.:8.3f}'
                        f'{1.:6.2f}{0.:6.2f}          {atom[0]:>2}  ')
        Path(filename).write_text('\n'.join(lines), encoding='ascii')


class FakePredictor:
    runtime = {'fixture': 'synthetic CPU; no learned model', 'device': 'cpu'}

    def __init__(self, *, error_on=None, error=None, mutate=None, prediction_options=None):
        self.calls, self.predictions = [], []
        self.error_on, self.error = error_on, error
        self.mutate, self.options = mutate, prediction_options or {}

    def predict(self, row, numbers):
        self.calls.append(row['id'])
        if row['id'] == self.error_on:
            raise self.error
        if self.mutate:
            self.mutate(row)
        prediction = FakePrediction(row, numbers, **self.options)
        self.predictions.append(prediction)
        return prediction


class FoldingContractTests(unittest.TestCase):
    def setUp(self):
        scratch = PACKAGE / '.cache/tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='fold_contract_', dir=scratch)
        self.root = Path(self.tmp.name)
        self.rows = [dict(id='fixture_' + str(i), heavy='A' * 75, light='G' * 72,
            antigen_id='synthetic_target', model='iMut-CDR-JM-Epi', version='v3', design='synthetic',
            mutation_trace=[{'requested': 10, 'changed': 4}], founder={'id': 'synthetic_founder'})
            for i in (1, 2)]
        self.pin = self.root / 'source_identity.txt'
        self.pin.write_text('synthetic pinned asset, not native weights')
        self.bundle = dict(pins={self.pin: file_sha256(self.pin)},
            receipt={'fixture': 'synthetic CPU runtime, not native assets'})
        self.predictor = FakePredictor()

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, *, output='folds', rows=None, assets=None, device='cpu'):
        return f.fold_antibodies(self.rows if rows is None else rows,
            project_root=self.root, output_dir=output, assets=assets or {}, device=device)

    def numberer(self, *args):
        return [numbered(row) for row in args[0]]

    def positive(self, *, output='folds', rows=None):
        with patch.object(f, '_verify_assets', return_value=self.bundle), \
             patch.object(f, '_number_all', side_effect=self.numberer), \
             patch.object(f, '_load_predictor', return_value=self.predictor):
            return self.invoke(output=output, rows=rows)

    def test_true_fold_contract_preserves_metadata_and_native_ranking(self):
        before = deepcopy(self.rows)
        rows, receipt = self.positive()
        self.assertEqual(self.rows, before)
        self.assertEqual([r['id'] for r in rows], [r['id'] for r in before])
        for original, result, prediction in zip(before, rows, self.predictor.predictions):
            for name, value in original.items():
                self.assertEqual(result[name], value)
            provenance = result['fold_provenance']
            self.assertEqual(provenance['saved_model_id'], 2)
            self.assertEqual(provenance['model_ranking'], [2, 3, 1])
            self.assertEqual(prediction.saved, 1)
            self.assertEqual(provenance['residue_counts'], {'H': 75, 'L': 72})
            self.assertEqual(provenance['antigen_id'], original['antigen_id'])
            path, identity = verify_mutant_fold(result, self.root)
            self.assertTrue(identity['exact_full_heavy_light_sequence_match'])
            self.assertEqual(file_sha256(path), result['pdb_sha256'])
        self.assertEqual(receipt['status'], 'complete_antibody_folds_not_complexes')
        self.assertEqual(receipt['folded'], 2)
        self.assertEqual(receipt['complex_pdbs_created'], 0)
        self.assertFalse(receipt['scientific_selection_performed'])
        self.assertFalse(receipt['learned_weight_predictor_parity_verified'])
        self.assertEqual(receipt['parameters']['model_ids'], [1, 2, 3])
        self.assertEqual(file_sha256(self.root / 'folds/fold_manifest.jsonl'),
                         receipt['output_manifest']['sha256'])
        self.assertFalse((self.root / 'folds/failure.json').exists())

    def test_empty_duplicate_short_or_missing_target_fail_before_model(self):
        for label, rows in [('empty', []), ('duplicate', [self.rows[0]] * 2),
                ('short', [dict(self.rows[0], heavy='A' * 70)]),
                ('target', [dict(self.rows[0], antigen_id='')]),
                ('fragment', [dict(self.rows[0], full_chain_reconstruction_required=True)])]:
            with self.subTest(label=label), patch.object(f, '_load_predictor') as loader:
                with self.assertRaises(InputError):
                    self.invoke(output=label, rows=rows)
                loader.assert_not_called()
                self.assertFalse((self.root / label).exists())

    def test_existing_fold_evidence_is_never_silently_replaced(self):
        for key in f.FOLD_FIELDS:
            with self.subTest(key=key), self.assertRaisesRegex(InputError, 'Existing fold'):
                self.invoke(rows=[dict(self.rows[0], **{key: None})])

    def test_implicit_device_or_invalid_ids_rejected(self):
        for device in ('cuda', 'auto', 'cuda:-1', 0):
            with self.subTest(device=device), self.assertRaises(InputError):
                self.invoke(device=device)
        for identifier in (None, '', True):
            with self.subTest(identifier=identifier), self.assertRaises(InputError):
                self.invoke(rows=[dict(self.rows[0], id=identifier)])

    def test_missing_assets_fail_before_numbering_or_output(self):
        with patch.object(f, '_number_all') as numberer, patch.object(f, '_load_predictor') as loader:
            with self.assertRaises(AssetError):
                self.invoke()
        numberer.assert_not_called()
        loader.assert_not_called()
        self.assertFalse((self.root / 'folds').exists())

    def test_native_hash_missing_file_and_escaping_asset_refused(self):
        spec = {'one': {'path': 'source_identity.txt', 'sha256': '0' * 64}}
        for digest, path in [('1' * 64, 'source_identity.txt'),
                             ('0' * 64, 'missing.bin'), ('0' * 64, '../outside.bin')]:
            spec['one'] = {'path': path, 'sha256': digest}
            with self.subTest(path=path), self.assertRaises(AssetError):
                f._verify_group(self.root, spec, {'one': '0' * 64}, 'fixture')

    def test_changed_weight_identity_rejected_without_reading_weight(self):
        specs = {name: {'path': name, 'sha256': '0' * 64} for name in f.WEIGHT_SHA256}
        with self.assertRaisesRegex(AssetError, 'frozen native identity'):
            f._verify_group(self.root, specs, f.WEIGHT_SHA256, 'AB2 weight')

    def test_output_cannot_overwrite_previous_attempt(self):
        rows, receipt = self.positive()
        prior = file_sha256(self.root / 'folds/receipt.json')
        with patch.object(f, '_verify_assets', return_value=self.bundle), \
             patch.object(f, '_number_all') as numberer, self.assertRaises(FileExistsError):
            self.invoke()
        numberer.assert_not_called()
        self.assertEqual(prior, file_sha256(self.root / 'folds/receipt.json'))

    def test_numbering_failure_precedes_learned_loading(self):
        with patch.object(f, '_verify_assets', return_value=self.bundle), \
             patch.object(f, '_number_all', side_effect=InputError('two domains')), \
             patch.object(f, '_load_predictor') as loader:
            with self.assertRaisesRegex(InputError, 'two domains'):
                self.invoke()
        loader.assert_not_called()
        self.assertEqual(json.loads((self.root / 'folds/failure.json').read_text())['completed_before_failure'], 0)
        self.assertFalse(list((self.root / 'folds').glob('*.pdb')))

    def test_all_numbered_sequences_checked_before_any_prediction(self):
        result = [numbered(r) for r in self.rows]
        result[1]['H'].pop()
        with patch.object(f, '_verify_assets', return_value=self.bundle), \
             patch.object(f, '_number_all', return_value=result), \
             patch.object(f, '_load_predictor') as loader:
            with self.assertRaisesRegex(InputError, 'full input chain'):
                self.invoke()
        loader.assert_not_called()

    def test_inference_and_resource_failures_preserve_partial_not_complete(self):
        for label, error in [('inference', RuntimeError('failed model')), ('memory', MemoryError('OOM')),
                             ('interrupt', KeyboardInterrupt())]:
            self.predictor = FakePredictor(error_on='fixture_2', error=error)
            with self.subTest(label=label), self.assertRaises(type(error)):
                self.positive(output=label)
            failure = json.loads((self.root / label / 'failure.json').read_text())
            self.assertEqual(failure['completed_before_failure'], 1)
            self.assertEqual(failure['failed_candidate'], 'fixture_2')
            self.assertEqual(failure['resource_error'], label == 'memory')
            self.assertTrue((self.root / label / 'antibody_000001.pdb').exists())
            self.assertTrue((self.root / label / 'partial.jsonl').exists())
            self.assertFalse((self.root / label / 'fold_manifest.jsonl').exists())
            self.assertFalse((self.root / label / 'receipt.json').exists())

    def test_hash_drift_during_prediction_prevents_completion(self):
        self.predictor = FakePredictor(mutate=lambda row: self.pin.write_text('changed asset'))
        with self.assertRaisesRegex(AssetError, 'changed during scoring'):
            self.positive()
        self.assertFalse((self.root / 'folds/receipt.json').exists())

    def test_input_metadata_mutation_prevents_completion(self):
        self.predictor = FakePredictor(mutate=lambda row: row.update(antigen_id='wrong'))
        with self.assertRaisesRegex(InputError, 'metadata changed'):
            self.positive()
        self.assertFalse((self.root / 'folds/receipt.json').exists())

    def test_missing_sidechains_and_wrong_residue_mapping_fail(self):
        for label, options in [('sidechains', {'missing_cb': True}), ('numbers', {'wrong_number': True})]:
            self.predictor = FakePredictor(prediction_options=options)
            with self.subTest(label=label), self.assertRaises(InputError):
                self.positive(output=label)
            self.assertFalse((self.root / label / 'receipt.json').exists())

    def test_numberer_recognition_is_full_length_single_domain_and_typed(self):
        calls = []
        def native_call(records, **kwargs):
            calls.append(kwargs)
            seq = records[0][1]
            entries = [((i + 1, ' '), aa) for i, aa in enumerate(seq)]
            return [[(entries, 0, len(seq) - 1)]], [[{'chain_type': 'K'}]], []
        result = f._number_chain('A' * 75, 'L', native_call)
        self.assertEqual(''.join(aa for _, aa in result), 'A' * 75)
        self.assertEqual([c['scheme'] for c in calls], ['imgt', 'chothia'])
        self.assertEqual(calls[0]['allow'], {'K', 'L'})
        self.assertEqual(calls[0]['allowed_species'], ['human', 'mouse'])
        self.assertFalse(calls[0]['output'])

    def test_numberer_no_domain_multidomain_wrong_role_or_truncation_rejected(self):
        seq = 'A' * 75
        entries = [((i + 1, ' '), aa) for i, aa in enumerate(seq)]
        cases = [([[]], [[{'chain_type': 'H'}]]),
                 ([[(entries, 0, 74)] * 2], [[{'chain_type': 'H'}]]),
                 ([[(entries, 0, 74)]], [[{'chain_type': 'K'}]]),
                 ([[(entries[1:], 1, 74)]], [[{'chain_type': 'H'}]]),
                 ([[(entries, 0, 74)]], [])]
        for nums, details in cases:
            with self.subTest(nums=len(nums[0])), self.assertRaises(InputError):
                f._number_chain(seq, 'H', lambda *a, **kw: (nums, details, []))

    def test_duplicate_number_keys_and_invalid_insertion_rejected(self):
        row = self.rows[0]
        for defect in ('duplicate', 'insertion'):
            nums = numbered(row)
            nums['H'][1] = ((1, ' ') if defect == 'duplicate' else (2, 'AA'), 'A')
            with self.subTest(defect=defect), self.assertRaises(InputError):
                f._validate_numbered(row, nums)

    def test_native_namespace_excludes_constructor_download_refinement_and_imports(self):
        # Synthetic source fixture tests AST isolation, NOT native numerical parity.
        texts = {'constants.py': 'fixture_value = 7\n',
            'util.py': 'import forbidden_fixture_dependency\n'
                'def get_one_hot(): return 1\n'
                'def get_encoding(): return 2\n'
                'def find_alignment_transform(): return 3\n'
                'def to_pdb(): return 4\n'
                'def download_file(): raise RuntimeError("download forbidden")\n'
                'raise RuntimeError("unsafe top-level")\n',
            'ABodyBuilder2.py': 'import forbidden_fixture_dependency\n'
                'class Antibody:\n'
                ' def __init__(self): self.ready = True\n'
                ' def save_single_unrefined(self): return 5\n'
                ' def save(self): raise RuntimeError("refine forbidden")\n'
                'class ABodyBuilder2:\n'
                ' def __init__(self): raise RuntimeError("autodownload forbidden")\n'
                'raise RuntimeError("unsafe top-level")\n'}
        paths, hashes = {}, {}
        for name, text in texts.items():
            paths[name] = self.root / name
            paths[name].write_text(text)
            hashes[name] = hashlib.sha256(text.encode()).hexdigest()
        before = dict(os_environ_snapshot())
        modules_before = {n for n in sys.modules if n.startswith('ImmuneBuilder')}
        with patch.object(f, 'SOURCE_SHA256', hashes):
            namespace = f._native_namespace(paths, include_models=False)
        self.assertTrue(namespace['Antibody']().ready)
        self.assertFalse(hasattr(namespace['Antibody'], 'save'))
        self.assertNotIn('ABodyBuilder2', namespace)
        self.assertNotIn('download_file', namespace)
        self.assertEqual(namespace['to_pdb'](), 4)
        self.assertEqual(before, dict(os_environ_snapshot()))
        self.assertEqual(modules_before, {n for n in sys.modules if n.startswith('ImmuneBuilder')})

    def test_source_rehashed_before_ast_compile(self):
        with self.assertRaisesRegex(AssetError, 'changed before compilation'):
            f._tree(self.pin, '0' * 64)

    def test_numbering_child_uses_local_tmp_and_no_runtime_mutation(self):
        captured = {}
        before = dict(os_environ_snapshot())
        def fake_run(command, **kwargs):
            captured.update(kwargs)
            request = json.loads(Path(command[-2]).read_text())
            Path(command[-1]).write_text(json.dumps([{'id': r['id'], 'numbered': numbered(r)}
                                                    for r in request['records']]))
            return type('Run', (), {'returncode': 0, 'stderr': '', 'stdout': ''})()
        output = self.root / 'numbering_attempt'
        output.mkdir()
        with patch.object(f.subprocess, 'run', side_effect=fake_run):
            result = f._number_all(self.rows, self.root, {}, output)
        self.assertEqual(len(result), 2)
        self.assertEqual(captured['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertTrue(Path(captured['env']['TMPDIR']).is_relative_to(output))
        self.assertFalse(list(output.iterdir()))
        self.assertEqual(before, dict(os_environ_snapshot()))

    def test_numbering_child_failure_cleans_only_owned_temporary_directory(self):
        output = self.root / 'numbering_failure'
        output.mkdir()
        untouched = output / 'keep.txt'
        untouched.write_text('existing evidence')
        run = type('Run', (), {'returncode': 1, 'stderr': 'numbering failed', 'stdout': ''})()
        with patch.object(f.subprocess, 'run', return_value=run), self.assertRaises(InputError):
            f._number_all(self.rows, self.root, {}, output)
        self.assertEqual(list(output.iterdir()), [untouched])


def os_environ_snapshot():
    import os
    return tuple(os.environ.items())


if __name__ == '__main__':
    unittest.main()
