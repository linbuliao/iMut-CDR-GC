"""Synthetic CPU CLI wiring, not real-checkpoint/GPU or scientific validation.

Only the learned-model loader is mocked. The real proposal adapter, safe NPZ
reader, CLI batching, founder preparation and full-chain reconstruction run.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from imut_cdr_gc.cli import main
from imut_cdr_gc.generation import JointMutationGenerator
from imut_cdr_gc.records import read_jsonl, sha256
from test_generation_proposals import FakeProposalModel


class GenerationCLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        scratch = PACKAGE / '.cache' / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='generation_cli_', dir=scratch)
        self.root = Path(self.temporary.name)
        fr = ['X'] * 277
        for start, motif in ((26, 'ACD'), (55, 'EFG'), (102, 'HIK'),
                             (165, 'LMN'), (194, 'PQR'), (241, 'STV')):
            fr[start:start + 3] = motif
        self.founder = dict(id='synthetic_founder', design='synthetic',
            antigen_id='synthetic_target', heavy='VVWLMNYPQRWSTVYWW',
            light='RRWACDYEFGWHIKYGG', fr_cdr_seq=''.join(fr))
        self.put_json('founder_input.json', self.founder)
        self.invoke('prepare', '--founder', 'founder_input.json', '--output-dir', 'prepared')
        self.rows = [dict(id=identifier, fr_cdr_seq=''.join(fr), founder_fr_cdr_seq=''.join(fr),
            proposal_positions=positions, sampling_seed=31, design='synthetic',
            antigen_id='synthetic_target', version='v3', heavy=self.founder['heavy'],
            light=self.founder['light'], scores={'esm2': {'score': .99}},
            likelihood={'score': -.01}, selected=True, pdb_path='not_a_new_structure.pdb')
            for identifier, positions in [('proposal_z', [26, 27, 28, 55, 56, 57, 102, 103, 104, 165]),
                                           ('proposal_a', [166]), ('proposal_m', [26])]]
        self.put_rows()
        (self.root / 'assets' / 'model').mkdir(parents=True)
        (self.root / 'assets' / 'model' / 'config.json').write_text('{}')
        (self.root / 'assets' / 'model' / 'vocab.txt').write_text('A\n')
        # This is never torch.loaded: only the loader return is mocked.
        (self.root / 'assets' / 'mock.pt').write_text('synthetic fixture, not learned weights')
        np.savez(self.root / 'graph.npz', x=np.zeros((3, 30), dtype=np.float32),
            pos=np.zeros((3, 3), dtype=np.float32),
            edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
            edge_attr=np.array([[1., 1.], [2., 0.]], dtype=np.float32))
        self.cfg = dict(generator=dict(model_dir='assets/model',
            checkpoint=dict(path='assets/mock.pt', sha256=sha256(self.root / 'assets/mock.pt')),
            asset_sha256={name: sha256(self.root / 'assets/model' / name)
                          for name in ('config.json', 'vocab.txt')}),
            antigen_graph=dict(path='graph.npz', sha256=sha256(self.root / 'graph.npz'),
                               antigen_id='synthetic_target'),
            batch_size=2, sampling=dict(device='cpu', precision='float32', temperature=0,
                                       top_k=0, parental_residue_logit_penalty=0, seed=31))
        self.put_json('config.json', self.cfg)
        self.model = FakeProposalModel(('A',))
        self.model.epi_config = SimpleNamespace(node_dim=30)
        self.generator = JointMutationGenerator(self.model, temperature=0,
            provenance={'fixture': 'mock learned loader; real CPU adapter',
                        'native_real_checkpoint_parity_verified': False})

    def tearDown(self):
        self.temporary.cleanup()

    def put_json(self, relative, value):
        (self.root / relative).write_text(json.dumps(value), encoding='utf-8')

    def put_rows(self):
        (self.root / 'proposals.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in self.rows), encoding='utf-8')

    def invoke(self, *arguments):
        out = io.StringIO()
        with redirect_stdout(out):
            result = main(['--project-root', str(self.root), *arguments])
        self.assertEqual(result, 0)
        return json.loads(out.getvalue())

    def generate(self, output='generated'):
        return self.invoke('generate', '--input', 'proposals.jsonl', '--founder',
            'prepared/founder.json', '--config', 'config.json', '--version', 'v3',
            '--output-dir', output)

    def loader(self):
        return patch('imut_cdr_gc.generation.load_local_generator', return_value=self.generator)

    def test_prepare_generate_chain_is_lossless_ordered_and_not_screened(self):
        with self.loader() as mocked:
            receipt = self.generate()
        rows = read_jsonl(self.root / 'generated/candidates.jsonl')
        self.assertEqual([r['id'] for r in rows], [r['id'] for r in self.rows])
        self.assertEqual([len(m) for m in self.model.seen_masks], [2, 1])
        self.assertEqual(rows[0]['light'], 'RRWAAAYAAAWAAAYGG')
        self.assertEqual(rows[0]['heavy'], 'VVWAMNYPQRWSTVYWW')
        self.assertEqual(rows[1]['heavy'], 'VVWLANYPQRWSTVYWW')
        self.assertEqual(rows[1]['light'], self.founder['light'])
        self.assertEqual(rows[2]['heavy'], self.founder['heavy'])
        self.assertEqual(rows[2]['light'], self.founder['light'])
        self.assertEqual([r['requested_count'] for r in rows], [10, 1, 1])
        self.assertEqual([r['mutation_count'] for r in rows], [9, 1, 0])
        for row in rows:
            self.assertFalse(row['full_chain_reconstruction_required'])
            self.assertTrue(row['full_chain_reconstruction_verified'])
            self.assertEqual(row['representation'], 'actual_full_mutant_chains')
            self.assertFalse(row['screening_applied'])
            self.assertEqual(row['model'], 'iMut-CDR-JM-Epi')
            self.assertEqual(row['founder']['heavy'], self.founder['heavy'])
            self.assertEqual(row['founder']['light'], self.founder['light'])
            self.assertEqual(row['mutation_count'], row['cumulative_mutation_count'])
            for old_field in ('scores', 'likelihood', 'selected', 'pdb_path', 'stepwise_trace'):
                self.assertNotIn(old_field, row)
        self.assertEqual(receipt['status'], 'generated_not_screened')
        self.assertEqual(receipt['count'], 3)
        self.assertEqual((receipt['selected'], receipt['pdb_generated']), (0, 0))
        self.assertFalse(receipt['v3_per_step_selection_executed'])
        self.assertFalse(receipt['generator_identity']['native_real_checkpoint_parity_verified'])
        self.assertEqual(receipt['input_sha256'], sha256(self.root / 'proposals.jsonl'))
        self.assertEqual(receipt['founder_sha256'], sha256(self.root / 'prepared/founder.json'))
        self.assertEqual(receipt['config_sha256'], sha256(self.root / 'config.json'))
        self.assertEqual(receipt['output_sha256'], sha256(self.root / 'generated/candidates.jsonl'))
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs['model_dir'].resolve(), self.root / 'assets/model')
        self.assertEqual(kwargs['checkpoint'].resolve(), self.root / 'assets/mock.pt')
        for key, value in self.cfg['sampling'].items():
            self.assertEqual(kwargs[key], value)

    def test_returned_identity_order_is_not_silently_repaired(self):
        real = self.generator.propose
        with self.loader(), patch.object(self.generator, 'propose',
                side_effect=lambda *a, **k: list(reversed(real(*a, **k)))):
            with self.assertRaisesRegex(ValueError, 'ordered proposal identities'):
                self.generate()
        self.assertFalse((self.root / 'generated/candidates.jsonl').exists())
        self.assertEqual(json.loads((self.root / 'generated/failure.json').read_text())['status'], 'failed')

    def test_wrong_founder_and_duplicate_ids_fail_before_loading(self):
        original = deepcopy(self.rows)
        for defect in ('founder', 'duplicate'):
            self.rows = deepcopy(original)
            if defect == 'founder':
                self.rows[0]['founder_fr_cdr_seq'] = 'A' * 277
            else:
                self.rows[1]['id'] = self.rows[0]['id']
            self.put_rows()
            with self.subTest(defect=defect), self.loader() as mocked, self.assertRaises(ValueError):
                self.generate(defect)
            mocked.assert_not_called()
            self.assertFalse((self.root / defect).exists())

    def test_wrong_explicit_target_design_or_version_fails_before_loading(self):
        original = deepcopy(self.rows)
        for field, value in (('antigen_id', 'other_target'), ('design', 'other_design'), ('version', 'v1')):
            self.rows = deepcopy(original)
            self.rows[0][field] = value
            self.put_rows()
            with self.subTest(field=field), self.loader() as mocked, self.assertRaises(ValueError):
                self.generate(field)
            mocked.assert_not_called()

    def test_invalid_founder_mapping_is_not_deferred_until_after_inference(self):
        f = json.loads((self.root / 'prepared/founder.json').read_text())
        f['cdr_position_map']['26']['chain'] = 'heavy'
        self.put_json('prepared/founder.json', f)
        with self.loader() as mocked, self.assertRaises(ValueError):
            self.generate()
        mocked.assert_not_called()
        self.assertEqual(self.model.calls, 0)

    def test_graph_target_and_source_hash_fail_before_model_loading(self):
        original = deepcopy(self.cfg)
        for field, value in (('antigen_id', 'other_target'), ('sha256', '0' * 64)):
            self.cfg = deepcopy(original)
            self.cfg['antigen_graph'][field] = value
            self.put_json('config.json', self.cfg)
            with self.subTest(field=field), self.loader() as mocked, self.assertRaises(ValueError):
                self.generate(field)
            mocked.assert_not_called()

    def test_model_directory_cannot_escape_explicit_project(self):
        (self.root / 'outside_link').symlink_to(PACKAGE, target_is_directory=True)
        for value in ('../model', str(PACKAGE), 'outside_link'):
            self.cfg['generator']['model_dir'] = value
            self.put_json('config.json', self.cfg)
            with self.subTest(path=value), self.loader() as mocked, self.assertRaises(ValueError):
                self.generate()
            mocked.assert_not_called()

    def test_request_limit_rejection_writes_no_candidate_and_runs_no_forward(self):
        self.rows[0]['proposal_positions'] = [26, 27, 28, 55, 56, 57, 102, 103, 104, 165, 166]
        self.put_rows()
        with self.loader(), self.assertRaisesRegex(ValueError, '1 to 10'):
            self.generate()
        self.assertEqual(self.model.calls, 0)
        self.assertFalse((self.root / 'generated/candidates.jsonl').exists())

    def test_changed_input_during_inference_is_not_bound_as_success(self):
        real = self.generator.propose
        for index, relative in enumerate(('proposals.jsonl', 'prepared/founder.json', 'config.json')):
            path = self.root / relative
            original = path.read_bytes()
            def mutate_after_read(*args, **kwargs):
                # Whitespace leaves JSON meaning unchanged but changes identity.
                path.write_bytes(original + b'\n')
                return real(*args, **kwargs)
            output = 'changed_input_' + str(index)
            try:
                with self.subTest(path=relative), self.loader(), patch.object(self.generator, 'propose',
                        side_effect=mutate_after_read), self.assertRaises(ValueError):
                    self.generate(output)
                self.assertFalse((self.root / output / 'generation.json').exists())
                self.assertFalse((self.root / output / 'candidates.jsonl').exists())
            finally:
                path.write_bytes(original)

    def test_existing_output_is_never_overwritten(self):
        output = self.root / 'existing'
        output.mkdir()
        (output / 'sentinel').write_text('user-owned')
        with self.loader(), self.assertRaises(FileExistsError):
            self.generate('existing')
        self.assertEqual((output / 'sentinel').read_text(), 'user-owned')
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['sentinel'])


if __name__ == '__main__':
    unittest.main()
