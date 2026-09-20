"""CPU contracts: native sampling, <=10 requests, unchanged sites and reversions."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from imut_cdr_gc.generation import JointMutationGenerator, sample_aa20, load_antigen_graph
from imut_cdr_gc.models.loading import verify_local_assets, sha256
from test_generation_models import TinyTokenizer, TinyEncoder, tiny_config, graph


class FakeProposalModel(nn.Module):
    def __init__(self, outputs=('A',)):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.tokenizer = TinyTokenizer()
        self.epi_config = SimpleNamespace(node_dim=3)
        self.model_max_len = 512
        self.outputs, self.calls, self.seen_masks = outputs, 0, []
    def forward(self, input_ids, attention_mask, masks, graphs):
        self.seen_masks.append(masks)
        logits = torch.full((*input_ids.shape, 23), -100.)
        aa = self.outputs[min(self.calls, len(self.outputs) - 1)]
        if aa == 'mixed':
            for p in range(input_ids.shape[1]):
                logits[:, p, self.tokenizer.convert_tokens_to_ids('A' if p % 2 else 'C')] = 100
        else:
            logits[:, :, self.tokenizer.convert_tokens_to_ids(aa)] = 100
        self.calls += 1
        return logits, None, None


def record(**changes):
    return dict(dict(id='case', fr_cdr_seq='A'*277, founder_fr_cdr_seq='A'*277,
        proposal_positions=list(range(26, 36)), design='design3', version='v3',
        antigen_id='explicit_graph_id', sampling_seed=93), **changes)


class GenerationProposalTests(unittest.TestCase):
    def test_ten_requested_sites_can_all_remain_unchanged(self):
        model = FakeProposalModel()
        result, = JointMutationGenerator(model, temperature=0).propose([record()], graphs=[graph()])
        self.assertEqual(result['requested_count'], 10)
        self.assertEqual(result['changed_count'], 0)
        self.assertEqual(result['cumulative_mutation_count'], 0)
        self.assertEqual(model.seen_masks, [[list(range(26, 36))]])
        self.assertTrue(result['full_chain_reconstruction_required'])
        self.assertFalse(result['screening_applied'])

    def test_extracted_model_and_proposal_adapter_run_together_on_cpu(self):
        from imut_cdr_gc.models.jm_epi import EpitopeConditionedFRCDRModel
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(249)
            model = EpitopeConditionedFRCDRModel(encoder=TinyEncoder(), tokenizer=TinyTokenizer(),
                                                config=tiny_config()).eval()
            result, = JointMutationGenerator(model, temperature=2.5, top_k=6,
                parental_residue_logit_penalty=3).propose([record()], graphs=[graph()])
        self.assertEqual(len(result['fr_cdr_seq']), 277)
        changed = [p for p, aa in enumerate(result['fr_cdr_seq']) if aa != 'A']
        self.assertEqual(changed, result['actual_mutation_positions'])
        self.assertTrue(set(changed) <= set(range(26, 36)))
        self.assertEqual(result['cumulative_mutation_count'], len(changed))

    def test_changed_count_not_requested_count(self):
        result, = JointMutationGenerator(FakeProposalModel(('mixed',)), temperature=0).propose(
            [record()], graphs=[graph()])
        self.assertEqual(result['changed_count'], 5)
        self.assertEqual(result['cumulative_mutation_count'], 5)
        self.assertEqual(len(result['iteration_trace'][0]['mutations']), 5)

    def test_iterations_recompute_actual_burden_including_reversions(self):
        result = JointMutationGenerator(FakeProposalModel(('C', 'A')), temperature=0).iterate(
            record(), graph=graph(), position_rounds=[list(range(26, 36)), list(range(26, 35))])
        self.assertEqual(result['total_requested_count'], 19)
        self.assertEqual(result['total_changed_count'], 19)
        self.assertEqual(result['cumulative_mutation_count'], 1)
        self.assertEqual(result['actual_mutation_positions'], [35])
        self.assertEqual([x['changed_count'] for x in result['iteration_trace']], [10, 9])

    def test_no_stale_parent_chains_or_scores_claimed_as_new_mutant(self):
        result, = JointMutationGenerator(FakeProposalModel(('C',)), temperature=0).propose(
            [record(heavy='PARENT', light='PARENT', structure_score=.9,
                    founder_heavy='ANCESTOR_H', founder_light='ANCESTOR_L')], graphs=[graph()])
        self.assertNotIn('heavy', result)
        self.assertNotIn('light', result)
        self.assertNotIn('structure_score', result)
        self.assertEqual(result['founder_heavy'], 'ANCESTOR_H')

    def test_invalid_requests_fail_before_forward(self):
        model = FakeProposalModel()
        generator = JointMutationGenerator(model)
        for positions in [list(range(26, 37)), [26, 26], [], [-1], [277], [0], [True]]:
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                generator.propose([record(proposal_positions=positions)], graphs=[graph()])
        self.assertEqual(model.calls, 0)
        sequence = list('A'*277)
        sequence[26] = 'X'
        with self.assertRaisesRegex(ValueError, 'padding'):
            generator.propose([record(fr_cdr_seq=''.join(sequence), founder_fr_cdr_seq=''.join(sequence))], graphs=[graph()])

    def test_no_silent_sequence_truncation_duplicate_ids_or_graph_mismatch(self):
        generator = JointMutationGenerator(FakeProposalModel())
        with self.assertRaisesRegex(ValueError, 'FR277'):
            generator.propose([record(fr_cdr_seq='A'*278)], graphs=[graph()])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            generator.propose([record(), record()], graphs=[graph(), graph()])
        with self.assertRaisesRegex(ValueError, 'One antigen'):
            generator.propose([record()], graphs=[])
        broken = graph()
        broken['x'] = torch.zeros(0, 3)
        with self.assertRaisesRegex(ValueError, 'empty'):
            generator.propose([record()], graphs=[broken])

    def test_native_aa20_sampling_order_exact_on_random_logits(self):
        ids = torch.arange(20)
        logits = torch.randn(23, generator=torch.Generator().manual_seed(451))
        for temperature, top_k in [(3., 10), (2.5, 6)]:
            a = torch.Generator(device='cpu').manual_seed(382)
            b = torch.Generator(device='cpu').manual_seed(382)
            observed = []
            expected = []
            aa = 'ACDEFGHIKLMNPQRSTVWY'
            for parent in aa:
                # Literal sampling operations of native run_v3_native_order.
                values = logits[ids].float().cpu() / temperature
                values[aa.index(parent)] -= 3.
                vals, inds = values.topk(top_k)
                expected.append(aa[int(inds[int(torch.multinomial(vals.softmax(-1), 1, generator=b))])])
                observed.append(sample_aa20(logits, parent=parent, aa_ids=ids,
                    temperature=temperature, top_k=top_k, penalty=3., generator=a))
            self.assertEqual(observed, expected)

    def test_missing_or_wrong_assets_fail_without_transformers_or_weight_loading(self):
        # Test files are project-local, synthetic text, not learned checkpoints.
        parent = Path(__file__).resolve().parents[1] / 'tests'
        with tempfile.TemporaryDirectory(prefix='.generation_asset_fixture_', dir=parent) as temp:
            folder = Path(temp)
            with self.assertRaises(FileNotFoundError):
                verify_local_assets(folder, folder/'absent.pt', '0'*64, {'config.json':'0'*64})
            (folder/'config.json').write_text('{}')
            (folder/'vocab.txt').write_text('A\n')
            (folder/'fake.pt').write_text('not learned weights')
            manifest = {name:sha256(folder/name) for name in ['config.json', 'vocab.txt']}
            digest = sha256(folder/'fake.pt')
            _, _, receipt = verify_local_assets(folder, folder/'fake.pt', digest, manifest)
            self.assertEqual(receipt['checkpoint_sha256'], digest)
            with self.assertRaisesRegex(ValueError, 'Checkpoint hash'):
                verify_local_assets(folder, folder/'fake.pt', '0'*64, manifest)
            (folder/'tokenizer_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'every present'):
                verify_local_assets(folder, folder/'fake.pt', digest, manifest)
            with self.assertRaisesRegex(ValueError, 'Pin config'):
                verify_local_assets(folder, folder/'fake.pt', digest, {'../config.json':'0'*64})

    def test_safe_graph_loader_is_identity_preserving_and_rejects_stale_hash(self):
        import numpy as np
        parent = Path(__file__).resolve().parents[1] / 'tests'
        with tempfile.TemporaryDirectory(prefix='.generation_graph_fixture_', dir=parent) as temp:
            path = Path(temp) / 'graph.npz'
            original = graph()
            arrays = {k:v.numpy() for k, v in original.items()}
            arrays['pos'] = np.zeros((3, 3), dtype=np.float32)
            np.savez(path, **arrays)
            loaded = load_antigen_graph(path, sha256(path), node_dim=3)
            for key, value in arrays.items():
                np.testing.assert_array_equal(loaded[key].numpy(), value)
            self.assertFalse(loaded['_source_identity']['pickle_loaded'])
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_antigen_graph(path, '0'*64, node_dim=3)

    def test_graph_loader_refuses_pickle_and_atom_schema_instead_of_guessing(self):
        import numpy as np
        parent = Path(__file__).resolve().parents[1] / 'tests'
        with tempfile.TemporaryDirectory(prefix='.generation_graph_fixture_', dir=parent) as temp:
            path = Path(temp) / 'graph.npz'
            np.savez(path, pos=np.zeros((1, 3), dtype=np.float32), res_name=np.array(['ALA']))
            with self.assertRaisesRegex(ValueError, 'not atom-level'):
                load_antigen_graph(path, sha256(path), node_dim=3)
            arrays = {k:v.numpy() for k, v in graph().items()}
            arrays['pos'] = np.zeros((3, 3), dtype=np.float32)
            arrays['x'] = np.array([['bad']], dtype=object)
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'Object arrays'):
                load_antigen_graph(path, sha256(path), node_dim=3)


if __name__ == '__main__':
    unittest.main()
