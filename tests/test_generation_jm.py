"""Unconditioned JM routing and strict local loading; synthetic CPU assets only."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from transformers import EsmConfig, EsmModel, EsmTokenizer

from imut_cdr_gc.generation import JointMutationGenerator, load_local_generator
from imut_cdr_gc.models import BackboneConfig, EpiConfig
from imut_cdr_gc.models.fr_cdr import ProteinMLMContrastModel
from imut_cdr_gc.models.jm_epi import EpitopeConditionedFRCDRModel
from imut_cdr_gc.models.loading import (
    REFERENCE_JM_EPI_SHA256, TOKENIZER_FILES, load_local_model,
    load_state_dict_strict, sha256,
)
from test_generation_models import TinyEncoder, TinyTokenizer, graph, native_namespace, tiny_config
from test_generation_proposals import FakeProposalModel, record

torch.set_num_threads(1)
PROJECT = Path(__file__).resolve().parents[1]


class SequenceOnlyFixture(nn.Module):
    """A three-argument forward; an accidental fourth graph argument fails."""
    def __init__(self, outputs=('C',)):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.tokenizer = TinyTokenizer()
        self.model_max_len = 512
        self.outputs, self.calls, self.seen_masks = outputs, 0, []

    def forward(self, input_ids, attention_mask, masks):
        self.seen_masks.append(copy.deepcopy(masks))
        aa = self.outputs[min(self.calls, len(self.outputs) - 1)]
        logits = torch.full((*input_ids.shape, len(self.tokenizer)), -100.)
        logits[:, :, self.tokenizer.convert_tokens_to_ids(aa)] = 100.
        self.calls += 1
        return logits, None, None


class JMProposalTests(unittest.TestCase):
    def test_explicit_sequence_only_forward_and_identity(self):
        model = SequenceOnlyFixture()
        generator = JointMutationGenerator(model, model_kind='jm', temperature=0)
        output, = generator.propose([record()])
        self.assertEqual(output['model'], 'iMut-CDR-JM')
        self.assertFalse(output['antigen_conditioned'])
        self.assertFalse(output['screening_applied'])
        self.assertEqual(output['changed_sites'], list(range(26, 36)))
        self.assertEqual(model.seen_masks, [[list(range(26, 36))]])
        self.assertEqual(generator.provenance['model_kind'], 'jm')
        self.assertFalse(generator.provenance['antigen_conditioned'])

    def test_graphs_are_rejected_not_silently_ignored(self):
        model = SequenceOnlyFixture()
        generator = JointMutationGenerator(model, model_kind='jm')
        for graphs in ([], [None], [graph()]):
            with self.subTest(graphs_type=str(type(graphs))), self.assertRaisesRegex(ValueError, 'does not accept'):
                generator.propose([record()], graphs=graphs)
        with self.assertRaisesRegex(ValueError, 'does not accept'):
            generator.iterate(record(), graph=graph(), position_rounds=[[26]])
        self.assertEqual(model.calls, 0)

    def test_model_architecture_and_provenance_cannot_be_relabeled(self):
        cases = [(FakeProposalModel(), 'jm'), (SequenceOnlyFixture(), 'jm-epi')]
        for model, kind in cases:
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'architecture'):
                JointMutationGenerator(model, model_kind=kind)
        for identity in ({'model_kind': 'jm-epi'}, {'model': 'iMut-CDR-JM-Epi'},
                         {'antigen_conditioned': True}):
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, 'identity'):
                JointMutationGenerator(SequenceOnlyFixture(), model_kind='jm', provenance=identity)
        with self.assertRaisesRegex(ValueError, 'model_kind'):
            JointMutationGenerator(SequenceOnlyFixture(), model_kind='unreviewed')

    def test_unchanged_and_reversions_keep_actual_burden(self):
        unchanged, = JointMutationGenerator(SequenceOnlyFixture(('A',)), model_kind='jm',
                                           temperature=0).propose([record()])
        self.assertEqual(unchanged['requested_count'], 10)
        self.assertEqual(unchanged['changed_count'], 0)
        output = JointMutationGenerator(SequenceOnlyFixture(('C', 'A')), model_kind='jm',
                                        temperature=0).iterate(record(),
            position_rounds=[list(range(26, 36)), list(range(26, 35))])
        self.assertEqual(output['model'], 'iMut-CDR-JM')
        self.assertEqual(output['total_changed_count'], 19)
        self.assertEqual(output['actual_mutation_positions'], [35])
        self.assertEqual(output['cumulative_mutation_count'], 1)

    def test_invalid_biological_sites_fail_before_jm_forward(self):
        model = SequenceOnlyFixture()
        generator = JointMutationGenerator(model, model_kind='jm')
        for positions in ([], [0], [277], [26, 26], list(range(26, 37))):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                generator.propose([record(proposal_positions=positions)])
        sequence = 'A'*26 + 'X' + 'A'*250
        with self.assertRaisesRegex(ValueError, 'padding'):
            generator.propose([record(fr_cdr_seq=sequence, founder_fr_cdr_seq=sequence)])
        self.assertEqual(model.calls, 0)

    def test_existing_epi_default_still_requires_and_uses_graph(self):
        model = FakeProposalModel()
        generator = JointMutationGenerator(model, temperature=0)
        with self.assertRaisesRegex(ValueError, 'One antigen'):
            generator.propose([record()])
        output, = generator.propose([record()], graphs=[graph()])
        self.assertEqual(output['model'], 'iMut-CDR-JM-Epi')
        self.assertTrue(output['antigen_conditioned'])
        self.assertEqual(model.calls, 1)

    def test_native_jm_state_and_all_outputs_are_identical_without_epi_modules(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1843)
            config, tokenizer = tiny_config(), TinyTokenizer()
            encoder = TinyEncoder()
            original = native_namespace(copy.deepcopy(encoder), tokenizer, config)['ProteinMLMContrastModel']().eval()
            successor = ProteinMLMContrastModel(encoder=copy.deepcopy(encoder), tokenizer=tokenizer,
                                                config=config.backbone).eval()
            receipt = load_state_dict_strict(successor, copy.deepcopy(original.state_dict()))
            self.assertTrue(receipt['strict'])
            self.assertFalse(any(k.startswith(('pocket.', 'cross.', 'film_mlp.'))
                                 for k in successor.state_dict()))
            tokens = tokenizer(['A'*277, 'C'*277], add_special_tokens=False)
            masks = [list(range(26, 36)), [55, 56, 102]]
            with torch.inference_mode():
                expected = original(tokens.input_ids, tokens.attention_mask, masks)
                actual = successor(tokens.input_ids, tokens.attention_mask, masks)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)


class JMLocalAssetTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='.jm_asset_fixture_', dir=PROJECT/'tests')
        self.addCleanup(self.scratch.cleanup)
        self.folder = Path(self.scratch.name)
        vocab = ['<cls>', '<pad>', '<eos>', '<unk>', *list('ACDEFGHIKLMNPQRSTVWY'), 'X', '<mask>']
        (self.folder/'vocab.txt').write_text('\n'.join(vocab) + '\n', encoding='utf-8')
        self.tokenizer = EsmTokenizer(vocab_file=str(self.folder/'vocab.txt'))
        self.tokenizer.save_pretrained(self.folder)
        cfg = EsmConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, max_position_embeddings=512,
            pad_token_id=1, mask_token_id=len(vocab)-1, position_embedding_type='rotary',
            token_dropout=False, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0)
        cfg._attn_implementation = 'eager'
        cfg.save_pretrained(self.folder)
        self.config = tiny_config().backbone
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(28491)
            self.model = ProteinMLMContrastModel(encoder=EsmModel(cfg), tokenizer=self.tokenizer,
                                                 config=self.config).eval()
        self.weights = self.folder/'synthetic.pt'
        torch.save(self.model.state_dict(), self.weights)
        self.manifest = {p.name: sha256(p) for p in self.folder.iterdir() if p.name in TOKENIZER_FILES}

    def arguments(self):
        return dict(model_dir=self.folder, checkpoint=self.weights,
            expected_checkpoint_sha256=sha256(self.weights), asset_sha256=self.manifest,
            model_config=self.config.as_dict(), model_kind='jm')

    def test_saved_tiny_HF_jm_reloads_strictly_and_runs_three_argument_forward(self):
        observed, receipt = load_local_model(**self.arguments())
        self.assertIsInstance(observed, ProteinMLMContrastModel)
        self.assertNotIsInstance(observed, EpitopeConditionedFRCDRModel)
        self.assertFalse(hasattr(observed, 'epi_config'))
        self.assertEqual(receipt['loaded_keys'], len(self.model.state_dict()))
        self.assertTrue(receipt['strict'])
        self.assertFalse(receipt['antigen_conditioned'])
        self.assertFalse(receipt['state_contract']['checkpoint_specific_compatibility'])
        self.assertEqual(receipt['model'], 'iMut-CDR-JM')
        tokens = self.tokenizer(['A'*277], return_tensors='pt', add_special_tokens=False)
        with torch.inference_mode():
            before = self.model(tokens.input_ids, tokens.attention_mask, [[26, 27]])
            after = observed(tokens.input_ids, tokens.attention_mask, [[26, 27]])
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_public_generator_loader_forwards_jm_and_has_no_antigen_dependency(self):
        generator = load_local_generator(**self.arguments(), temperature=0)
        output, = generator.propose([record()])
        self.assertEqual(output['model'], 'iMut-CDR-JM')
        self.assertEqual(generator.provenance['model_kind'], 'jm')
        self.assertTrue(generator.provenance['strict'])

    def test_epitope_keys_are_not_dropped_to_make_a_jm_state_fit(self):
        state = self.model.state_dict()
        state['film_mlp.0.weight'] = torch.ones(1)
        torch.save(state, self.weights)
        with self.assertRaisesRegex(ValueError, 'State keys differ'):
            load_local_model(**self.arguments())

    def test_jm_config_and_known_epi_checkpoint_are_rejected_before_asset_reads(self):
        args = self.arguments()
        with patch('imut_cdr_gc.models.loading.verify_local_assets', side_effect=AssertionError('asset read')):
            with self.assertRaisesRegex(ValueError, 'not a JM checkpoint'):
                load_local_model(**{**args, 'expected_checkpoint_sha256': REFERENCE_JM_EPI_SHA256})
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_local_model(**{**args, 'model_config': EpiConfig()})
            with self.assertRaisesRegex(ValueError, 'model_kind'):
                load_local_model(**{**args, 'model_kind': 'unknown'})

    def test_backbone_configuration_roundtrip_and_epi_fields_rejected(self):
        self.assertEqual(BackboneConfig.from_dict(self.config.as_dict()), self.config)
        with self.assertRaises(TypeError):
            BackboneConfig.from_dict({'pocket_hidden': 256})


if __name__ == '__main__':
    unittest.main()
