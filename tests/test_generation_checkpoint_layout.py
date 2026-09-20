"""Strict saved-layout compatibility, real HF architecture but no learned assets."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import EsmConfig, EsmModel
from imut_cdr_gc.models.config import EpiConfig
from imut_cdr_gc.models.jm_epi import EpitopeConditionedFRCDRModel
from imut_cdr_gc.models.loading import (
    CONTACT_KEYS, POSITION_IDS, REFERENCE_ESM_ASSETS, REFERENCE_JM_EPI_SHA256,
    _apply_reference_state_contract, _bind_saved_esm_layout, load_state_dict_strict,
)
from test_generation_models import TinyTokenizer, tiny_config, native_namespace, graph, AA


def small_esm():
    config = EsmConfig(vocab_size=23, hidden_size=8, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=40,
        pad_token_id=22, mask_token_id=21, position_embedding_type='rotary',
        token_dropout=False, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0)
    config._attn_implementation = 'eager'
    return EsmModel(config)


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41537)
        model = EpitopeConditionedFRCDRModel(encoder=small_esm(), tokenizer=TinyTokenizer(),
                                            config=tiny_config()).eval()
        with torch.no_grad():
            model.film_mlp[-1].weight.normal_(0, .08)
            model.film_mlp[-1].bias.normal_(0, .05)
            model.rdt_block.depth_queries.normal_(0, .1)
    full = {k: v.clone() for k, v in model.state_dict().items()}
    saved = {k: v.clone() for k, v in full.items() if k not in CONTACT_KEYS}
    saved[POSITION_IDS] = model.encoder.embeddings.position_ids.clone()
    return model, full, saved


class SavedCheckpointLayoutTests(unittest.TestCase):
    def test_real_HF_and_native_three_outputs_unchanged_with_all_saved_keys_loaded(self):
        model, original, saved = fixture()
        native = native_namespace(copy.deepcopy(model.encoder), model.tokenizer,
                                  model.epi_config)['EpitopeConditionedFRCDRModel']().eval()
        load_state_dict_strict(native, original)
        before = {k: v.clone() for k, v in saved.items()}
        with self.assertRaisesRegex(ValueError, 'State keys differ'):
            load_state_dict_strict(model, saved)
        contract = _bind_saved_esm_layout(model, saved)
        receipt = load_state_dict_strict(model, saved)
        self.assertTrue(receipt['strict'])
        self.assertEqual(receipt['loaded_keys'], len(saved))
        self.assertEqual(set(model.state_dict()), set(saved))
        self.assertEqual(contract['checkpoint_keys_removed'], [])
        self.assertEqual(contract['initialized_forward_parameters'], [])
        for key, value in saved.items():
            self.assertTrue(torch.equal(value, before[key]))
            self.assertTrue(torch.equal(model.state_dict()[key], value))
        self.assertIs(model.lm_head.weight, model.encoder.embeddings.word_embeddings.weight)
        tokens = model.tokenizer([AA, AA[:16]], add_special_tokens=False)
        args = [tokens.input_ids, tokens.attention_mask,
                [list(range(10)), [2, 6, 12]], [graph(3), graph(4)]]
        with torch.inference_mode():
            expected, observed = native(*args), model(*args)
        for first, second in zip(expected, observed):
            self.assertTrue(torch.isfinite(second).all())
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_contact_prediction_cannot_use_random_missing_head(self):
        model, _, saved = fixture()
        _bind_saved_esm_layout(model, saved)
        self.assertEqual(list(model.encoder.contact_head.parameters()), [])
        with self.assertRaisesRegex(RuntimeError, 'Contact prediction unavailable'):
            model.encoder.contact_head(None, None)
        tokens = model.tokenizer([AA], add_special_tokens=False)
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'Contact prediction unavailable'):
            model.encoder.predict_contacts(tokens.input_ids, tokens.attention_mask)

    def test_any_extra_or_missing_learned_key_rejected_before_layout_mutation(self):
        for change in ('extra', 'missing', 'partial_contact'):
            with self.subTest(change=change):
                model, full, saved = fixture()
                if change == 'extra': saved['unreviewed.weight'] = torch.ones(1)
                elif change == 'missing': saved.pop('film_mlp.2.weight')
                else: saved[next(iter(CONTACT_KEYS))] = full[next(iter(CONTACT_KEYS))]
                with self.assertRaisesRegex(ValueError, 'Unreviewed ESM state-key'):
                    _bind_saved_esm_layout(model, saved)
                self.assertIn('position_ids', model.encoder.embeddings._non_persistent_buffers_set)
                self.assertEqual(set(model.state_dict()), set(full))

    def test_position_ids_content_shape_dtype_and_current_buffer_rejected(self):
        for change in ('value', 'shape', 'dtype', 'current', 'already_persistent'):
            with self.subTest(change=change):
                model, _, saved = fixture()
                if change == 'value': saved[POSITION_IDS][0, 4] = 0
                elif change == 'shape': saved[POSITION_IDS] = saved[POSITION_IDS][:, :-1]
                elif change == 'dtype': saved[POSITION_IDS] = saved[POSITION_IDS].float()
                elif change == 'current': model.encoder.embeddings.position_ids[0, 4] = 0
                else: model.encoder.embeddings.register_buffer('position_ids', saved[POSITION_IDS], persistent=True)
                with self.assertRaises(ValueError): _bind_saved_esm_layout(model, saved)

    def test_nontensor_or_nested_state_rejected(self):
        model, _, saved = fixture()
        saved[POSITION_IDS] = saved[POSITION_IDS].tolist()
        with self.assertRaisesRegex(ValueError, 'plain tensor'):
            _bind_saved_esm_layout(model, saved)

    def test_strict_loader_does_not_cast_checkpoint_dtype(self):
        model, full, _ = fixture()
        full['lm_head.bias'] = full['lm_head.bias'].half()
        with self.assertRaisesRegex(ValueError, 'dtype mismatch'):
            load_state_dict_strict(model, full)

    def test_unknown_hash_never_activates_reference_exceptions(self):
        model, full, saved = fixture()
        contract = _apply_reference_state_contract(model, saved, '0'*64,
            REFERENCE_ESM_ASSETS, EpiConfig().as_dict(), '4.46.3')
        self.assertFalse(contract['checkpoint_specific_compatibility'])
        self.assertEqual(set(model.state_dict()), set(full))
        with self.assertRaisesRegex(ValueError, 'State keys differ'):
            load_state_dict_strict(model, saved)

    def test_reference_version_assets_architecture_and_counts_are_exact_gates(self):
        model, _, saved = fixture()
        cases = [({}, '4.46.3', EpiConfig().as_dict(), 'exact tokenizer'),
                 (REFERENCE_ESM_ASSETS, '4.47.0', EpiConfig().as_dict(), 'Transformers'),
                 (REFERENCE_ESM_ASSETS, '4.46.3', tiny_config().as_dict(), 'exact tokenizer'),
                 (REFERENCE_ESM_ASSETS, '4.46.3', EpiConfig().as_dict(), 'state counts')]
        for assets, version, architecture, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _apply_reference_state_contract(model, saved, REFERENCE_JM_EPI_SHA256,
                                                assets, architecture, version)

    def test_full_650M_architecture_meta_only_matches_saved_699_key_contract(self):
        # Genuine HF/full JM architecture on meta: verifies the saved key/shape
        # layout without allocating/loading 3 GB of learned parameters.
        class Vocabulary33(TinyTokenizer):
            def __len__(self): return 33
        cfg = EsmConfig(vocab_size=33, hidden_size=1280, num_hidden_layers=33,
            num_attention_heads=20, intermediate_size=5120, max_position_embeddings=1026,
            pad_token_id=1, mask_token_id=32, position_embedding_type='rotary',
            token_dropout=True, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0)
        cfg._attn_implementation = 'eager'
        with torch.device('meta'):
            model = EpitopeConditionedFRCDRModel(encoder=EsmModel(cfg),
                tokenizer=Vocabulary33(), config=EpiConfig()).eval()
        model.encoder.embeddings.register_buffer('position_ids',
            torch.arange(1026, dtype=torch.int64).reshape(1, -1), persistent=False)
        full = model.state_dict()
        self.assertEqual(len(full), 700)
        saved = {k: v for k, v in full.items() if k not in CONTACT_KEYS}
        saved[POSITION_IDS] = model.encoder.embeddings.position_ids.clone()
        self.assertEqual(len(saved), 699)
        self.assertTrue(all(v.device.type == 'meta' for k, v in saved.items() if k != POSITION_IDS))
        receipt = _apply_reference_state_contract(model, saved, REFERENCE_JM_EPI_SHA256,
            REFERENCE_ESM_ASSETS, EpiConfig().as_dict(), '4.46.3')
        self.assertEqual(len(model.state_dict()), 699)
        self.assertEqual(set(model.state_dict()), set(saved))
        self.assertTrue(receipt['strict_all_checkpoint_keys'])
        self.assertFalse(receipt['contact_prediction_available'])


if __name__ == '__main__':
    unittest.main()
