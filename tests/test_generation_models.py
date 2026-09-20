"""CPU parity against model AST nodes from the *executed* frozen sources."""
import ast
import copy
from contextlib import nullcontext
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from imut_cdr_gc.models import BackboneConfig, EpiConfig
from imut_cdr_gc.models.fr_cdr import ProteinMLMContrastModel
from imut_cdr_gc.models.jm_epi import EpitopeConditionedFRCDRModel
from imut_cdr_gc.models.loading import load_state_dict_strict
from test_generation_native_reference import NATIVE_FR_SOURCE, NATIVE_EPI_SOURCE

torch.set_num_threads(1)
AA = 'ACDEFGHIKLMNPQRSTVWY'
IDENTITY = Path(__file__).resolve().parents[1] / 'imut_cdr_gc/models/source_identity.json'


class TinyTokenizer:
    mask_token = '<mask>'
    def __len__(self):
        return 23
    def convert_tokens_to_ids(self, token):
        return {**dict(zip(AA, range(20))), 'X':20, '<mask>':21, '<pad>':22}[token]
    def __call__(self, sequences, **kwargs):
        assert kwargs.get('add_special_tokens') is False
        width = max(map(len, sequences))
        ids = [[self.convert_tokens_to_ids(a) for a in s] + [22]*(width-len(s)) for s in sequences]
        attention = [[1]*len(s) + [0]*(width-len(s)) for s in sequences]
        return SimpleNamespace(input_ids=torch.tensor(ids), attention_mask=torch.tensor(attention))


class TinyEncoder(nn.Module):
    """Injected synthetic encoder, not a substitute scientific ESM benchmark."""
    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, max_position_embeddings=512)
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(23, hidden)
        self.context = nn.Linear(hidden, hidden)
        self.pooler = None
        self.contact_head = None
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.context(self.embeddings.word_embeddings(input_ids)))


def tiny_config():
    return EpiConfig(backbone=BackboneConfig(encoder_hidden_size=8, projection_dim=6,
        rdt_heads=2, rdt_steps=3, rdt_ffn_mult=2, rdt_dropout=0.0),
        node_dim=3, pocket_hidden=8, pocket_layers=2, pocket_heads=2, pocket_rbf_k=4, cross_heads=2)


def native_namespace(encoder, tokenizer, cfg):
    b = cfg.backbone
    ns = dict(torch=torch, nn=nn, F=F, math=math, autocast_cuda=nullcontext,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw:tokenizer),
        EsmModel=SimpleNamespace(from_pretrained=lambda *a, **kw:encoder),
        LOCAL_MODEL_DIR='synthetic_fixture_only', print0=lambda *a, **kw:None,
        FREEZE_ENCODER=b.freeze_encoder, USE_GC=b.gradient_checkpointing,
        TIE_WEIGHTS=b.tie_weights, PROJ_DIM=b.projection_dim, SOFT_MASK_BIAS_INIT=b.soft_mask_bias_init,
        USE_RDT=b.use_rdt, RDT_STEPS=b.rdt_steps, RDT_NUM_HEADS=b.rdt_heads,
        RDT_FFN_MULT=b.rdt_ffn_mult, RDT_DROPOUT=b.rdt_dropout, RDT_INIT_SCALE=b.rdt_init_scale,
        RDT_PRELUDE_LAYERS=b.rdt_prelude_layers, RDT_CODA_LAYERS=b.rdt_coda_layers,
        RDT_FINAL_LN=b.rdt_final_ln, ATTN_RESIDUAL_ENABLED=b.attention_residual,
        ATTN_RESIDUAL_HIDDEN=b.attention_residual_hidden, RDT_LOOP_EMBED=b.rdt_loop_embedding,
        RDT_POSITIVE_TEACHER=b.positive_teacher, POOLING_TYPE=b.pooling_type,
        ANCHOR_POOL_STRATEGY=b.anchor_pool_strategy, POSITIVE_POOL_STRATEGY=b.positive_pool_strategy,
        AA_VEC_DIM=cfg.node_dim, POCKET_HIDDEN=cfg.pocket_hidden, POCKET_LAYERS=cfg.pocket_layers,
        POCKET_HEADS=cfg.pocket_heads, POCKET_RBF_K=cfg.pocket_rbf_k, CROSS_HEADS=cfg.cross_heads)
    for text in (NATIVE_FR_SOURCE, NATIVE_EPI_SOURCE):
        tree = ast.parse(text)
        assert all(isinstance(n, (ast.ClassDef, ast.FunctionDef)) for n in tree.body)
        if text is NATIVE_EPI_SOURCE:
            ns['SequenceBackbone'] = ns['ProteinMLMContrastModel']
        exec(compile(tree, '<hash-bound-native-model-only-fixture>', 'exec'), ns)
    return ns


def graph(nodes=3):
    positions = torch.arange(nodes, dtype=torch.float32)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return dict(x=torch.stack([positions, positions + .25, positions * .5], 1),
                edge_index=edges, edge_attr=torch.tensor([[2., 1.], [2., 1.]]))


class GenerationModelTests(unittest.TestCase):
    def test_fixture_exact_node_hashes_and_unchanged_native_AST(self):
        identity = json.loads(IDENTITY.read_text())
        self.assertEqual(identity['source_sha256'][
            'campaigns/design345/analysis/jm_diagnosis/source_evidence/jm_epi_trainer.py'],
            '95562e5d5f443c4d3ae96c47b4c80f0704b5f01c8c1a0f0e08b10e7646f4cd95')
        for name, source in [('fr', NATIVE_FR_SOURCE), ('epi', NATIVE_EPI_SOURCE)]:
            for node in ast.parse(source).body:
                observed = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
                public = identity['public_fixture']['selected_nodes'][name][node.name]
                self.assertEqual(observed, public['sha256'])
                self.assertEqual(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
                                 public['ast_sha256'])
                self.assertEqual(public['original_selected_node_sha256'],
                                 identity['selected_nodes'][name][node.name]['sha256'])

    def compare(self, conditional):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2817)
            config, tokenizer = tiny_config(), TinyTokenizer()
            encoder = TinyEncoder()
            native = native_namespace(copy.deepcopy(encoder), tokenizer, config)
            old_cls = native['EpitopeConditionedFRCDRModel' if conditional else 'ProteinMLMContrastModel']
            original = old_cls().eval()
            if conditional:
                # The trainer initializes identity FiLM. Activate its actual
                # learned-capacity paths so parity is not a trivial identity.
                with torch.no_grad():
                    original.film_mlp[-1].weight.normal_(0, .08)
                    original.film_mlp[-1].bias.normal_(0, .05)
                    original.rdt_block.depth_queries.normal_(0, .1)
            new_cls = EpitopeConditionedFRCDRModel if conditional else ProteinMLMContrastModel
            successor = new_cls(encoder=copy.deepcopy(encoder), tokenizer=tokenizer,
                config=config if conditional else config.backbone).eval()
            receipt = load_state_dict_strict(successor, copy.deepcopy(original.state_dict()))
            self.assertTrue(receipt['strict'])
            self.assertEqual(set(original.state_dict()), set(successor.state_dict()))
            tokens = tokenizer([AA, AA[:16]], add_special_tokens=False)
            masks = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 6, 12]]
            args = [tokens.input_ids, tokens.attention_mask, masks]
            if conditional:
                args.append([graph(3), graph(4)])
            with torch.inference_mode():
                a, b = original(*args), successor(*args)
            for left, right in zip(a, b):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(right).all())
            return successor

    def test_native_backbone_state_and_all_forward_outputs_exact(self):
        self.compare(False)

    def test_native_epitope_state_and_all_forward_outputs_exact(self):
        self.compare(True)

    def test_state_rejects_missing_shape_nonfinite_and_tied_alias_conflicts(self):
        model = self.compare(True)
        good = copy.deepcopy(model.state_dict())
        broken = dict(good)
        broken.pop(next(iter(broken)))
        with self.assertRaisesRegex(ValueError, 'keys'):
            load_state_dict_strict(model, broken)
        broken = copy.deepcopy(good)
        broken['lm_head.bias'] = broken['lm_head.bias'][:1]
        with self.assertRaisesRegex(ValueError, 'shape'):
            load_state_dict_strict(model, broken)
        broken = copy.deepcopy(good)
        broken['lm_head.bias'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            load_state_dict_strict(model, broken)
        # Clone each alias independently to make an actual contradictory state.
        broken = {k:v.clone() for k, v in good.items()}
        broken['lm_head.weight'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'aliases'):
            load_state_dict_strict(model, broken)

    def test_imports_do_not_select_devices_seed_rng_read_assets_or_mutate_environment(self):
        env, rng = dict(os.environ), torch.random.get_rng_state().clone()
        modules = ['imut_cdr_gc.models.fr_cdr', 'imut_cdr_gc.models.jm_epi',
                   'imut_cdr_gc.models.loading', 'imut_cdr_gc.generation']
        with patch('torch.load', side_effect=AssertionError('checkpoint read')), \
             patch('torch.cuda.set_device', side_effect=AssertionError('GPU selection')), \
             patch('torch.manual_seed', side_effect=AssertionError('global RNG seeding')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('directory creation')), \
             patch.object(Path, 'open', side_effect=AssertionError('asset read')):
            for name in modules:
                importlib.reload(importlib.import_module(name))
        self.assertEqual(dict(os.environ), env)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_configuration_is_explicit_not_environment_derived(self):
        with patch.dict(os.environ, RDT_STEPS='999', FR_BASE_SCRIPT='/must/not/be/imported'):
            config = EpiConfig()
            self.assertEqual(config.backbone.rdt_steps, 3)
        self.assertEqual(EpiConfig.from_dict(config.as_dict()), config)
        with self.assertRaisesRegex(ValueError, 'hidden-size'):
            tiny_config().validate(16)


if __name__ == '__main__':
    unittest.main()
