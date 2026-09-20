"""Private, explicit-asset loader for the unchanged native likelihood scorer.

No weights or private source are distributed here. Loading is opt-in and never
adds directories to sys.path or installs global aliases in sys.modules.
"""
from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
import platform
import re
from types import ModuleType

from .assets import AssetError, file_sha256, verified_asset

SOURCE_SHA256 = {
    'scorer_source': 'baa5d6302cbe9fff18bfc7678e5b4506aa6d408243d53722825fb695801df13a',
    'model_source': '22209826809e0f66d279ef7a0afa5beba474fc53f929449d61972c49cafcd20b',
}
CONFIG_FILES = frozenset({'config.json', 'tokenizer_config.json', 'special_tokens_map.json',
                          'vocab.txt', 'tokenizer.json', 'added_tokens.json'})
ARCHITECTURE = dict(RDT_STEPS=3, RDT_PRELUDE_LAYERS=1, RDT_CODA_LAYERS=1,
    RDT_NUM_HEADS=8, RDT_FFN_MULT=4, RDT_DROPOUT=0.1, RDT_INIT_SCALE=0.1,
    RDT_FINAL_LN=True, ATTN_RESIDUAL_ENABLED=True, ATTN_RESIDUAL_HIDDEN=256,
    RDT_LOOP_EMBED=True)


def identity_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def verify_native_assets(project_root, assets):
    """Hash all declared code/config/weights before imports or model creation.

    Intended runtime operation: tests supply only tiny synthetic assets. The
    implementation/review does not inspect any private checkpoint.
    """
    if not isinstance(assets, dict) or assets.get('schema') != 'campaign-likelihood-assets.v1':
        raise AssetError('Require campaign-likelihood-assets.v1 manifest')
    root = Path(project_root).resolve(strict=True)
    paths = {}
    for role in ('scorer_source', 'model_source', 'weights'):
        paths[role] = verified_asset(root, assets.get(role), role=role)
        if role in SOURCE_SHA256 and assets[role]['sha256'] != SOURCE_SHA256[role]:
            raise AssetError('Unreviewed native source revision: ' + role)
    value = assets.get('local_model_dir')
    if not isinstance(value, str) or not value:
        raise AssetError('An explicit local_model_dir is required')
    directory = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if not directory.is_dir() or root not in directory.parents:
        raise AssetError('local_model_dir must exist inside the explicit project')
    declared = assets.get('config_tokenizer_sha256')
    present = {p.name for p in directory.iterdir() if p.is_file() and p.name in CONFIG_FILES}
    if (not isinstance(declared, dict) or set(declared) != present
            or 'config.json' not in present or not present & {'vocab.txt', 'tokenizer.json'}):
        raise AssetError('Pin every present config/tokenizer file, including config and vocabulary')
    for filename, expected in declared.items():
        if filename not in CONFIG_FILES:
            raise AssetError('Unsupported config/tokenizer asset')
        paths['config/' + filename] = verified_asset(root,
            dict(path=str(directory / filename), sha256=expected), role=filename)
        if paths['config/' + filename].parent != directory:
            raise AssetError('Tokenizer assets cannot resolve outside local_model_dir')
        if filename.endswith('.json'):
            content = json.loads(paths['config/' + filename].read_text())
            if not isinstance(content, dict) or content.get('auto_map'):
                raise AssetError('Custom remote-code tokenizer/config hooks are unsupported')
    config = json.loads((directory / 'config.json').read_text())
    if config.get('model_type') != 'esm' or config.get('max_position_embeddings', 0) < 298:
        raise AssetError('Require an ESM config capable of the complete native 298 positions')
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    identity = dict(native_sources={k: hashes[k] for k in SOURCE_SHA256},
        checkpoint_sha256=hashes['weights'], config_tokenizer_sha256=dict(declared),
        architecture=dict(ARCHITECTURE), local_files_only=True)
    return paths, directory, hashes, identity


def _execute_source(path, *, expected_sha256, imports=None):
    """Resolve the native sibling import inside this module only, not globally."""
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise AssetError('Native source changed before execution')
    module = ModuleType('_imut_likelihood_' + hashlib.sha256(source).hexdigest())
    module.__file__ = str(path)
    local_builtins = dict(vars(builtins))
    original_import = builtins.__import__

    def bound_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and imports and name in imports:
            return imports[name]
        return original_import(name, globals, locals, fromlist, level)

    local_builtins['__import__'] = bound_import
    module.__dict__['__builtins__'] = local_builtins
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


class _SafeTorch:
    """Disable only the native legacy pickle fallback, never tensor mathematics."""
    def __init__(self, torch_module):
        self._torch = torch_module

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def load(self, *args, **kwargs):
        if kwargs.get('weights_only') is not True:
            raise AssetError('Native deserialization without weights_only=True is disabled')
        return self._torch.load(*args, **kwargs)


def load_native_scorer(project_root, assets, *, device='cpu'):
    """Explicit future inference entry; no device discovery and no download.

    Retains the original model/scorer source, masks, autocast, per-antibody
    forward isolation and reduction. This packaging interface is not evidence
    of private-weight/GPU parity and is not connected to production workers.
    """
    if not isinstance(device, str) or re.fullmatch(r'cpu|cuda:\d+', device) is None:
        raise ValueError('Choose cpu or one explicit visible cuda:N device')
    paths, directory, hashes, identity = verify_native_assets(project_root, assets)
    model_module = _execute_source(paths['model_source'], expected_sha256=hashes['model_source'])
    observed = {key: getattr(model_module, key, None) for key in ARCHITECTURE}
    if observed != ARCHITECTURE:
        raise AssetError('Ambient architecture overrides differ from the reviewed native model')
    model_module.torch = _SafeTorch(model_module.torch)
    scorer_module = _execute_source(paths['scorer_source'],
        expected_sha256=hashes['scorer_source'],
        imports={'imut_cdr_jm_model': model_module})
    scorer = scorer_module.IMutCDRJMLikelihood(weights_path=str(paths['weights']),
        local_model_dir=str(directory), device=device)
    for name, path in paths.items():
        if file_sha256(path) != hashes[name]:
            raise AssetError('Native asset changed while loading: ' + name)
    import torch
    import transformers
    runtime = dict(python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, cuda=torch.version.cuda, device=device,
        tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        autocast='native default CUDA autocast; CPU nullcontext')
    if device.startswith('cuda:'):
        runtime.update(device_name=torch.cuda.get_device_name(device),
            compute_capability=list(torch.cuda.get_device_capability(device)),
            cudnn=torch.backends.cudnn.version())
    identity['runtime'] = runtime
    return scorer, identity
