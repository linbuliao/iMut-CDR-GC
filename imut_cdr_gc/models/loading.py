"""Explicit local-only asset loading, separate from import-safe model math."""
from __future__ import annotations
from collections import defaultdict
import hashlib
from pathlib import Path
import re
from .config import BackboneConfig, EpiConfig

MODEL_NAMES = {'jm': 'iMut-CDR-JM', 'jm-epi': 'iMut-CDR-JM-Epi'}

TOKENIZER_FILES = {'config.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'vocab.txt', 'tokenizer.json', 'added_tokens.json'}

# Exact previously audited production artifact, not a family-wide key exception.
REFERENCE_JM_EPI_SHA256 = 'a19bd33544bbfcb8a18dfb88bb007439cd57e702c4d89963e143df79991c144e'
REFERENCE_ESM_ASSETS = {
    'config.json': '539095c22efc52a09d6147074ba4ca119f76a890df5901213b2b55f7d2f96b2b',
    'special_tokens_map.json': '3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1',
    'tokenizer_config.json': '7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad',
    'vocab.txt': '0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03',
}
POSITION_IDS = 'encoder.embeddings.position_ids'
CONTACT_KEYS = frozenset({'encoder.contact_head.regression.weight',
                          'encoder.contact_head.regression.bias'})


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_local_assets(model_dir, checkpoint, expected_checkpoint_sha256, asset_sha256):
    directory = Path(model_dir).resolve()
    weights = Path(checkpoint).resolve()
    if not directory.is_dir() or not weights.is_file():
        raise FileNotFoundError('Provide explicit existing local model/config and checkpoint assets')
    if not re.fullmatch('[0-9a-f]{64}', str(expected_checkpoint_sha256)):
        raise ValueError('An explicit expected checkpoint SHA-256 is required')
    if not isinstance(asset_sha256, dict) or 'config.json' not in asset_sha256:
        raise ValueError('Pin config and tokenizer assets by relative filename and SHA-256')
    present = {p.name for p in directory.iterdir() if p.name in TOKENIZER_FILES and p.is_file()}
    if present != set(asset_sha256) or not ({'vocab.txt', 'tokenizer.json'} & present):
        raise ValueError('Manifest must pin every present config/tokenizer asset and include vocabulary')
    observed = {}
    for relative, expected in sorted(asset_sha256.items()):
        if relative not in TOKENIZER_FILES or not re.fullmatch('[0-9a-f]{64}', str(expected)):
            raise ValueError('Invalid local asset manifest entry')
        path = directory / relative
        if path.resolve().parent != directory:
            raise ValueError('Model asset may not escape its explicit directory')
        actual = sha256(path)
        if actual != expected:
            raise ValueError('Local asset hash mismatch: ' + relative)
        observed[relative] = actual
    actual = sha256(weights)
    if actual != expected_checkpoint_sha256:
        raise ValueError('Checkpoint hash mismatch')
    return directory, weights, dict(checkpoint_sha256=actual, asset_sha256=observed)


def load_state_dict_strict(model, state):
    """No silent missing/unexpected keys or contradictory tied-weight aliases."""
    import torch
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or
            not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Expected a plain tensor state_dict, not an executable or nested checkpoint')
    expected = model.state_dict()
    if set(state) != set(expected):
        missing, extra = sorted(set(expected)-set(state)), sorted(set(state)-set(expected))
        raise ValueError('State keys differ; no partial checkpoint loading is allowed; '
                         f'missing={missing[:20]}, unexpected={extra[:20]}')
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape:
            raise ValueError('State tensor shape mismatch: ' + name)
        if tensor.dtype != expected[name].dtype:
            raise ValueError('State tensor dtype mismatch: ' + name)
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite checkpoint tensor: ' + name)
    aliases = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].append(name)
    for names in aliases.values():
        if len(names) > 1 and any(not torch.equal(state[names[0]], state[k]) for k in names[1:]):
            raise ValueError('Tied-weight aliases disagree in checkpoint')
    model.load_state_dict(state, strict=True)
    return dict(loaded_keys=len(state), strict=True, tied_aliases_checked=True)


def _bind_saved_esm_layout(model, state):
    """Match saved buffer/module layout without filtering any checkpoint key.

    This small operation is separately tested with a real, tiny HF EsmModel.
    Only the hash/version/config-gated caller below may use it for an asset.
    The generation forward never calls the contact-prediction module; attempts
    to call that absent head fail explicitly instead of using random weights.
    """
    import torch
    from transformers.models.esm.modeling_esm import EsmModel, EsmContactPredictionHead
    if not isinstance(state, dict) or any(not isinstance(k, str) or
            not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Expected plain tensor state_dict for audited ESM layout')
    own = model.state_dict()
    if set(own)-set(state) != CONTACT_KEYS or set(state)-set(own) != {POSITION_IDS}:
        raise ValueError('Unreviewed ESM state-key differences; compatibility refused')
    encoder = model.encoder
    if type(encoder) is not EsmModel or type(encoder.contact_head) is not EsmContactPredictionHead:
        raise ValueError('Only the inspected native HF ESM/contact classes are supported')
    embedding = encoder.embeddings
    current = embedding.position_ids
    saved = state[POSITION_IDS]
    width = int(encoder.config.max_position_embeddings)
    expected = torch.arange(width, dtype=torch.int64, device=current.device).reshape(1, -1)
    if (POSITION_IDS.split('.')[-1] not in embedding._non_persistent_buffers_set or
            saved.dtype != torch.int64 or current.dtype != torch.int64 or
            tuple(saved.shape) != (1, width) or tuple(current.shape) != (1, width) or
            saved.device.type != 'cpu' or current.device.type != 'cpu' or
            not torch.equal(saved, current) or not torch.equal(saved, expected)):
        raise ValueError('Saved/current position IDs must be the exact native int64 arange')
    head = encoder.contact_head.state_dict()
    head_width = int(encoder.config.num_hidden_layers) * int(encoder.config.num_attention_heads)
    if (set(head) != {'regression.weight', 'regression.bias'} or
            tuple(head['regression.weight'].shape) != (1, head_width) or
            tuple(head['regression.bias'].shape) != (1,)):
        raise ValueError('Unreviewed contact-head layout')

    class UnavailableContactPrediction(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError('Contact prediction unavailable: this generation checkpoint has no contact-head weights')

    # Restore exactly the serialized layout. No tensor from `state` is removed,
    # renamed, substituted, cast or initialized to manufacture a successful load.
    embedding.register_buffer('position_ids', current, persistent=True)
    encoder.contact_head = UnavailableContactPrediction()
    if set(model.state_dict()) != set(state):
        raise ValueError('Saved ESM state layout remains inconsistent')
    return dict(profile='saved_position_ids_no_contact_prediction.v1',
                checkpoint_keys_removed=[], checkpoint_keys_renamed=[],
                initialized_forward_parameters=[], persistent_position_ids_restored=True,
                contact_prediction_available=False, strict_all_checkpoint_keys=True)


def _apply_reference_state_contract(model, state, checkpoint_sha256, asset_sha256,
                                    architecture, transformers_version):
    if checkpoint_sha256 != REFERENCE_JM_EPI_SHA256:
        return dict(profile='unchanged_HF_state_dict', checkpoint_specific_compatibility=False)
    if transformers_version != '4.46.3':
        raise ValueError('Audited JM-Epi state contract requires Transformers 4.46.3')
    if asset_sha256 != REFERENCE_ESM_ASSETS or architecture != EpiConfig().as_dict():
        raise ValueError('Audited JM-Epi state contract requires its exact tokenizer/config and architecture')
    if len(state) != 699 or len(model.state_dict()) != 700:
        raise ValueError('Audited JM-Epi state counts changed')
    return _bind_saved_esm_layout(model, state)


def load_local_model(*, model_dir, checkpoint, expected_checkpoint_sha256, asset_sha256,
                     model_config=None, device='cpu', model_kind='jm-epi'):
    """Build ESM from local config, then load the complete fine-tuned state.

    JM instantiates the native sequence-only backbone; JM-Epi additionally
    instantiates its antigen-conditioning modules. No pretrained-weight download,
    trainer import, graph substitution or partial-state loading occurs. Synthetic
    tests and separately recorded real-asset validation have different scopes.
    """
    if model_kind not in MODEL_NAMES:
        raise ValueError('model_kind must be jm or jm-epi')
    if model_kind == 'jm' and expected_checkpoint_sha256 == REFERENCE_JM_EPI_SHA256:
        raise ValueError('The antigen-conditioned JM-Epi checkpoint is not a JM checkpoint')
    config_type = BackboneConfig if model_kind == 'jm' else EpiConfig
    config = config_type() if model_config is None else model_config
    if isinstance(config, dict):
        config = config_type.from_dict(config)
    if not isinstance(config, config_type):
        raise ValueError('Model configuration does not match the selected model_kind')
    directory, weights, identity = verify_local_assets(model_dir, checkpoint,
        expected_checkpoint_sha256, asset_sha256)
    import torch
    import transformers
    from transformers import AutoTokenizer, EsmConfig, EsmModel
    encoder_config = EsmConfig.from_pretrained(str(directory), local_files_only=True)
    if encoder_config.model_type != 'esm':
        raise ValueError('Only the executed ESM2 backbone family is supported')
    config.validate(int(encoder_config.hidden_size))
    encoder_config._attn_implementation = 'eager'
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    encoder = EsmModel(encoder_config)
    if model_kind == 'jm':
        from .fr_cdr import ProteinMLMContrastModel
        model = ProteinMLMContrastModel(encoder=encoder, tokenizer=tokenizer, config=config)
    else:
        from .jm_epi import EpitopeConditionedFRCDRModel
        model = EpitopeConditionedFRCDRModel(encoder=encoder, tokenizer=tokenizer, config=config)
    state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    if model_kind == 'jm-epi':
        state_contract = _apply_reference_state_contract(model, state, identity['checkpoint_sha256'],
            identity['asset_sha256'], config.as_dict(), transformers.__version__)
    else:
        # No JM-specific key exception is inferred from the JM-Epi artifact.
        # An incompatible historical JM serialization must fail, not be filtered.
        state_contract = dict(profile='unchanged_HF_state_dict', checkpoint_specific_compatibility=False)
    loading = load_state_dict_strict(model, state)
    del state
    _, _, rechecked = verify_local_assets(directory, weights, expected_checkpoint_sha256, asset_sha256)
    if rechecked != identity:
        raise ValueError('Assets changed during loading')
    model.to(torch.device(device)).eval()
    return model, dict(**identity, **loading, architecture=config.as_dict(), device=str(device),
                      model_kind=model_kind, model=MODEL_NAMES[model_kind],
                      antigen_conditioned=model_kind == 'jm-epi',
                      state_contract=state_contract,
                      local_files_only=True, pretrained_weight_download=False,
                      native_real_checkpoint_parity_verified=False)
