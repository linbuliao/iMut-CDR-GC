"""Explicit architecture settings; no environment-derived model parameters."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class BackboneConfig:
    encoder_hidden_size: int = 1280
    freeze_encoder: bool = False
    gradient_checkpointing: bool = False
    tie_weights: bool = True
    projection_dim: int = 128
    soft_mask_bias_init: float = -2.0
    use_rdt: bool = True
    rdt_steps: int = 3
    rdt_heads: int = 8
    rdt_ffn_mult: int = 4
    rdt_dropout: float = 0.1
    rdt_init_scale: float = 0.1
    rdt_prelude_layers: int = 1
    rdt_coda_layers: int = 1
    rdt_final_ln: bool = True
    attention_residual: bool = True
    attention_residual_hidden: int = 256
    rdt_loop_embedding: bool = True
    positive_teacher: bool = True
    pooling_type: str = 'attn'
    anchor_pool_strategy: str = 'soft'
    positive_pool_strategy: str = 'soft'

    def validate(self, actual_hidden_size):
        if actual_hidden_size != self.encoder_hidden_size:
            raise ValueError('Encoder/config hidden-size mismatch')
        for name in ('encoder_hidden_size', 'projection_dim', 'rdt_heads', 'rdt_ffn_mult',
                     'attention_residual_hidden'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError('Invalid positive architecture integer: ' + name)
        for name in ('rdt_steps', 'rdt_prelude_layers', 'rdt_coda_layers'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError('Invalid nonnegative architecture integer: ' + name)
        if self.encoder_hidden_size % self.rdt_heads:
            raise ValueError('RDT heads must divide hidden size')
        if not 0 <= self.rdt_dropout < 1:
            raise ValueError('Invalid dropout')
        if not all(math.isfinite(v) for v in (self.rdt_init_scale, self.soft_mask_bias_init)):
            raise ValueError('Nonfinite model configuration')
        if self.pooling_type not in ('mean', 'attn'):
            raise ValueError('Unknown pooling type')

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class EpiConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    node_dim: int = 30
    pocket_hidden: int = 256
    pocket_layers: int = 4
    pocket_heads: int = 8
    pocket_rbf_k: int = 16
    cross_heads: int = 4

    def validate(self, actual_hidden_size):
        self.backbone.validate(actual_hidden_size)
        for name in ('node_dim', 'pocket_hidden', 'pocket_layers', 'pocket_heads',
                     'pocket_rbf_k', 'cross_heads'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError('Invalid positive architecture integer: ' + name)
        if self.pocket_hidden % self.pocket_heads or actual_hidden_size % self.cross_heads:
            raise ValueError('Attention heads must divide their hidden size')

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data['backbone'] = BackboneConfig(**data.get('backbone', {}))
        return cls(**data)
