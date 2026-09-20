"""Strict native scoring entry points. Heavy dependencies load only on demand."""
from .esm2 import score_esm2
from .native_structure import score_3d

__all__ = ['score_esm2', 'score_3d']
