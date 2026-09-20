"""Model configuration is lightweight; neural definitions load on demand."""
from .config import BackboneConfig, EpiConfig

__all__ = ['BackboneConfig', 'EpiConfig', 'EpitopeConditionedFRCDRModel',
           'ProteinMLMContrastModel']


def __getattr__(name):
    if name == 'EpitopeConditionedFRCDRModel':
        from .jm_epi import EpitopeConditionedFRCDRModel
        return EpitopeConditionedFRCDRModel
    if name == 'ProteinMLMContrastModel':
        from .fr_cdr import ProteinMLMContrastModel
        return ProteinMLMContrastModel
    raise AttributeError(name)
