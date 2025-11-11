"""
OWLv2 Detection Module
"""
from . import params
from .loader import load_owlv2
from .detector import detect_owlv2

__all__ = [
    'params',
    'load_owlv2',
    'detect_owlv2',
]
