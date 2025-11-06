"""
GroundingDINO Detection Module
"""
from . import params
from .loader import load_grounding_dino
from .detector import detect_grounding_dino

__all__ = [
    'params',
    'load_grounding_dino',
    'detect_grounding_dino',
]
