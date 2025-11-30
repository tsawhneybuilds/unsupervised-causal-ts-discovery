"""
SVAR-GFCI: Structural Vector Autoregression Greedy FCI

A hybrid causal discovery algorithm that combines:
- SVAR-GES (score-based) for initial graph estimation
- SVAR-FCI-style CI-based pruning and orientation

Based on Algorithm 3.2 from Malinsky & Spirtes (2018):
"Causal Structure Learning from Multivariate Time Series in Settings with Unmeasured Confounding"

This implementation reuses components from svar_fci where applicable.
"""

from .graph import DynamicCPDAG
from .score import local_score, ScoreCache
from .ges import SVAR_GES
from .algo import SVAR_GFCI

__all__ = [
    'DynamicCPDAG',
    'local_score',
    'ScoreCache', 
    'SVAR_GES',
    'SVAR_GFCI',
]

