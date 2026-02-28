"""
MCT5 — Morphogenic Compute Topology v5

A self-structuring, continuously-learning compute graph with:
- Dual-signal learning: local contrastive + retrograde error
- Holographic Residue: complex-valued phase-rotation routing memory
- Low-rank parametric nodes (D×r)
- Vectorized depth-layer batch execution
- 12 primitive operators with proper derivatives
"""

from .config import MCT5Config
from .types import Node, GraphState, NodeType, ActiveRecord
from .primitives import Primitive, apply_primitive, primitive_derivative
from .residue import HolographicResidue
from .forward import ForwardExecutor
from .learning import LearningEngine
from .structural import StructuralEvolution
from .engine import MCT5

__version__ = "5.0.0"
__all__ = [
    "MCT5",
    "MCT5Config",
    "Node",
    "GraphState",
    "NodeType",
    "ActiveRecord",
    "Primitive",
    "apply_primitive",
    "primitive_derivative",
    "HolographicResidue",
    "ForwardExecutor",
    "LearningEngine",
    "StructuralEvolution",
]
