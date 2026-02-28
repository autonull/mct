"""
MCT5 Unified - Morphogenic Compute Topology v5

A next-generation self-structuring learning system with:
- Hybrid learning: PyTorch autograd + dual-signal local learning
- Intelligent structural evolution with adaptive mutation
- Holographic residue for context-aware computation
- Anytime inference with graceful degradation
"""

from .config import MCT5Config, LearningMode
from .types import Node, GraphState, NodeType, ActiveRecord
from .primitives import Primitive, apply_primitive, primitive_derivative
from .residue import HolographicResidue
from .forward import ForwardExecutor
from .learning import LearningEngine, AutogradLearning, DualSignalLearning, HybridLearning
from .structural import StructuralEvolution
from .engine import MCT5

__version__ = "5.0.0-unified"
__all__ = [
    # Main API
    "MCT5",
    "MCT5Config",
    "LearningMode",
    
    # Types
    "Node",
    "GraphState", 
    "NodeType",
    "ActiveRecord",
    
    # Primitives
    "Primitive",
    "apply_primitive",
    "primitive_derivative",
    
    # Components
    "HolographicResidue",
    "ForwardExecutor",
    "LearningEngine",
    "AutogradLearning",
    "DualSignalLearning",
    "HybridLearning",
    "StructuralEvolution",
]
