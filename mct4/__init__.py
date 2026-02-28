"""
MCT4 - Morphogenic Compute Topology v4.0

A self-structuring, continuously-learning compute graph.
"""

from .core import GraphState, Node, Context, NodeType, ActivePathRecord
from .primitives import Primitive, apply_primitive
from .forward import ForwardExecutor
from .learning import LearningEngine
from .structural import StructuralEvolution
from .engine import MCT4, MCT4Config, TrainingMetrics

__version__ = "4.0.0"
__all__ = [
    # Main engine
    "MCT4",
    "MCT4Config",
    "TrainingMetrics",
    
    # Core types
    "GraphState",
    "Node", 
    "Context",
    "NodeType",
    "ActivePathRecord",
    
    # Primitives
    "Primitive",
    "apply_primitive",
    
    # Components
    "ForwardExecutor",
    "LearningEngine",
    "StructuralEvolution",
]
