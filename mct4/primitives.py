"""
MCT4 Primitive Operators

All operators output ℝᴰ. Post-W application, the primitive is the nonlinearity.
"""

import numpy as np
from enum import Enum, auto
from typing import Union, List


class Primitive(Enum):
    """Available primitive operators for nodes."""
    # Unary (1 input)
    RELU = auto()
    TANH = auto()
    GELU = auto()
    SOFTMAX = auto()
    L2NORM = auto()
    FORK = auto()
    
    # Binary/N-ary (2+ inputs)
    ADD = auto()
    ATTENTION = auto()
    GATE = auto()
    CONCAT = auto()


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation: x * Φ(x) where Φ is standard normal CDF."""
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Approximate GELU derivative."""
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3))) + \
           x * 0.5 * (1.0 - np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)) ** 2) * \
           np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x ** 2)


def apply_primitive(primitive: Primitive, inputs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """
    Apply a primitive operator to input(s).
    
    Args:
        primitive: The primitive operator to apply
        inputs: Single array for unary, list of arrays for binary/n-ary
        
    Returns:
        Output array of shape (D,)
    """
    if primitive == Primitive.RELU:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return np.maximum(0, x)
    
    elif primitive == Primitive.TANH:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return np.tanh(x)
    
    elif primitive == Primitive.GELU:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return gelu(x)
    
    elif primitive == Primitive.SOFTMAX:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        # Stable softmax
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x) + 1e-9)
    
    elif primitive == Primitive.L2NORM:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        norm = np.linalg.norm(x) + 1e-9
        return x / norm
    
    elif primitive == Primitive.FORK:
        # Pass-through with fan-out capability (handled by graph wiring)
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x.copy()
    
    elif primitive == Primitive.ADD:
        # Residual connection: X + Y (or mean of multiple)
        if isinstance(inputs, list):
            return np.mean(inputs, axis=0)
        return inputs
    
    elif primitive == Primitive.ATTENTION:
        # Full attention operation: softmax(XY^T/√D) * Y
        # For single node, treat inputs as Q and K/V
        if isinstance(inputs, list) and len(inputs) >= 2:
            Q, K = inputs[0], inputs[1]
            V = inputs[2] if len(inputs) > 2 else K
            D = len(Q)
            # Single-head attention scaled
            attn = np.softmax(np.dot(Q, K) / np.sqrt(D))
            return attn * V
        # Single input: self-attention style
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x.copy()
    
    elif primitive == Primitive.GATE:
        # Multiplicative gating: X ⊙ σ(Y)
        if isinstance(inputs, list) and len(inputs) >= 2:
            X, Y = inputs[0], inputs[1]
            gate = 1.0 / (1.0 + np.exp(-Y))  # sigmoid
            return X * gate
        return inputs if isinstance(inputs, np.ndarray) else inputs[0]
    
    elif primitive == Primitive.CONCAT:
        # Concatenate then mean-pool to D
        if isinstance(inputs, list):
            return np.mean(inputs, axis=0)
        return inputs
    
    else:
        raise ValueError(f"Unknown primitive: {primitive}")


def primitive_derivative(primitive: Primitive, x: np.ndarray, output: np.ndarray) -> np.ndarray:
    """
    Compute element-wise derivative of primitive for local learning.
    Returns the diagonal of the Jacobian (sufficient for element-wise ops).
    """
    if primitive == Primitive.RELU:
        return (x > 0).astype(float)
    
    elif primitive == Primitive.TANH:
        return 1.0 - output ** 2
    
    elif primitive == Primitive.GELU:
        return gelu_derivative(x)
    
    elif primitive == Primitive.SOFTMAX:
        # Jacobian of softmax: diag(s) - s s^T, use diagonal approximation
        return output * (1.0 - output)
    
    elif primitive == Primitive.L2NORM:
        norm = np.linalg.norm(x) + 1e-9
        return (1.0 / norm) * (1.0 - (x ** 2) / (norm ** 2))
    
    elif primitive in [Primitive.FORK, Primitive.ADD, Primitive.CONCAT]:
        return np.ones_like(x)
    
    elif primitive == Primitive.ATTENTION:
        return np.ones_like(x)
    
    elif primitive == Primitive.GATE:
        return np.ones_like(x)
    
    else:
        return np.ones_like(x)
