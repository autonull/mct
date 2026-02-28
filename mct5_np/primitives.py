"""
MCT5 Primitive Operators

12 operators, all outputting ℝᴰ.  Each has a paired derivative function
(diagonal Jacobian approximation) used in the learning phase.
"""

from __future__ import annotations
import numpy as np
from enum import Enum, auto
from typing import Union, List


class Primitive(Enum):
    # ── Unary ────────────────────────────────────
    RELU       = auto()  # max(0, x)
    GELU       = auto()  # x · Φ(x)
    TANH       = auto()  # tanh(x)
    SWISH      = auto()  # x · σ(x)
    SOFTMAX    = auto()  # stable softmax
    L2NORM     = auto()  # x / ||x||
    FORK       = auto()  # identity (routing pass-through)
    QUADRATIC  = auto()  # x ⊙ x + x   ← squared self-features
    # ── Binary/N-ary ─────────────────────────────
    ADD              = auto()  # mean(inputs)
    GATE             = auto()  # x ⊙ σ(y)
    BILINEAR         = auto()  # dot(x,y)/√D · x   (attention-like)
    CONCAT_PROJECT   = auto()  # learned projection of cat(x, y)
    PRODUCT          = auto()  # x ⊙ y (element-wise product ← KEY for XOR cross-terms)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _gelu(x: np.ndarray) -> np.ndarray:
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _gelu_deriv(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_v = np.tanh(inner)
    sech2 = 1.0 - tanh_v ** 2
    return 0.5 * (1.0 + tanh_v) + 0.5 * x * sech2 * c * (1.0 + 3 * 0.044715 * x ** 2)


def _stable_softmax(x: np.ndarray) -> np.ndarray:
    xc = x - x.max()
    e = np.exp(xc)
    return e / (e.sum() + 1e-9)


# ---------------------------------------------------------------------------
# Forward application
# ---------------------------------------------------------------------------

def apply_primitive(primitive: Primitive,
                    inputs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """
    Apply primitive operator to input(s).

    All outputs are ℝᴰ. Binary/N-ary ops receive a list; unary ops receive
    a single array. The pre-primitive linear transform is applied by the
    calling node before this function is invoked.
    """

    # ── Unary ────────────────────────────────────────────────────────────────
    if primitive is Primitive.RELU:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return np.maximum(0.0, x)

    elif primitive is Primitive.GELU:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return _gelu(x)

    elif primitive is Primitive.TANH:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return np.tanh(x)

    elif primitive is Primitive.SWISH:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x * _sigmoid(x)

    elif primitive is Primitive.SOFTMAX:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return _stable_softmax(x)

    elif primitive is Primitive.L2NORM:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x / (np.linalg.norm(x) + 1e-9)

    elif primitive is Primitive.FORK:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x.copy()

    elif primitive is Primitive.QUADRATIC:
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x * x + x   # x² + x (maps XOR-style patterns to linearly separable space)

    # ── Binary / N-ary ────────────────────────────────────────────────────────
    elif primitive is Primitive.ADD:
        if isinstance(inputs, list):
            return np.mean(inputs, axis=0)
        return inputs

    elif primitive is Primitive.GATE:
        if isinstance(inputs, list) and len(inputs) >= 2:
            X, Y = inputs[0], inputs[1]
            return X * _sigmoid(Y)
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x

    elif primitive is Primitive.CONCAT_PROJECT:
        if isinstance(inputs, list) and len(inputs) >= 2:
            # Mean-aggregate pairs then return (correct D-dimensional result)
            # A learned projection would require stored state; we use mean-pool here
            # and rely on the calling node's W matrix to provide the learned projection.
            return np.mean(inputs, axis=0)
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x

    elif primitive is Primitive.PRODUCT:
        # Element-wise product of two inputs — creates cross-term features.
        # PRODUCT(x, y)[i] = x[i] * y[i]
        # This makes XOR linearly separable: for x=[a,b], if we feed x twice,
        # PRODUCT(x,x) = x^2, but PRODUCT(input, tanh(input)) gives signed cross-terms.
        if isinstance(inputs, list) and len(inputs) >= 2:
            X, Y = inputs[0], inputs[1]
            return X * Y  # element-wise product
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x * x  # self-product fallback

    elif primitive is Primitive.BILINEAR:
        if isinstance(inputs, list) and len(inputs) >= 2:
            X, Y = inputs[0], inputs[1]
            D = len(X)
            s = float(np.dot(X, Y)) / (np.sqrt(D) + 1e-9)
            return s * X
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x

    elif primitive is Primitive.CONCAT_PROJECT:
        if isinstance(inputs, list) and len(inputs) >= 2:
            return np.mean(inputs, axis=0)
        x = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        return x

    else:
        raise ValueError(f"Unknown primitive: {primitive}")


# ---------------------------------------------------------------------------
# Derivative (diagonal Jacobian approximation)
# Used by the learning engine for local gradient computation.
# ---------------------------------------------------------------------------

def primitive_derivative(primitive: Primitive,
                          x: np.ndarray,
                          out: np.ndarray) -> np.ndarray:
    """
    Compute element-wise derivative ∂primitive/∂x  (diagonal of Jacobian).

    Args:
        x:   Pre-primitive input (post-W)
        out: Output of primitive (for ops where derivative reuses output)

    Returns:
        Array of shape (D,) representing ∂output_i / ∂input_i
    """
    if primitive is Primitive.RELU:
        return (x > 0).astype(float)

    elif primitive is Primitive.GELU:
        return _gelu_deriv(x)

    elif primitive is Primitive.TANH:
        return 1.0 - out ** 2

    elif primitive is Primitive.SWISH:
        sig = _sigmoid(x)
        return sig + x * sig * (1.0 - sig)

    elif primitive is Primitive.SOFTMAX:
        # Diagonal of Jacobian: s_i(1 - s_i)
        return out * (1.0 - out)

    elif primitive is Primitive.L2NORM:
        norm = np.linalg.norm(x) + 1e-9
        return (1.0 / norm) * (1.0 - (x / norm) ** 2)

    elif primitive is Primitive.QUADRATIC:
        return 2.0 * x + 1.0

    elif primitive in (Primitive.FORK,
                       Primitive.ADD,
                       Primitive.GATE,
                       Primitive.BILINEAR,
                       Primitive.CONCAT_PROJECT,
                       Primitive.PRODUCT):
        return np.ones_like(x)

    else:
        return np.ones_like(x)
