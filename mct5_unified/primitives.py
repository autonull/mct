"""
MCT5 Unified Primitive Operators

16 primitive operators with proper derivatives for both autograd and manual learning.
"""

import torch
import torch.nn.functional as F
from enum import Enum, auto
from typing import Union, List, Optional
import math


class Primitive(Enum):
    """
    Primitive operators available for MCT5 nodes.
    
    Organized by arity and function:
    - Unary: Single input, various transformations
    - Binary/N-ary: Multiple inputs, combination/aggregation
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # UNARY PRIMITIVES
    # ═══════════════════════════════════════════════════════════════════════
    
    # Standard activations
    RELU = auto()        # max(0, x)
    GELU = auto()        # x · Φ(x) - smooth, good default
    TANH = auto()        # tanh(x) - bounded, zero-centered
    SWISH = auto()       # x · σ(x) - learned activation
    SILU = auto()        # x · σ(x) - same as swish, PyTorch native
    LEAKY_RELU = auto()  # max(αx, x) with α=0.01
    
    # Normalization
    SOFTMAX = auto()     # Stable softmax over last dim
    L2NORM = auto()      # x / ||x||₂
    
    # Signal processing
    FORK = auto()        # Identity/pass-through
    ABS = auto()         # |x| - enables sign-invariant features
    SIGN = auto()        # sign(x) · √|x| - signed sqrt
    SINE = auto()        # sin(x) - periodic, enables Fourier features
    
    # Nonlinear expansion (key for XOR-type problems)
    QUADRATIC = auto()   # x ⊙ x + x - creates squared features
    
    # ═══════════════════════════════════════════════════════════════════════
    # BINARY / N-ARY PRIMITIVES
    # ═══════════════════════════════════════════════════════════════════════
    
    ADD = auto()         # mean(inputs) - simple aggregation
    GATE = auto()        # x ⊙ σ(y) - gating mechanism
    BILINEAR = auto()    # (x·y/√D) · x - attention-like self-projection
    PRODUCT = auto()     # x ⊙ y - element-wise multiplication
    MAX = auto()         # element-wise max - enables logical OR
    CONCAT_PROJECT = auto()  # Concatenate and project back to D
    
    # Advanced
    ATTENTION_LITE = auto()  # Simplified self-attention
    ROUTED_SWITCH = auto()   # Router selects between inputs


# ═══════════════════════════════════════════════════════════════════════════
# PRIMITIVE APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def apply_primitive(
    primitive: Primitive,
    inputs: Union[torch.Tensor, List[torch.Tensor]]
) -> torch.Tensor:
    """
    Apply primitive operator to input(s).
    
    Args:
        primitive: The primitive operator to apply
        inputs: Single tensor for unary, list of tensors for binary/n-ary
    
    Returns:
        Output tensor of same shape as input (or broadcasted for n-ary)
    """
    
    # ───────────────────────────────────────────────────────────────────────
    # UNARY PRIMITIVES
    # ───────────────────────────────────────────────────────────────────────
    
    if primitive in [Primitive.RELU, Primitive.GELU, Primitive.TANH, 
                     Primitive.SWISH, Primitive.SILU, Primitive.LEAKY_RELU,
                     Primitive.FORK, Primitive.ABS, Primitive.SIGN, 
                     Primitive.SINE, Primitive.QUADRATIC]:
        
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        
        if primitive is Primitive.RELU:
            return F.relu(x)
        
        elif primitive is Primitive.GELU:
            return F.gelu(x)
        
        elif primitive is Primitive.TANH:
            return torch.tanh(x)
        
        elif primitive is Primitive.SWISH:
            return x * torch.sigmoid(x)
        
        elif primitive is Primitive.SILU:
            return F.silu(x)
        
        elif primitive is Primitive.LEAKY_RELU:
            return F.leaky_relu(x, negative_slope=0.01)
        
        elif primitive is Primitive.FORK:
            return x
        
        elif primitive is Primitive.ABS:
            return torch.abs(x)
        
        elif primitive is Primitive.SIGN:
            return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-9)
        
        elif primitive is Primitive.SINE:
            return torch.sin(x)
        
        elif primitive is Primitive.QUADRATIC:
            # x² + x provides both nonlinearity and gradient flow
            return x * x + x
    
    # ───────────────────────────────────────────────────────────────────────
    # NORMALIZATION PRIMITIVES
    # ───────────────────────────────────────────────────────────────────────
    
    elif primitive in [Primitive.SOFTMAX, Primitive.L2NORM]:
        
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        
        if primitive is Primitive.SOFTMAX:
            return F.softmax(x, dim=-1)
        
        elif primitive is Primitive.L2NORM:
            norm = x.norm(p=2, dim=-1, keepdim=True)
            return x / (norm + 1e-9)
    
    # ───────────────────────────────────────────────────────────────────────
    # BINARY / N-ARY PRIMITIVES
    # ───────────────────────────────────────────────────────────────────────
    
    elif primitive in [Primitive.ADD, Primitive.GATE, Primitive.BILINEAR,
                       Primitive.PRODUCT, Primitive.MAX, Primitive.CONCAT_PROJECT,
                       Primitive.ATTENTION_LITE, Primitive.ROUTED_SWITCH]:
        
        # Ensure list format
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        
        if len(inputs) == 0:
            return torch.zeros(1, device=inputs[0].device if inputs else None)
        
        if len(inputs) == 1:
            # Single input - apply unary fallback
            if primitive is Primitive.ADD:
                return inputs[0]
            elif primitive is Primitive.PRODUCT:
                return inputs[0] * inputs[0]
            elif primitive is Primitive.MAX:
                return inputs[0]
            else:
                return inputs[0]
        
        # Two or more inputs
        if primitive is Primitive.ADD:
            # Mean aggregation for n-ary
            return torch.stack(inputs, dim=0).mean(dim=0)
        
        elif primitive is Primitive.GATE:
            # x ⊙ σ(y), cycle through if more than 2
            x, y = inputs[0], inputs[1]
            result = x * torch.sigmoid(y)
            for z in inputs[2:]:
                result = result * torch.sigmoid(z)
            return result
        
        elif primitive is Primitive.BILINEAR:
            # (x·y/√D) · x - attention-like
            x, y = inputs[0], inputs[1]
            D = x.shape[-1]
            scale = math.sqrt(D)
            similarity = (x * y).sum(dim=-1, keepdim=True) / scale
            result = similarity * x
            # For n>2, average the results
            for z in inputs[2:]:
                sim = (x * z).sum(dim=-1, keepdim=True) / scale
                result = result + sim * x
            return result / len(inputs)
        
        elif primitive is Primitive.PRODUCT:
            # Element-wise product
            result = inputs[0] * inputs[1]
            for z in inputs[2:]:
                result = result * z
            return result
        
        elif primitive is Primitive.MAX:
            # Element-wise maximum
            result = torch.max(inputs[0], inputs[1])
            for z in inputs[2:]:
                result = torch.max(result, z)
            return result
        
        elif primitive is Primitive.CONCAT_PROJECT:
            # Concatenate along new dim, mean pool back to D
            stacked = torch.stack(inputs, dim=0)  # (N, B, D)
            return stacked.mean(dim=0)
        
        elif primitive is Primitive.ATTENTION_LITE:
            # Simplified self-attention: weighted sum based on pairwise similarity
            N = len(inputs)
            D = inputs[0].shape[-1]
            scale = math.sqrt(D)
            
            # Compute attention weights
            weights = []
            for i, xi in enumerate(inputs):
                # Similarity to mean
                mean = torch.stack(inputs, dim=0).mean(dim=0)
                sim = (xi * mean).sum(dim=-1, keepdim=True) / scale
                weights.append(sim)
            
            # Softmax over inputs
            weight_tensor = torch.stack(weights, dim=0)  # (N, B, 1)
            attn = F.softmax(weight_tensor, dim=0)
            
            # Weighted sum
            stacked = torch.stack(inputs, dim=0)  # (N, B, D)
            return (attn * stacked).sum(dim=0)
        
        elif primitive is Primitive.ROUTED_SWITCH:
            # First input as router, rest as candidates
            router = torch.sigmoid(inputs[0])  # (B, D)
            candidates = inputs[1:]
            if len(candidates) == 1:
                return router * candidates[0]
            # Weighted combination
            stacked = torch.stack(candidates, dim=0)
            return (router.unsqueeze(0) * stacked).sum(dim=0)
    
    # Fallback
    raise ValueError(f"Unknown primitive: {primitive}")


# ═══════════════════════════════════════════════════════════════════════════
# PRIMITIVE DERIVATIVES (for dual-signal learning)
# ═══════════════════════════════════════════════════════════════════════════

def primitive_derivative(
    primitive: Primitive,
    V_weighted: torch.Tensor,
    V_out: torch.Tensor
) -> torch.Tensor:
    """
    Compute derivative of primitive for manual learning updates.
    
    This is f'(V_weighted) where V_out = f(V_weighted).
    
    Used in dual-signal learning mode for computing weight gradients.
    In autograd mode, PyTorch handles this automatically.
    
    Args:
        primitive: The primitive operator
        V_weighted: Pre-activation input to primitive
        V_out: Post-activation output from primitive
    
    Returns:
        Gradient tensor of same shape as V_weighted
    """
    
    if primitive in [Primitive.RELU, Primitive.GELU, Primitive.TANH,
                     Primitive.SWISH, Primitive.SILU, Primitive.LEAKY_RELU,
                     Primitive.FORK, Primitive.ABS, Primitive.SIGN,
                     Primitive.SINE, Primitive.QUADRATIC,
                     Primitive.SOFTMAX, Primitive.L2NORM]:
        
        x = V_weighted
        
        if primitive is Primitive.RELU:
            return (x > 0).float()
        
        elif primitive is Primitive.GELU:
            # Approximate derivative of GELU
            return torch.sigmoid(x) + x * torch.sigmoid(x) * (1 - torch.sigmoid(x))
        
        elif primitive is Primitive.TANH:
            return 1 - torch.tanh(x) ** 2
        
        elif primitive is Primitive.SWISH:
            sig = torch.sigmoid(x)
            return sig + x * sig * (1 - sig)
        
        elif primitive is Primitive.SILU:
            sig = torch.sigmoid(x)
            return sig + x * sig * (1 - sig)
        
        elif primitive is Primitive.LEAKY_RELU:
            return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, 0.01))
        
        elif primitive is Primitive.FORK:
            return torch.ones_like(x)
        
        elif primitive is Primitive.ABS:
            return torch.sign(x)
        
        elif primitive is Primitive.SIGN:
            # Derivative of sign(x) * sqrt(|x|)
            return torch.sign(x) * 0.5 / torch.sqrt(torch.abs(x) + 1e-9)
        
        elif primitive is Primitive.SINE:
            return torch.cos(x)
        
        elif primitive is Primitive.QUADRATIC:
            return 2 * x + 1
        
        elif primitive is Primitive.SOFTMAX:
            # Jacobian is complex; use diagonal approximation
            return V_out * (1 - V_out)
        
        elif primitive is Primitive.L2NORM:
            # Approximate gradient
            norm = x.norm(p=2, dim=-1, keepdim=True) + 1e-9
            return (1 / norm) - (x ** 2) / (norm ** 3)
    
    # For binary/n-ary primitives, return ones (identity-like gradient)
    return torch.ones_like(V_weighted)


# ═══════════════════════════════════════════════════════════════════════════
# PRIMITIVE METADATA
# ═══════════════════════════════════════════════════════════════════════════

PRIMITIVE_METADATA = {
    # name: (is_unary, enables_nonlinearity, computational_cost)
    Primitive.RELU: (True, True, 1),
    Primitive.GELU: (True, True, 3),
    Primitive.TANH: (True, True, 2),
    Primitive.SWISH: (True, True, 3),
    Primitive.SILU: (True, True, 3),
    Primitive.LEAKY_RELU: (True, True, 1),
    Primitive.SOFTMAX: (True, False, 3),
    Primitive.L2NORM: (True, False, 2),
    Primitive.FORK: (True, False, 0),
    Primitive.ABS: (True, True, 1),
    Primitive.SIGN: (True, True, 2),
    Primitive.SINE: (True, True, 3),
    Primitive.QUADRATIC: (True, True, 2),
    Primitive.ADD: (False, False, 1),
    Primitive.GATE: (False, True, 2),
    Primitive.BILINEAR: (False, True, 3),
    Primitive.PRODUCT: (False, True, 1),
    Primitive.MAX: (False, True, 1),
    Primitive.CONCAT_PROJECT: (False, False, 2),
    Primitive.ATTENTION_LITE: (False, True, 4),
    Primitive.ROUTED_SWITCH: (False, True, 3),
}


def get_primitive_info(primitive: Primitive) -> dict:
    """Get metadata about a primitive."""
    meta = PRIMITIVE_METADATA.get(primitive, (True, False, 1))
    return {
        "name": primitive.name,
        "is_unary": meta[0],
        "enables_nonlinearity": meta[1],
        "computational_cost": meta[2],
    }


def select_primitive_for_task(
    task_type: str,
    diversity: float = 0.3
) -> Primitive:
    """
    Select a primitive based on task type with exploration.
    
    Args:
        task_type: "classification", "regression", "sequence", etc.
        diversity: Probability of random selection vs informed choice
    
    Returns:
        Selected primitive
    """
    import random
    
    if random.random() < diversity:
        # Random exploration
        return random.choice(list(Primitive))
    
    # Informed selection
    if task_type == "classification":
        # GELU/SWISH work well for classification
        return random.choice([Primitive.GELU, Primitive.SWISH, Primitive.RELU])
    
    elif task_type == "xor_like":
        # Need quadratic/product for XOR
        return random.choice([Primitive.QUADRATIC, Primitive.PRODUCT, Primitive.GATE])
    
    elif task_type == "sequence":
        # Gating helps with information flow
        return random.choice([Primitive.GELU, Primitive.GATE, Primitive.SILU])
    
    else:
        # Default
        return Primitive.GELU
