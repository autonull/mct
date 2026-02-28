import torch
import torch.nn.functional as F
from enum import Enum, auto
from typing import Union, List

class Primitive(Enum):
    # ── Unary ────────────────────────────────────────────────────────────────
    RELU       = auto()  # max(0, x)
    GELU       = auto()  # x · Φ(x)
    TANH       = auto()  # tanh(x)
    SWISH      = auto()  # x · σ(x)
    SOFTMAX    = auto()  # stable softmax
    L2NORM     = auto()  # x / ||x||
    FORK       = auto()  # identity
    QUADRATIC  = auto()  # x ⊙ x + x
    # ── Binary/N-ary ─────────────────────────────────────────────────────────
    ADD              = auto()  # mean(inputs)
    GATE             = auto()  # x ⊙ σ(y)
    BILINEAR         = auto()  # dot(x,y)/√D · x
    CONCAT_PROJECT   = auto()  # mean pool (learned projection in weights)
    PRODUCT          = auto()  # x ⊙ y

def apply_primitive(primitive: Primitive, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """
    Apply primitive operator to input(s).
    Inputs are torch Tensors of shape (B, D).
    """
    # ── Unary
    if primitive is Primitive.RELU:
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return F.relu(x)
    elif primitive is Primitive.GELU:
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return F.gelu(x)
    elif primitive is Primitive.TANH:
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return torch.tanh(x)
    elif primitive is Primitive.SWISH:
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return x * torch.sigmoid(x)
    elif primitive is Primitive.SOFTMAX:
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return F.softmax(x, dim=-1)
    elif primitive is Primitive.L2NORM:
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return F.normalize(x, p=2, dim=-1)
    elif primitive is Primitive.FORK:
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return x
    elif primitive is Primitive.QUADRATIC:
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return x * x + x

    # ── Binary / N-ary
    elif primitive is Primitive.ADD:
        if isinstance(inputs, list):
            return torch.stack(inputs, dim=0).mean(dim=0)
        return inputs
    elif primitive is Primitive.GATE:
        if isinstance(inputs, list) and len(inputs) >= 2:
            return inputs[0] * torch.sigmoid(inputs[1])
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return x
    elif primitive is Primitive.BILINEAR:
        if isinstance(inputs, list) and len(inputs) >= 2:
            X, Y = inputs[0], inputs[1]
            D = X.shape[-1]
            s = (X * Y).sum(dim=-1, keepdim=True) / (torch.sqrt(torch.tensor(D, dtype=torch.float32, device=X.device)) + 1e-9)
            return s * X
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return x
    elif primitive is Primitive.CONCAT_PROJECT:
        if isinstance(inputs, list) and len(inputs) >= 2:
            return torch.stack(inputs, dim=0).mean(dim=0)
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return x
    elif primitive is Primitive.PRODUCT:
        if isinstance(inputs, list) and len(inputs) >= 2:
            return inputs[0] * inputs[1]
        x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        return x * x
    else:
        raise ValueError(f"Unknown primitive: {primitive}")
