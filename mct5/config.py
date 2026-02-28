from dataclasses import dataclass
import torch

@dataclass
class MCT5Config:
    # ── Dimensionality ────────────────────────────────────────────────────────
    D: int = 64
    r: int = 16
    input_dim: int = 2
    n_classes: int = 2
    device: str = "cpu"

    # ── Learning Rates ────────────────────────────────────────────────────────
    eta_W: float = 0.01      # AdamW lr for weights (A, B)
    eta_S: float = 0.005     # AdamW lr for routing signatures (S)
    weight_decay: float = 0.01

    # ── Execution ─────────────────────────────────────────────────────────────
    t_budget: int = 12       # Nominal anytime deadline
    sigma_max: float = 1.0   # Max active path length threshold
    lambda_tau: float = 1.0  # Threshold decay speed

    # ── Contrastive Goodness (Future hook, currently standard CE loss used) ───
    goodness_threshold: float = 0.5
    lambda_contrastive: float = 0.7
    lambda_retrograde: float = 0.3

    # ── Holographic Base ──────────────────────────────────────────────────────
    max_nodes: int = 1000    # Pre-allocated orthogonal vectors
    phi_max: float = 4.0     # Trace magnitude pruning threshold

    # ── Structural Evolution ──────────────────────────────────────────────────
    evolve_interval: int = 10
    sigma_mut: float = 0.05
    K: int = 2               # Nodes spawned per capacity insertion
    tau_lateral: float = 0.3 # Threshold for lateral edge creation
    quadratic_spawn_bias: float = 0.3
    
    def __post_init__(self):
        # Auto-detect CUDA if device not explicitly set
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
