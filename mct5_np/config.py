"""
MCT5 Configuration

Centralised hyperparameter dataclass.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MCT5Config:
    # ── Graph dimensionality ──────────────────────────────────────────────────
    D: int = 64             # Uniform vector dimension for all signals
    r: int = 16             # Low-rank factor dimension (W = A @ B.T,  A,B ∈ ℝᴰˣʳ)

    # ── Task metadata ─────────────────────────────────────────────────────────
    input_dim: int = 2      # Raw input feature dimension (padded/projected to D)
    n_classes: int = 2      # Number of output classes

    # ── Execution ─────────────────────────────────────────────────────────────
    t_budget: int = 15      # Max hop count per forward pass
    lambda_tau: float = 0.15  # Latency threshold steepness
    lambda_async: float = 0.1  # Inbox signal time-decay rate

    # ── Holographic Residue ───────────────────────────────────────────────────
    omega: float = 0.05     # Phase rotation speed (rad/hop)
    phi_max: float = 1.0    # Max Residue norm (relative to √D)
    residue_decay: float = 0.0  # Per-pass decay of R (0 = no decay, persistent)

    # ── Learning ──────────────────────────────────────────────────────────────
    eta_W: float = 0.01     # Weight learning rate (Adam)
    eta_S: float = 0.002    # Routing signature learning rate
    eta_rho: float = 0.001  # Health-signal learning rate (direct update)
    alpha: float = 0.01     # Health reward (catalysis)
    beta: float = 0.05      # Health penalty (solvent)
    gamma: float = 0.001    # Atrophy rate
    W_max: float = None     # Max weight Frobenius norm (defaults to √D)

    # Adam optimiser
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # Dual-signal blend  (contrastive + retrograde)
    lambda_contrastive: float = 0.7   # Weight on local goodness signal
    lambda_retrograde: float = 0.3    # Weight on retrograde error signal
    goodness_threshold: float = 0.5   # Target goodness for +/- examples

    # ── Structural evolution ──────────────────────────────────────────────────
    sigma_mut: float = 0.05     # Mutation noise std dev
    K: int = 2                  # Nodes spawned per lysis event
    evolve_interval: int = 10   # Steps between structural evolution passes
    tau_lateral: float = 0.3    # Tension threshold for lateral wiring
    kappa_thresh: int = 100     # Passes without lysis before dampening
    quadratic_spawn_bias: float = 0.3  # Extra probability of spawning QUADRATIC

    # ── Initialisation ────────────────────────────────────────────────────────
    rho_init: float = 1.0   # Initial node health
    W_init_scale: float = 0.1  # Std-dev for weight init noise around identity

    def __post_init__(self):
        if self.W_max is None:
            self.W_max = float(np.sqrt(self.D))
        assert self.lambda_contrastive + self.lambda_retrograde <= 1.01, \
            "Learning signal weights must sum to ≤1"
