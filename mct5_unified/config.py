"""
MCT5 Unified Configuration

Centralized hyperparameters with intelligent defaults for breakthrough performance.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Literal
import torch


class LearningMode(Enum):
    """Learning strategy selection."""
    AUTOGRAD = auto()      # Standard PyTorch backprop (fast, reliable)
    DUAL_SIGNAL = auto()   # Local contrastive + retrograde (biologically plausible)
    HYBRID = auto()        # Autograd for weights, dual-signal for structure (best of both)


@dataclass
class MCT5Config:
    """
    MCT5 Unified Configuration.
    
    Intelligent defaults tuned for breakthrough performance across diverse tasks.
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # DIMENSIONALITY
    # ═══════════════════════════════════════════════════════════════════════
    
    D: int = 96                    # Hidden dimension (increased for capacity)
    r: int = 24                    # Low-rank factor dimension (D×r params per node)
    input_dim: int = 2             # Input feature dimension
    n_classes: int = 2             # Number of output classes
    
    # ═══════════════════════════════════════════════════════════════════════
    # LEARNING MODE & RATES
    # ═══════════════════════════════════════════════════════════════════════
    
    learning_mode: LearningMode = LearningMode.HYBRID
    
    # Weight learning rates
    eta_W: float = 0.015           # Base weight learning rate
    eta_S: float = 0.003           # Routing signature learning rate
    eta_rho: float = 0.002         # Health learning rate
    
    # Optimizer settings (AdamW)
    weight_decay: float = 0.02     # L2 regularization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    
    # Gradient clipping
    grad_clip: float = 5.0         # Max gradient norm
    
    # ═══════════════════════════════════════════════════════════════════════
    # DUAL-SIGNAL LEARNING (when mode != AUTOGRAD)
    # ═══════════════════════════════════════════════════════════════════════
    
    lambda_contrastive: float = 0.65   # Local goodness weight
    lambda_retrograde: float = 0.35    # Retrograde error weight
    goodness_threshold: float = 0.5    # Target goodness for +/- examples
    goodness_scale_input: bool = True  # Scale threshold by ||V_in||²
    
    # Retrograde propagation
    spectral_clamp_min: float = 0.3    # Min spectral norm scaling
    spectral_clamp_max: float = 4.0    # Max spectral norm scaling
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXECUTION (FORWARD PASS)
    # ═══════════════════════════════════════════════════════════════════════
    
    t_budget: int = 15             # Max hop count (anytime deadline)
    lambda_tau: float = 0.12       # Threshold steepness (lower = more permissive)
    lambda_async: float = 0.08     # Inbox decay rate
    
    # Firing dynamics
    rho_baseline: float = 1.0      # Base activation threshold
    residue_boost_scale: float = 1.0  # Scale for Re(⟨S, R⟩) boost
    
    # ═══════════════════════════════════════════════════════════════════════
    # HOLOGRAPHIC RESIDUE
    # ═══════════════════════════════════════════════════════════════════════
    
    max_nodes: int = 2000          # Max pre-allocated orthogonal vectors
    phi_max: float = 5.0           # Max residue norm (absolute)
    omega: float = 0.04            # Phase rotation speed (rad/hop)
    residue_decay: float = 0.0     # Per-pass decay (0 = persistent)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURAL EVOLUTION
    # ═══════════════════════════════════════════════════════════════════════
    
    evolve_interval: int = 8       # Steps between structural updates
    sigma_mut: float = 0.04        # Base mutation noise std
    sigma_mut_min: float = 0.01    # Min mutation noise
    sigma_mut_max: float = 0.15    # Max mutation noise
    K: int = 2                     # Nodes spawned per insertion
    
    # Adaptive mutation (key innovation)
    adaptive_mutation: bool = True     # Auto-adjust sigma_mut
    adaptation_sensitivity: float = 0.3  # How fast to adapt
    
    # Pruning
    prune_threshold: float = 0.0   # Remove nodes with rho < this
    min_nodes: int = 4             # Minimum graph size
    
    # Lateral wiring
    tau_lateral: float = 0.25      # Tension threshold for shortcuts
    lateral_wiring: bool = True    # Enable lateral shortcut growth
    
    # Spawn bias toward nonlinearity
    quadratic_spawn_bias: float = 0.35
    ensure_nonlinearity: bool = True  # Force QUADRATIC/PRODUCT on input path
    
    # Convergence detection
    kappa_thresh: int = 150        # Passes without pruning = converged
    
    # ═══════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    
    # Graph initialization
    init_hidden_primitive: str = "GELU"  # Default hidden node primitive
    init_rho_input: float = 3.0          # Input node health
    init_rho_hidden: float = 1.5         # Hidden node health
    init_rho_output: float = 2.0         # Output node health
    
    # Weight initialization
    W_init_scale: float = 0.08     # Std dev for A, B init
    
    # ═══════════════════════════════════════════════════════════════════════
    # REGULARIZATION & STABILITY
    # ═══════════════════════════════════════════════════════════════════════
    
    W_max: Optional[float] = None  # Max weight Frobenius norm (auto = √D)
    rho_clamp: tuple = (-5.0, 10.0)  # Health bounds
    
    # Loss tracking
    ema_loss_alpha: float = 0.95   # EMA smoothing factor
    stagnation_threshold: float = 0.005  # Delta for stagnation detection
    
    # ═══════════════════════════════════════════════════════════════════════
    # DEVICE & MISC
    # ═══════════════════════════════════════════════════════════════════════
    
    device: str = "auto"           # "auto", "cpu", or "cuda"
    seed: Optional[int] = None     # Random seed for reproducibility
    
    # Debug/verbose
    verbose: bool = False
    track_activations: bool = False  # Store all activations (memory intensive)
    
    def __post_init__(self):
        """Validate and compute derived defaults."""
        # Auto-detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        
        # Auto-compute W_max
        if self.W_max is None:
            self.W_max = float(self.D ** 0.5) * 2.0
        
        # Validate dual-signal weights
        signal_sum = self.lambda_contrastive + self.lambda_retrograde
        if abs(signal_sum - 1.0) > 0.01:
            # Normalize
            total = signal_sum
            self.lambda_contrastive /= total
            self.lambda_retrograde /= total
        
        # Validate dimensions
        assert self.r <= self.D, "Low-rank dimension r must be <= D"
        # Allow smaller D for testing
        assert self.D >= 16, "D should be at least 16"
        assert self.t_budget >= 3, "t_budget should be at least 3"
        
        # Clamp adaptive mutation bounds
        self.sigma_mut_min = max(0.001, self.sigma_mut_min)
        self.sigma_mut_max = min(0.5, self.sigma_mut_max)
        self.sigma_mut = max(self.sigma_mut_min, min(self.sigma_mut_max, self.sigma_mut))
    
    @property
    def device_torch(self) -> torch.device:
        """Get torch.device for the configured device."""
        return torch.device(self.device)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "MCT5Config":
        """Create config from dictionary."""
        if "learning_mode" in d and isinstance(d["learning_mode"], int):
            d = d.copy()
            d["learning_mode"] = LearningMode(d["learning_mode"])
        return cls(**d)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRESET CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    @classmethod
    def small(cls, **kwargs) -> "MCT5Config":
        """Small model for quick prototyping."""
        cfg = cls(D=48, r=12, **kwargs)
        return cfg
    
    @classmethod
    def medium(cls, **kwargs) -> "MCT5Config":
        """Medium model for standard tasks."""
        cfg = cls(D=96, r=24, **kwargs)
        return cfg
    
    @classmethod
    def large(cls, **kwargs) -> "MCT5Config":
        """Large model for complex tasks."""
        cfg = cls(D=192, r=48, max_nodes=5000, **kwargs)
        return cfg
    
    @classmethod
    def fast(cls, **kwargs) -> "MCT5Config":
        """Fast inference mode (smaller, less evolution)."""
        cfg = cls(
            D=64, r=16,
            evolve_interval=20,
            adaptive_mutation=False,
            t_budget=10,
            **kwargs
        )
        return cfg
    
    @classmethod
    def research(cls, **kwargs) -> "MCT5Config":
        """Research mode with full dual-signal learning."""
        cfg = cls(
            learning_mode=LearningMode.DUAL_SIGNAL,
            D=128, r=32,
            lambda_contrastive=0.7,
            lambda_retrograde=0.3,
            track_activations=True,
            **kwargs
        )
        return cfg
