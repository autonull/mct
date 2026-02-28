"""
MCT5 Unified Holographic Residue

Complex-valued phase-rotation memory for context-aware computation.

The residue R ∈ ℂ^D encodes structural sequence history through:
- Orthogonal basis vectors B_i assigned to each node
- Ghost injection on near-miss activations
- Phase-rotated superposition for temporal encoding
- Decoding provides activation boost: Re(⟨S_i, R⟩)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import numpy as np


class HolographicResidue(nn.Module):
    """
    Holographic Residue - Complex-valued context memory.
    
    This module implements a biologically-inspired memory mechanism where:
    1. Each node gets an orthogonal basis vector B_i
    2. Near-miss activations inject "ghost" signals into R
    3. Active nodes decode R to get context-dependent boost
    4. The memory persists across samples within a sequence/batch
    
    Mathematical formulation:
    - Ghost injection: R ← R + (ρ/√D) · B_i · e^(iωt)
    - Decoding: boost_i = Re(⟨S_i, R⟩)
    - Norm pruning: if ||R|| > φ_max, scale down
    """
    
    def __init__(
        self,
        D: int = 96,
        max_nodes: int = 2000,
        phi_max: float = 5.0,
        omega: float = 0.04,
        decay_rate: float = 0.0,
        device: str = "cpu"
    ):
        super().__init__()

        self.D = D
        self.max_nodes = max_nodes
        self.phi_max = phi_max
        self.omega = omega
        self.decay_rate = decay_rate
        self.device = device
        self._device = torch.device(device)
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPLEX-VALUED RESIDUE MEMORY
        # ═══════════════════════════════════════════════════════════════════
        
        # R ∈ ℂ^D - the holographic residue
        self.register_buffer("R", torch.zeros(D, dtype=torch.complex64, device=self._device))
        
        # Orthogonal basis matrix B ∈ ℝ^(max_nodes × D)
        # Each row is an orthogonal basis vector for a node
        self.register_buffer("B", torch.zeros(max_nodes, D, dtype=torch.float32, device=self._device))
        
        # Active node indices (which basis vectors are in use)
        self.active_indices: List[int] = []
        
        # Next available basis index
        self.next_idx: int = 0
        
        # Node ID → basis index mapping
        self.node_to_basis: Dict[int, int] = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # STATISTICS (for debugging/analysis)
        # ═══════════════════════════════════════════════════════════════════
        
        self.total_injections: int = 0
        self.total_decodes: int = 0
        self.prune_events: int = 0
    
    # ═══════════════════════════════════════════════════════════════════════
    # BASIS VECTOR MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def register_node(self, node_id: int) -> int:
        """
        Assign an orthogonal basis vector to a new node.
        
        Uses Gram-Schmidt orthogonalization against existing vectors.
        
        Args:
            node_id: Unique identifier for the node
        
        Returns:
            Internal basis index assigned to this node
        """
        # Check if already registered
        if node_id in self.node_to_basis:
            return self.node_to_basis[node_id]
        
        # Grow basis buffer if needed
        if self.next_idx >= self.B.size(0):
            self._grow_buffer()
        
        # Generate random vector and orthogonalize
        idx = self.next_idx
        v = torch.randn(self.D, device=self._device)
        
        # Gram-Schmidt against all active vectors
        if self.active_indices:
            active_B = self.B[self.active_indices]  # (num_active, D)
            # Projection: (B_active · v) * B_active for each active
            projections = (active_B @ v).unsqueeze(-1) * active_B  # (num_active, D)
            v = v - projections.sum(dim=0)
        
        # Normalize
        norm = v.norm()
        if norm < 1e-5:
            # Degenerate case - generate new random vector
            v = torch.randn(self.D, device=self._device)
        v = v / v.norm()
        
        # Store
        self.B[idx] = v
        self.active_indices.append(idx)
        self.node_to_basis[node_id] = idx
        self.next_idx += 1
        
        return idx
    
    def release_node(self, node_id: int):
        """
        Release a node's basis vector when it's pruned.
        
        The basis slot becomes available for reuse.
        """
        if node_id not in self.node_to_basis:
            return
        
        idx = self.node_to_basis[node_id]
        
        if idx in self.active_indices:
            self.active_indices.remove(idx)
        
        del self.node_to_basis[node_id]
        
        # Note: We don't zero out B[idx] to avoid recomputation
        # It will be overwritten when the slot is reused
    
    def _grow_buffer(self):
        """Double the basis buffer size."""
        new_size = self.B.size(0) * 2
        new_B = torch.zeros(new_size, self.D, dtype=torch.float32, device=self._device)
        new_B[:self.B.size(0)] = self.B
        self.B = new_B
    
    # ═══════════════════════════════════════════════════════════════════════
    # GHOST INJECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def inject_ghost(
        self,
        node_id: int,
        rho: float,
        t: float = 0.0
    ):
        """
        Inject a ghost signal from a near-miss activation.
        
        When a node fails to fire (ρ_active < τ), its potential is
        encoded into the residue as a phase-rotated contribution.
        
        Formula: R ← R + (ρ/√D) · B_i · e^(iωt)
        
        Args:
            node_id: The node that nearly fired
            rho: The activation potential (can be negative)
            t: Elapsed time/hop count for phase rotation
        """
        if node_id not in self.node_to_basis:
            return
        
        idx = self.node_to_basis[node_id]
        if idx < 0 or idx >= self.B.size(0):
            return
        
        # Get basis vector
        basis = self.B[idx].to(torch.complex64)  # (D,)
        
        # Compute phase rotation: e^(iωt) = cos(ωt) + i·sin(ωt)
        phase_angle = torch.tensor(self.omega * t, device=self._device)
        phase = torch.complex(
            torch.cos(phase_angle),
            torch.sin(phase_angle)
        )
        
        # Amplitude scaled by activation potential
        amplitude = rho / np.sqrt(self.D)
        
        # Contribution
        contrib = amplitude * basis * phase
        
        # Inject
        self.R = self.R + contrib
        self.total_injections += 1
        
        # Prune if necessary
        self._prune_norm()
    
    def inject_ghost_batch(
        self,
        node_id: int,
        rho_batch: torch.Tensor,
        t: float = 0.0
    ):
        """
        Inject ghost signals from a batch of near-miss activations.
        
        Args:
            node_id: The node
            rho_batch: Batch of activation potentials (B,)
            t: Time/hop count
        """
        if node_id not in self.node_to_basis:
            return
        
        idx = self.node_to_basis[node_id]
        if idx < 0 or idx >= self.B.size(0):
            return
        
        basis = self.B[idx].to(torch.complex64)  # (D,)
        
        phase_angle = torch.tensor(self.omega * t, device=self._device)
        phase = torch.complex(
            torch.cos(phase_angle),
            torch.sin(phase_angle)
        )
        
        # Average amplitude over batch
        avg_rho = rho_batch.mean()
        amplitude = avg_rho / np.sqrt(self.D)
        
        contrib = amplitude * basis * phase
        self.R = self.R + contrib
        self.total_injections += 1
        
        self._prune_norm()
    
    # ═══════════════════════════════════════════════════════════════════════
    # DECODING
    # ═══════════════════════════════════════════════════════════════════════
    
    def decode(self, node_id: int) -> torch.Tensor:
        """
        Decode context boost for a node.
        
        Computes: boost_i = Re(⟨S_i, R⟩)
        
        This provides a context-dependent activation boost based on
        the structural history encoded in R.
        
        Args:
            node_id: The node to decode for
        
        Returns:
            Scalar boost value (can be negative)
        """
        if node_id not in self.node_to_basis:
            return torch.tensor(0.0, device=self._device)
        
        idx = self.node_to_basis[node_id]
        if idx < 0 or idx >= self.B.size(0):
            return torch.tensor(0.0, device=self._device)
        
        self.total_decodes += 1
        
        basis = self.B[idx].to(torch.complex64)  # (D,)
        
        # Inner product and real part
        inner = torch.dot(basis, self.R)
        return torch.real(inner)
    
    def decode_batch(self, node_ids: List[int]) -> torch.Tensor:
        """
        Decode context boosts for multiple nodes.
        
        Args:
            node_ids: List of node IDs
        
        Returns:
            Tensor of boosts (N,)
        """
        boosts = []
        for nid in node_ids:
            boosts.append(self.decode(nid))
        return torch.stack(boosts)
    
    # ═══════════════════════════════════════════════════════════════════════
    # NORM MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def _prune_norm(self):
        """
        Keep residue norm bounded.
        
        If ||R|| > φ_max, scale down proportionally.
        This prevents unbounded growth from repeated injections.
        """
        norm = self.R.norm()
        
        if norm > self.phi_max:
            # Soft thresholding
            scale = self.phi_max / norm
            self.R = self.R * scale
            self.prune_events += 1
    
    def reset_norm(self):
        """Reset residue to zero norm."""
        self.R.zero_()
    
    # ═══════════════════════════════════════════════════════════════════════
    # SEQUENCE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def reset(self):
        """
        Reset residue at sequence/episode boundary.

        Call this between independent samples or at episode boundaries.
        """
        self.R.zero_()

    def _apply_decay(self):
        """
        Apply decay to residue.

        Call this at end of each forward pass if decay > 0.
        """
        if self.decay_rate > 0:
            self.R = self.R * (1 - self.decay_rate)

    def end_of_pass(self):
        """Call at end of forward pass."""
        self._apply_decay()
        self._prune_norm()
    
    # ═══════════════════════════════════════════════════════════════════════
    # INSPECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> dict:
        """Get residue statistics."""
        return {
            "norm": float(self.R.norm().item()),
            "active_bases": len(self.active_indices),
            "total_injections": self.total_injections,
            "total_decodes": self.total_decodes,
            "prune_events": self.prune_events,
            "real_mean": float(self.R.real.mean().item()),
            "imag_mean": float(self.R.imag.mean().item()),
        }
    
    def get_basis_similarity(self, node_id1: int, node_id2: int) -> float:
        """
        Compute similarity between two nodes' basis vectors.
        
        Should be near zero for orthogonal bases.
        """
        if node_id1 not in self.node_to_basis or node_id2 not in self.node_to_basis:
            return 0.0
        
        idx1 = self.node_to_basis[node_id1]
        idx2 = self.node_to_basis[node_id2]
        
        b1 = self.B[idx1]
        b2 = self.B[idx2]
        
        return float(torch.dot(b1, b2).item())
    
    def extra_repr(self) -> str:
        return f"D={self.D}, norm={self.R.norm().item():.3f}, bases={len(self.active_indices)}"


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_orthogonal_basis(
    n: int,
    D: int,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Create n orthogonal basis vectors in ℝ^D.
    
    Uses QR decomposition for numerical stability.
    
    Args:
        n: Number of vectors
        D: Dimension (must be >= n)
        device: torch device
    
    Returns:
        Orthogonal basis matrix (n, D)
    """
    assert D >= n, f"Cannot create {n} orthogonal vectors in {D} dimensions"
    
    # Random matrix
    M = torch.randn(n, D, device=device)
    
    # QR decomposition
    Q, _ = torch.linalg.qr(M)
    
    return Q
