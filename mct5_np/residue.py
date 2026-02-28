"""
MCT5 Holographic Residue

Complex-valued superposition memory for routing context.
Each node has a unique orthogonal basis vector; ghost signals are injected
as phase-rotated contributions so that they can be selectively decoded.

Math:
  R ∈ ℂᴰ
  B ∈ ℝᴺˣᴰ  — orthogonal basis matrix (Gram-Schmidt), N = max_nodes

  Ghost injection (node i fails to fire at elapsed hop t):
      R ← R + (ρᵢ / √D) · Bᵢ · exp(i·ω·t)

  Decode node i's contribution from R:
      contribution = Re(⟨Bᵢ, R⟩)   ∈ ℝ

  Norm pruning after each pass:
      If ‖R‖₂ > Φ_max, zero the lowest-magnitude components until within budget.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional


class HolographicResidue:
    """
    Complex-valued global context memory for MCT5.
    """

    def __init__(self,
                 D: int,
                 max_nodes: int = 512,
                 omega: float = 0.05,
                 phi_max: float = 1.0,
                 decay: float = 0.0):
        """
        Args:
            D:         Vector dimension
            max_nodes: Upper bound on total nodes (determines basis matrix size)
            omega:     Phase rotation rate (rad/hop)
            phi_max:   Max allowed ‖R‖₂ (relative units; actual = phi_max · √D)
            decay:     Per-pass multiplicative decay of R (0 = no decay)
        """
        self.D = D
        self.omega = omega
        self.phi_max_abs = phi_max * np.sqrt(D)
        self.decay = decay

        # Complex residue vector
        self.R: np.ndarray = np.zeros(D, dtype=complex)

        # Orthogonal basis matrix: rows are per-node basis vectors
        self._B_raw: np.ndarray = np.random.randn(max_nodes, D)
        self._B: np.ndarray = self._gram_schmidt(self._B_raw)  # Normalised rows

        # Slot-to-node mapping
        self._node_to_slot: Dict[int, int] = {}
        self._slot_to_node: Dict[int, int] = {}
        self._free_slots: List[int] = list(range(max_nodes))

    # ── Basis management ──────────────────────────────────────────────────────

    @staticmethod
    def _gram_schmidt(X: np.ndarray) -> np.ndarray:
        """Row-wise Gram-Schmidt orthonormalisation."""
        Q = np.zeros_like(X)
        for i, row in enumerate(X):
            v = row.copy()
            for j in range(i):
                v -= np.dot(v, Q[j]) * Q[j]
            norm = np.linalg.norm(v)
            Q[i] = v / norm if norm > 1e-12 else v
        return Q

    def register_node(self, node_id: int) -> bool:
        """Assign an orthogonal basis slot to a new node.  Returns True on success."""
        if node_id in self._node_to_slot:
            return True
        if not self._free_slots:
            # Approximate: reuse a random existing slot (minor interference)
            slot = np.random.randint(0, self._B.shape[0])
        else:
            slot = self._free_slots.pop(0)
        self._node_to_slot[node_id] = slot
        self._slot_to_node[slot] = node_id
        return True

    def release_node(self, node_id: int):
        """Free a node's basis slot on lysis."""
        slot = self._node_to_slot.pop(node_id, None)
        if slot is not None:
            self._slot_to_node.pop(slot, None)
            self._free_slots.append(slot)

    def basis_vec(self, node_id: int) -> np.ndarray:
        """Return the real-valued basis vector for node_id."""
        slot = self._node_to_slot.get(node_id)
        if slot is None:
            return np.zeros(self.D)
        return self._B[slot]

    # ── Residue operations ────────────────────────────────────────────────────

    def reset(self):
        """Clear residue at sequence boundary."""
        self.R[:] = 0.0

    def inject_ghost(self, node_id: int, rho: float, t_elapsed: int):
        """
        Inject a ghost signal from a node that failed to fire.

        R ← R + (ρ / √D) · Bᵢ · exp(i · ω · t)
        """
        B_i = self.basis_vec(node_id)
        phase = np.exp(1j * self.omega * t_elapsed)
        self.R += (rho / np.sqrt(self.D)) * B_i * phase

    def decode(self, node_id: int) -> float:
        """
        Extract this node's contribution from R.

        Returns Re(⟨Bᵢ, R⟩) — a real scalar.
        """
        B_i = self.basis_vec(node_id)
        return float(np.real(np.dot(B_i, self.R)))

    def end_of_pass(self):
        """
        Apply optional decay and norm pruning.
        Called once at the end of each forward pass.
        """
        # Optional decay
        if self.decay > 0:
            self.R *= (1.0 - self.decay)

        # Norm pruning: zero lowest-magnitude components
        norm = np.linalg.norm(self.R)
        if norm > self.phi_max_abs:
            # Sort by magnitude and zero smallest until within budget
            mags = np.abs(self.R)
            sorted_idx = np.argsort(mags)
            cumulative = 0.0
            # We need to remove enough energy to bring norm below phi_max
            # Approximate: zero components until enough is removed
            target_sq = self.phi_max_abs ** 2
            current_sq = norm ** 2
            for idx in sorted_idx:
                if current_sq <= target_sq:
                    break
                current_sq -= mags[idx] ** 2
                self.R[idx] = 0.0

    def get_activation_boost(self, node_id: int) -> float:
        """
        Convenience: activation boost from residue for a node.

        Returns Re(⟨Sᵢ, R⟩) approximated via the basis decode.
        The S·R contribution uses the basis vector as a proxy —
        the full dot is done in forward.py using S directly.
        """
        return self.decode(node_id)
