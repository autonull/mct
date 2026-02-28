"""
MCT5 Phase 2 — Dual-Signal Learning

Two complementary error signals are blended:

  1. LOCAL CONTRASTIVE GOODNESS (Forward-Forward inspired)
     ─────────────────────────────────────────────────────
     Each node computes goodness = ‖V_out‖².
     On positive examples: push goodness above threshold θ.
     On negative examples: push goodness below threshold θ.
     Weight update:
         δ_goodness = goodness - θ          (positive) or  θ - goodness  (negative)
         ΔW ≈ η · δ_goodness · V_out ⊗ V_in / (‖V_in‖ + ε)

  2. RETROGRADE ERROR SIGNAL
     ─────────────────────────────────────────────────────
     Output tension: T_v = (Y* − Ŷ) / √D, |T| = √MSE
     Propagated upstream in reverse topo order with proportional blame:
         w_blame_j = ‖V_in_j‖ / (Σ ‖V_in‖ + ε)
         T_upstream_j = T_current · w_blame_j · σ_spectral(W_i)
     Weight update (rank-1):
         ΔW ≈ η · T_local ⊗ V_in

     σ_spectral(W) replaces the hardcoded 0.5 factor, giving signal-preserving
     propagation regardless of graph depth.

  Combined:
      ΔW_total = λ_c · ΔW_local_contrastive + λ_r · ΔW_retrograde

  Health update on active path:
      Δρ = α(1 − |T|) − β(1 + |T|²)|T|·w_blame

  Routing signature update for every active node:
      S ← S + η_S · T_v · Re(⟨S, R⟩)

  Atrophy for all nodes (global):
      if steps_idle > 50: rho -= γ · steps_idle
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .types import GraphState, Node, NodeType, ActiveRecord
from .residue import HolographicResidue
from .primitives import primitive_derivative


class LearningEngine:
    """
    Dual-signal learning for MCT5.
    """

    def __init__(self,
                 state: GraphState,
                 residue: HolographicResidue,
                 eta_W: float = 0.01,
                 eta_S: float = 0.002,
                 alpha: float = 0.01,
                 beta: float = 0.05,
                 gamma: float = 0.001,
                 W_max: float = 8.0,
                 adam_beta1: float = 0.9,
                 adam_beta2: float = 0.999,
                 adam_eps: float = 1e-8,
                 lambda_contrastive: float = 0.7,
                 lambda_retrograde: float = 0.3,
                 goodness_threshold: float = 0.5):
        self.state = state
        self.residue = residue
        self.eta_W = eta_W
        self.eta_S = eta_S
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.W_max = W_max
        self.b1 = adam_beta1
        self.b2 = adam_beta2
        self.eps = adam_eps
        self.lam_c = lambda_contrastive
        self.lam_r = lambda_retrograde
        self.theta = goodness_threshold

        # Track edge tensions for structural evolution
        self.edge_tension: Dict[Tuple[int, int], float] = defaultdict(float)
        # EMA of loss for stagnation detection
        self._ema_loss: float = 1.0

    # ── Public API ────────────────────────────────────────────────────────────

    def learn(self, Y_hat: np.ndarray, Y_star: np.ndarray,
              output_node_id: int, is_positive: bool = True) -> float:
        """
        Execute the full learning phase.

        Args:
            Y_hat:          Actual output vector from forward pass
            Y_star:         Target output vector
            output_node_id: ID of the output node that produced Y_hat
            is_positive:    True = correct label (positive example)
                            False = deliberately wrong label (negative example)

        Returns:
            Scalar loss (MSE)
        """
        T_v, T_norm = self._compute_tension(Y_hat, Y_star)
        loss = T_norm ** 2

        # Retrograde pass → node tensions
        node_tensions = self._retrograde_flow(T_v, output_node_id)

        # Weight updates (both signals)
        self._update_weights(node_tensions, is_positive)

        # Health update (active path only)
        self._update_health(T_norm, node_tensions)

        # Routing signature update
        self._update_signatures(T_v)

        # Atrophy
        self._apply_atrophy()

        # EMA loss tracking
        self._ema_loss = 0.95 * self._ema_loss + 0.05 * loss

        return loss

    def max_tension_edge(self) -> Optional[Tuple[int, int]]:
        """Return edge with highest attributed tension (for structural evolution)."""
        if not self.edge_tension:
            return None
        return max(self.edge_tension.items(), key=lambda kv: kv[1])[0]

    def ema_loss(self) -> float:
        return self._ema_loss

    # ── Tension computation ───────────────────────────────────────────────────

    def _compute_tension(self, Y_hat: np.ndarray,
                         Y_star: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        T_v = (Y* − Ŷ) / √D
        |T| = √MSE(Y*, Ŷ)  clamped to [0, 1]
        """
        D = len(Y_hat)
        T_v = (Y_star - Y_hat) / np.sqrt(D)
        mse = float(np.mean((Y_star - Y_hat) ** 2))
        T_norm = float(np.clip(np.sqrt(mse), 0.0, 1.0))
        return T_v, T_norm

    # ── Retrograde flow ───────────────────────────────────────────────────────

    def _retrograde_flow(self, T_v: np.ndarray,
                         output_node_id: int) -> Dict[int, np.ndarray]:
        """
        Propagate tension backwards in reverse topological order.

        For each active node i:
          T_current  →  each upstream sender j with:
              w_blame_j = ‖V_in_j‖ / (Σ_k ‖V_in_k‖ + ε)
              T_upstream_j = T_current · w_blame_j · ‖W_i‖_spectral
        """
        active_path = self.state.active_path
        if not active_path:
            return {}

        # Sort descending by hop depth
        sorted_path = sorted(active_path, key=lambda r: -r.hop)
        node_tensions: Dict[int, np.ndarray] = {output_node_id: T_v}
        self.edge_tension = defaultdict(float)

        for record in sorted_path:
            nid = record.node_id
            if nid not in node_tensions:
                continue

            T_curr = node_tensions[nid]
            node = self.state.nodes.get(nid)
            if node is None:
                continue

            contribs = record.inbox_contributions
            if not contribs:
                continue

            # Compute blame weights
            norms = {sid: np.linalg.norm(v) for sid, v in contribs.items()}
            total_norm = sum(norms.values()) + 1e-9

            # Signal-preserving scaling factor
            spec = node.spectral_norm()
            # Clamp to prevent explosion/vanish: keep in (0.2, 5.0)
            signal_scale = float(np.clip(spec, 0.2, 5.0))

            for src_id, v_src in contribs.items():
                w_blame = norms[src_id] / total_norm
                T_upstream = T_curr * w_blame * signal_scale

                if src_id in node_tensions:
                    node_tensions[src_id] = node_tensions[src_id] + T_upstream
                else:
                    node_tensions[src_id] = T_upstream.copy()

                self.edge_tension[(src_id, nid)] = max(
                    self.edge_tension[(src_id, nid)],
                    float(np.linalg.norm(T_upstream))
                )

        return node_tensions

    # ── Weight updates ────────────────────────────────────────────────────────

    def _update_weights(self, node_tensions: Dict[int, np.ndarray],
                        is_positive: bool):
        """
        Apply dual-signal weight updates (contrastive + retrograde) via Adam.
        """
        for record in self.state.active_path:
            nid = record.node_id
            node = self.state.nodes.get(nid)
            if node is None or node.node_type == NodeType.INPUT:
                continue

            V_in = record.V_in
            V_out = record.V_out

            # ── Signal 1: Local contrastive (goodness) ────────────────────
            goodness = record.goodness
            theta = self.theta * float(np.dot(V_in, V_in) + 1e-9)  # scale by input norm
            delta_goodness = (goodness - theta) if is_positive else (theta - goodness)
            
            # Gradient of goodness w.r.t. W: ∂‖f(W V_in)‖² / ∂W = 2 V_out ⊙ f'(V_weighted) ⊗ V_in
            f_prime = primitive_derivative(node.primitive, record.V_weighted, V_out)
            dW_contrastive = np.outer(V_out * f_prime * delta_goodness, V_in)

            # ── Signal 2: Retrograde error ────────────────────────────────
            if nid in node_tensions:
                T_local = node_tensions[nid]
                dW_retrograde = np.outer(T_local * f_prime, V_in) # Also pass retrograde through primitive
            else:
                dW_retrograde = np.zeros((node.D, node.D))

            # Blend
            dW = self.lam_c * dW_contrastive + self.lam_r * dW_retrograde

            # Clip gradient
            g_norm = np.linalg.norm(dW, 'fro')
            if g_norm > 5.0:
                dW *= 5.0 / g_norm

            # Adam update on low-rank factors
            # Gradient w.r.t. A (holding B fixed): dW @ B
            # Gradient w.r.t. B (holding A fixed): dW.T @ A
            grad_A = dW @ node.B       # (D, r)
            grad_B = dW.T @ node.A     # (D, r)

            node.adam_t += 1
            t = node.adam_t
            b1, b2, eps = self.b1, self.b2, self.eps

            # Update A
            node.m_A = b1 * node.m_A + (1 - b1) * grad_A
            node.v_A = b2 * node.v_A + (1 - b2) * grad_A ** 2
            m_hat_A = node.m_A / (1 - b1 ** t)
            v_hat_A = node.v_A / (1 - b2 ** t)
            node.A += self.eta_W * m_hat_A / (np.sqrt(v_hat_A) + eps)

            # Update B
            node.m_B = b1 * node.m_B + (1 - b1) * grad_B
            node.v_B = b2 * node.v_B + (1 - b2) * grad_B ** 2
            m_hat_B = node.m_B / (1 - b1 ** t)
            v_hat_B = node.v_B / (1 - b2 ** t)
            node.B += self.eta_W * m_hat_B / (np.sqrt(v_hat_B) + eps)

            # Clamp Frobenius norm of effective W = A @ B.T
            # Approximate by clamping each factor
            for factor in [node.A, node.B]:
                fnorm = np.linalg.norm(factor, 'fro')
                if fnorm > self.W_max:
                    factor *= self.W_max / fnorm

            # Store tension attribution
            if nid in node_tensions:
                node.last_attributed_tension = float(np.linalg.norm(node_tensions[nid]))

    # ── Health update ─────────────────────────────────────────────────────────

    def _update_health(self, T_norm: float, node_tensions: Dict[int, np.ndarray]):
        """
        Δρ = α(1 − |T|) − β(1 + |T|²)|T|·w_blame,i
        """
        active_ids = {r.node_id for r in self.state.active_path}
        for record in self.state.active_path:
            nid = record.node_id
            node = self.state.nodes.get(nid)
            if node is None:
                continue

            w_blame = 1.0
            if nid in node_tensions and T_norm > 0:
                T_local_norm = float(np.linalg.norm(node_tensions[nid]))
                w_blame = min(1.0, T_local_norm / (T_norm + 1e-9))

            reward  = self.alpha * (1.0 - T_norm)
            penalty = self.beta  * (1.0 + T_norm ** 2) * T_norm * w_blame
            delta   = np.clip(reward - penalty, -0.1, 0.1)
            node.rho = float(np.clip(node.rho + delta, -10.0, 10.0))

    # ── Routing signature update ──────────────────────────────────────────────

    def _update_signatures(self, T_v: np.ndarray):
        """S_i ← S_i + η_S · T_v · Re(⟨S_i, R⟩)"""
        for record in self.state.active_path:
            node = self.state.nodes.get(record.node_id)
            if node is None or node.node_type == NodeType.INPUT:
                continue
            r_dot = float(np.real(np.dot(node.S, self.residue.R)))
            node.S = node.S + self.eta_S * T_v * r_dot
            # Re-normalise to unit sphere
            snorm = np.linalg.norm(node.S)
            if snorm > 1e-9:
                node.S /= snorm

    # ── Atrophy ───────────────────────────────────────────────────────────────

    def _apply_atrophy(self):
        """Penalise persistently idle nodes (health decay)."""
        gamma = self.gamma
        if self.state.is_converged():
            gamma *= 0.5

        for node in self.state.nodes.values():
            if node.steps_idle > 50:
                node.rho -= gamma * node.steps_idle
                node.rho = max(node.rho, -10.0)

            # Update tension EMA
            node.tension_trace *= 0.9
