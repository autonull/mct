"""
MCT5 Unified Learning Engine

Phase 2: Three learning modes for different use cases:

1. AUTOGRAD - Standard PyTorch backpropagation
   - Fast, reliable, production-ready
   - Exact gradients via autograd
   - Best for maximum accuracy

2. DUAL_SIGNAL - Biologically plausible local learning
   - Local contrastive goodness (Forward-Forward inspired)
   - Retrograde error propagation (backprop-like without graph)
   - Best for research, biological plausibility

3. HYBRID - Best of both worlds
   - Autograd for weight updates
   - Dual-signal for structure evaluation
   - Best overall performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

from .types import GraphState, Node, NodeType, ActiveRecord
from .primitives import Primitive, primitive_derivative
from .residue import HolographicResidue
from .config import LearningMode


class LearningEngine(nn.Module):
    """
    Unified learning engine supporting all three modes.
    
    Delegates to specialized engines based on configuration.
    """
    
    def __init__(
        self,
        state: GraphState,
        residue: HolographicResidue,
        learning_mode: LearningMode = LearningMode.HYBRID,
        **kwargs
    ):
        super().__init__()
        
        self.state = state
        self.residue = residue
        self.learning_mode = learning_mode
        
        # Create specialized engines
        self.autograd_engine = AutogradLearning(state, **kwargs)
        self.dual_signal_engine = DualSignalLearning(state, residue, **kwargs)
        self.hybrid_engine = HybridLearning(state, residue, **kwargs)
        
        # EMA loss tracking
        self.register_buffer("ema_loss", torch.tensor(1.0))
        self.ema_alpha = kwargs.get("ema_loss_alpha", 0.95)

    def learn(
        self,
        Y_hat: torch.Tensor,
        Y_star: torch.Tensor,
        output_node_id: int,
        is_positive: bool = True,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Execute learning phase.

        Args:
            Y_hat: Model output (B, D) or (D,)
            Y_star: Target output (B, D) or (D,)
            output_node_id: ID of output node
            is_positive: True for positive example, False for negative
            labels: Optional class labels for cross-entropy

        Returns:
            Loss scalar
        """
        # Select engine based on mode
        if self.learning_mode == LearningMode.AUTOGRAD:
            loss = self.autograd_engine.learn(
                Y_hat, Y_star, output_node_id, is_positive, labels
            )
        elif self.learning_mode == LearningMode.DUAL_SIGNAL:
            loss = self.dual_signal_engine.learn(
                Y_hat, Y_star, output_node_id, is_positive
            )
        else:  # HYBRID
            loss = self.hybrid_engine.learn(
                Y_hat, Y_star, output_node_id, is_positive, labels
            )

        # Update EMA loss
        self.ema_loss = self.ema_alpha * self.ema_loss + (1 - self.ema_alpha) * loss

        return loss

    # Alias for torch compatibility
    forward = learn
    
    def get_max_tension_edge(self) -> Optional[Tuple[int, int]]:
        """Get edge with highest attributed tension."""
        if self.learning_mode == LearningMode.AUTOGRAD:
            return self.autograd_engine.get_max_tension_edge()
        elif self.learning_mode == LearningMode.DUAL_SIGNAL:
            return self.dual_signal_engine.get_max_tension_edge()
        else:
            return self.hybrid_engine.get_max_tension_edge()
    
    def get_stats(self) -> dict:
        """Get learning statistics."""
        if self.learning_mode == LearningMode.AUTOGRAD:
            return self.autograd_engine.get_stats()
        elif self.learning_mode == LearningMode.DUAL_SIGNAL:
            return self.dual_signal_engine.get_stats()
        else:
            return self.hybrid_engine.get_stats()


# ═══════════════════════════════════════════════════════════════════════════
# AUTOGRAD LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class AutogradLearning(nn.Module):
    """
    Standard PyTorch backpropagation learning.
    
    Features:
    - Exact gradients via autograd
    - Cross-entropy or MSE loss
    - Gradient clipping for stability
    """
    
    def __init__(
        self,
        state: GraphState,
        grad_clip: float = 5.0,
        weight_decay: float = 0.02,
        **kwargs
    ):
        super().__init__()
        
        self.state = state
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        
        # Edge tension tracking (approximate via gradient magnitude)
        self.edge_tensions: Dict[Tuple[int, int], float] = {}
    
    def learn(
        self,
        Y_hat: torch.Tensor,
        Y_star: torch.Tensor,
        output_node_id: int,
        is_positive: bool = True,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss - gradients handled by PyTorch autograd.
        
        In autograd mode, this just computes the loss.
        The actual gradient computation happens in the main training loop.
        """
        if labels is not None:
            # Cross-entropy loss for classification
            # Y_hat should be logits (B, n_classes)
            loss = F.cross_entropy(Y_hat, labels)
        else:
            # MSE loss
            loss = F.mse_loss(Y_hat, Y_star)
        
        # Track edge tensions based on gradient flow
        self._track_tensions()
        
        return loss
    
    def _track_tensions(self):
        """Track edge tensions from gradient magnitudes."""
        self.edge_tensions.clear()
        
        for record in self.state.active_path:
            node = self.state.get_node(record.node_id)
            if node is None or node.node_type == NodeType.INPUT:
                continue
            
            # Approximate tension from gradient norm
            if node.A.grad is not None:
                grad_norm = node.A.grad.norm().item()
                
                # Attribute to incoming edges
                for src_id in record.inbox_contributions.keys():
                    edge = (src_id, record.node_id)
                    self.edge_tensions[edge] = self.edge_tensions.get(edge, 0) + grad_norm
    
    def get_max_tension_edge(self) -> Optional[Tuple[int, int]]:
        """Get edge with highest tension."""
        if not self.edge_tensions:
            return None
        return max(self.edge_tensions.items(), key=lambda x: x[1])[0]
    
    def get_stats(self) -> dict:
        return {
            "mode": "autograd",
            "edge_tensions": len(self.edge_tensions),
            "grad_clip": self.grad_clip,
        }


# ═══════════════════════════════════════════════════════════════════════════
# DUAL-SIGNAL LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class DualSignalLearning(nn.Module):
    """
    Biologically plausible dual-signal local learning.
    
    Two complementary signals:
    
    1. LOCAL CONTRASTIVE GOODNESS (70%)
       - Positive examples: push goodness ‖V_out‖² above threshold
       - Negative examples: push goodness below threshold
       - Update: ΔW ∝ (goodness - θ) · V_out ⊗ V_in
    
    2. RETROGRADE ERROR (30%)
       - Output tension: T = (Y* - Ŷ) / √D
       - Upstream propagation with spectral-norm scaling
       - Proportional blame attribution
       - Update: ΔW ∝ T_local ⊗ V_in
    
    Combined: ΔW = λ_c · ΔW_contrastive + λ_r · ΔW_retrograde
    """
    
    def __init__(
        self,
        state: GraphState,
        residue: HolographicResidue,
        eta_W: float = 0.015,
        eta_S: float = 0.003,
        lambda_contrastive: float = 0.65,
        lambda_retrograde: float = 0.35,
        goodness_threshold: float = 0.5,
        goodness_scale_input: bool = True,
        spectral_clamp_min: float = 0.3,
        spectral_clamp_max: float = 4.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        W_max: float = None,
        **kwargs
    ):
        super().__init__()
        
        self.state = state
        self.residue = residue
        
        # Learning rates
        self.eta_W = eta_W
        self.eta_S = eta_S
        
        # Dual-signal blend
        self.lambda_c = lambda_contrastive
        self.lambda_r = lambda_retrograde
        
        # Goodness parameters
        self.theta = goodness_threshold
        self.goodness_scale_input = goodness_scale_input
        
        # Spectral norm clamping
        self.spec_min = spectral_clamp_min
        self.spec_max = spectral_clamp_max
        
        # Adam parameters
        self.b1 = adam_beta1
        self.b2 = adam_beta2
        self.eps = adam_eps
        
        # Weight clamping
        self.W_max = W_max if W_max else np.sqrt(state.D) * 2
        
        # Edge tension tracking
        self.edge_tensions: Dict[Tuple[int, int], float] = {}
    
    def learn(
        self,
        Y_hat: torch.Tensor,
        Y_star: torch.Tensor,
        output_node_id: int,
        is_positive: bool = True
    ) -> torch.Tensor:
        """
        Execute dual-signal learning.
        
        Args:
            Y_hat: Model output (B, D) or (D,)
            Y_star: Target output (B, D) or (D,)
            output_node_id: ID of output node
            is_positive: True for positive, False for negative
        
        Returns:
            MSE loss scalar
        """
        # Ensure 2D
        if Y_hat.dim() == 1:
            Y_hat = Y_hat.unsqueeze(0)
            Y_star = Y_star.unsqueeze(0)
        
        B, D = Y_hat.shape
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPUTE OUTPUT TENSION
        # ═══════════════════════════════════════════════════════════════════
        
        # T_v = (Y* - Ŷ) / √D
        T_v = (Y_star - Y_hat) / np.sqrt(D)  # (B, D)
        
        # |T| = √MSE, clamped to [0, 1]
        mse = ((Y_star - Y_hat) ** 2).mean()
        T_norm = torch.clamp(torch.sqrt(mse), 0.0, 1.0)
        
        loss = mse  # Return MSE as loss
        
        # ═══════════════════════════════════════════════════════════════════
        # RETROGRADE FLOW - Propagate tension upstream
        # ═══════════════════════════════════════════════════════════════════
        
        node_tensions = self._retrograde_flow(T_v, output_node_id)
        
        # ═══════════════════════════════════════════════════════════════════
        # WEIGHT UPDATES - Dual signal blend
        # ═══════════════════════════════════════════════════════════════════
        
        self._update_weights(node_tensions, is_positive)
        
        # ═══════════════════════════════════════════════════════════════════
        # HEALTH UPDATE
        # ═══════════════════════════════════════════════════════════════════
        
        self._update_health(T_norm, node_tensions, is_positive)
        
        # ═══════════════════════════════════════════════════════════════════
        # ROUTING SIGNATURE UPDATE
        # ═══════════════════════════════════════════════════════════════════
        
        self._update_signatures(T_v)
        
        return loss
    
    def _retrograde_flow(
        self,
        T_v: torch.Tensor,
        output_node_id: int
    ) -> Dict[int, torch.Tensor]:
        """
        Propagate tension backwards in reverse topological order.
        
        For each node i:
          T_current → upstream senders j with:
            w_blame_j = ||V_in_j|| / Σ||V_in||
            T_upstream_j = T_current · w_blame_j · σ_spectral(W_i)
        """
        active_path = self.state.active_path
        if not active_path:
            return {}
        
        # Sort by hop descending (reverse topo order)
        sorted_path = sorted(active_path, key=lambda r: -r.hop)
        
        node_tensions: Dict[int, torch.Tensor] = {output_node_id: T_v}
        self.edge_tensions.clear()
        
        for record in sorted_path:
            nid = record.node_id
            if nid not in node_tensions:
                continue
            
            node = self.state.get_node(nid)
            if node is None:
                continue
            
            contribs = record.inbox_contributions
            if not contribs:
                continue
            
            T_curr = node_tensions[nid]  # (B, D)
            
            # Compute blame weights based on input norms
            norms = {
                sid: v.norm(dim=-1).mean() 
                for sid, v in contribs.items()
            }
            total_norm = sum(norms.values()) + 1e-9
            
            # Signal-preserving spectral scaling
            spec_norm = node.spectral_norm()
            signal_scale = torch.clamp(
                torch.tensor(spec_norm, device=T_curr.device),
                self.spec_min, self.spec_max
            )
            
            for src_id, v_src in contribs.items():
                w_blame = norms[src_id] / total_norm
                T_upstream = T_curr * w_blame * signal_scale
                
                # Accumulate tension
                if src_id in node_tensions:
                    node_tensions[src_id] = node_tensions[src_id] + T_upstream
                else:
                    node_tensions[src_id] = T_upstream.clone()
                
                # Track edge tension
                edge = (src_id, nid)
                edge_tension = T_upstream.norm(dim=-1).mean().item()
                self.edge_tensions[edge] = max(
                    self.edge_tensions.get(edge, 0),
                    edge_tension
                )
        
        return node_tensions
    
    def _update_weights(
        self,
        node_tensions: Dict[int, torch.Tensor],
        is_positive: bool
    ):
        """Apply dual-signal weight updates via Adam."""
        
        for record in self.state.active_path:
            nid = record.node_id
            node = self.state.get_node(nid)
            
            if node is None or node.node_type == NodeType.INPUT:
                continue
            
            V_in = record.V_in  # (B, D)
            V_out = record.V_out  # (B, D)
            V_weighted = record.V_weighted
            
            # ═══════════════════════════════════════════════════════════════
            # SIGNAL 1: LOCAL CONTRASTIVE GOODNESS
            # ═══════════════════════════════════════════════════════════════
            
            goodness = record.goodness
            
            # Scale threshold by input norm
            if self.goodness_scale_input:
                input_norm_sq = (V_in ** 2).sum(dim=-1).mean()
                theta = self.theta * (input_norm_sq.item() + 1e-9)
            else:
                theta = self.theta
            
            # Delta goodness: how far from target
            if is_positive:
                delta_goodness = goodness - theta  # Push up
            else:
                delta_goodness = theta - goodness  # Push down
            
            # Gradient of goodness w.r.t. W
            f_prime = primitive_derivative(node.primitive, V_weighted, V_out)
            
            # dW_contrastive = δ_goodness · V_out ⊙ f' ⊗ V_in (averaged over batch)
            # For efficiency: outer product of means
            V_out_grad = (V_out * f_prime * delta_goodness).mean(dim=0)  # (D,)
            V_in_mean = V_in.mean(dim=0)  # (D,)
            dW_contrastive = torch.outer(V_out_grad, V_in_mean)  # (D, D)
            
            # ═══════════════════════════════════════════════════════════════
            # SIGNAL 2: RETROGRADE ERROR
            # ═══════════════════════════════════════════════════════════════
            
            if nid in node_tensions:
                T_local = node_tensions[nid]  # (B, D)
                T_mean = T_local.mean(dim=0)  # (D,)
                
                # dW_retrograde = T_local ⊗ V_in
                dW_retrograde = torch.outer(T_mean * f_prime.mean(dim=0), V_in_mean)
            else:
                dW_retrograde = torch.zeros_like(dW_contrastive)
            
            # ═══════════════════════════════════════════════════════════════
            # BLEND SIGNALS
            # ═══════════════════════════════════════════════════════════════
            
            dW = self.lambda_c * dW_contrastive + self.lambda_r * dW_retrograde
            
            # Clip gradient
            g_norm = dW.norm(p='fro')
            if g_norm > 5.0:
                dW = dW * (5.0 / g_norm)
            
            # ═══════════════════════════════════════════════════════════════
            # ADAM UPDATE ON LOW-RANK FACTORS
            # ═══════════════════════════════════════════════════════════════
            
            # Gradient w.r.t. A (holding B fixed): dW @ B
            # Gradient w.r.t. B (holding A fixed): dW.T @ A
            grad_A = dW @ node.B  # (D, r)
            grad_B = dW.T @ node.A  # (D, r)

            # Update Adam step counter
            node.adam_step += 1
            t = node.adam_step.item()

            # Update A with Adam
            node.m_A = self.b1 * node.m_A + (1 - self.b1) * grad_A
            node.v_A = self.b2 * node.v_A + (1 - self.b2) * (grad_A ** 2)
            m_hat_A = node.m_A / (1 - self.b1 ** t)
            v_hat_A = node.v_A / (1 - self.b2 ** t)
            node.A.data = node.A.data - self.eta_W * m_hat_A / (torch.sqrt(v_hat_A) + self.eps)

            # Update B with Adam
            node.m_B = self.b1 * node.m_B + (1 - self.b1) * grad_B
            node.v_B = self.b2 * node.v_B + (1 - self.b2) * (grad_B ** 2)
            m_hat_B = node.m_B / (1 - self.b1 ** t)
            v_hat_B = node.v_B / (1 - self.b2 ** t)
            node.B.data = node.B.data - self.eta_W * m_hat_B / (torch.sqrt(v_hat_B) + self.eps)

            # Clamp weights
            node.clamp_weights(self.W_max)
            
            # Store tension attribution
            if nid in node_tensions:
                node.last_attributed_tension = float(
                    node_tensions[nid].norm(dim=-1).mean().item()
                )
    
    def _update_health(
        self,
        T_norm: torch.Tensor,
        node_tensions: Dict[int, torch.Tensor],
        is_positive: bool
    ):
        """
        Update node health based on tension.
        
        Δρ = α(1 - |T|) - β(1 + |T|²)|T|·w_blame
        """
        alpha = 0.01
        beta = 0.05
        
        for record in self.state.active_path:
            nid = record.node_id
            node = self.state.get_node(nid)
            
            if node is None:
                continue
            
            # Compute blame weight
            w_blame = 1.0
            if nid in node_tensions:
                T_local_norm = node_tensions[nid].norm(dim=-1).mean()
                if T_norm > 0:
                    w_blame = min(1.0, (T_local_norm / (T_norm + 1e-9)).item())
            
            # Health delta
            reward = alpha * (1 - T_norm.item())
            penalty = beta * (1 + T_norm.item() ** 2) * T_norm.item() * w_blame
            
            if is_positive:
                delta = reward - penalty
            else:
                delta = -reward + penalty  # Invert for negative examples
            
            delta = np.clip(delta, -0.1, 0.1)
            node.rho = torch.clamp(
                node.rho + delta,
                -5.0, 10.0
            )
            
            # Update tension trace
            node.tension_trace = 0.9 * node.tension_trace + 0.1 * w_blame
    
    def _update_signatures(self, T_v: torch.Tensor):
        """
        Update routing signatures.
        
        S ← S + η_S · T_v · Re(⟨S, R⟩)
        """
        for record in self.state.active_path:
            node = self.state.get_node(record.node_id)

            if node is None or node.node_type == NodeType.INPUT:
                continue

            # Decode residue
            r_dot = self.residue.decode(node.id)

            # Update (average over batch)
            T_mean = T_v.mean(dim=0)
            node.S.data = node.S.data + self.eta_S * T_mean * r_dot

            # Re-normalize to unit sphere
            norm = node.S.norm()
            if norm > 1e-9:
                node.S.data = node.S.data / norm
    
    def get_max_tension_edge(self) -> Optional[Tuple[int, int]]:
        """Get edge with highest tension."""
        if not self.edge_tensions:
            return None
        return max(self.edge_tensions.items(), key=lambda x: x[1])[0]
    
    def get_stats(self) -> dict:
        return {
            "mode": "dual_signal",
            "lambda_contrastive": self.lambda_c,
            "lambda_retrograde": self.lambda_r,
            "edge_tensions": len(self.edge_tensions),
        }


# ═══════════════════════════════════════════════════════════════════════════
# HYBRID LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class HybridLearning(nn.Module):
    """
    Hybrid learning: Autograd for weights, dual-signal for structure.
    
    Uses PyTorch autograd for efficient weight updates while leveraging
    dual-signal tension tracking for intelligent structural evolution.
    """
    
    def __init__(
        self,
        state: GraphState,
        residue: HolographicResidue,
        grad_clip: float = 5.0,
        goodness_threshold: float = 0.5,
        lambda_contrastive: float = 0.65,
        lambda_retrograde: float = 0.35,
        **kwargs
    ):
        super().__init__()
        
        self.state = state
        self.residue = residue
        self.grad_clip = grad_clip
        self.theta = goodness_threshold
        self.lambda_c = lambda_contrastive
        self.lambda_r = lambda_retrograde
        
        # Edge tension tracking
        self.edge_tensions: Dict[Tuple[int, int], float] = {}
        
        # Goodness tracking for structural decisions
        self.node_goodness: Dict[int, float] = {}
    
    def learn(
        self,
        Y_hat: torch.Tensor,
        Y_star: torch.Tensor,
        output_node_id: int,
        is_positive: bool = True,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hybrid learning: autograd computes gradients, dual-signal tracks tensions.
        
        The actual weight updates happen via autograd in the training loop.
        This method computes structural metrics for evolution decisions.
        """
        # Ensure 2D
        if Y_hat.dim() == 1:
            Y_hat = Y_hat.unsqueeze(0)
            Y_star = Y_star.unsqueeze(0)
        
        B, D = Y_hat.shape
        
        # Compute loss (for reporting)
        if labels is not None:
            loss = F.cross_entropy(Y_hat, labels)
        else:
            loss = F.mse_loss(Y_hat, Y_star)
        
        # Compute output tension
        T_v = (Y_star - Y_hat) / np.sqrt(D)
        T_norm = torch.clamp(torch.sqrt(((Y_star - Y_hat) ** 2).mean()), 0.0, 1.0)
        
        # Track goodness for each active node
        self._track_goodness(is_positive)
        
        # Compute edge tensions using dual-signal logic
        self._compute_tensions(T_v, output_node_id)
        
        # Update health based on tension (structural signal)
        self._update_health(T_norm)
        
        return loss
    
    def _track_goodness(self, is_positive: bool):
        """Track goodness scores for structural decisions."""
        self.node_goodness.clear()
        
        for record in self.state.active_path:
            nid = record.node_id
            goodness = record.goodness
            
            # Compare to threshold
            if is_positive:
                # Should be above threshold
                self.node_goodness[nid] = goodness - self.theta
            else:
                # Should be below threshold
                self.node_goodness[nid] = self.theta - goodness
    
    def _compute_tensions(
        self,
        T_v: torch.Tensor,
        output_node_id: int
    ):
        """Compute edge tensions using retrograde-like flow."""
        self.edge_tensions.clear()
        
        active_path = self.state.active_path
        if not active_path:
            return
        
        # Reverse topo order
        sorted_path = sorted(active_path, key=lambda r: -r.hop)
        
        tensions: Dict[int, torch.Tensor] = {output_node_id: T_v}
        
        for record in sorted_path:
            nid = record.node_id
            if nid not in tensions:
                continue
            
            node = self.state.get_node(nid)
            if node is None:
                continue
            
            contribs = record.inbox_contributions
            if not contribs:
                continue
            
            T_curr = tensions[nid]
            
            # Blame weights
            norms = {sid: v.norm(dim=-1).mean() for sid, v in contribs.items()}
            total_norm = sum(norms.values()) + 1e-9
            
            spec_norm = node.spectral_norm()
            signal_scale = np.clip(spec_norm, 0.3, 4.0)
            
            for src_id, v_src in contribs.items():
                w_blame = norms[src_id] / total_norm
                T_upstream = T_curr * w_blame * signal_scale
                
                if src_id in tensions:
                    tensions[src_id] = tensions[src_id] + T_upstream
                else:
                    tensions[src_id] = T_upstream.clone()
                
                # Track edge tension
                edge = (src_id, nid)
                self.edge_tensions[edge] = max(
                    self.edge_tensions.get(edge, 0),
                    float(T_upstream.norm(dim=-1).mean().item())
                )
    
    def _update_health(self, T_norm: torch.Tensor):
        """Update health based on tension (for structural evolution)."""
        for record in self.state.active_path:
            node = self.state.get_node(record.node_id)
            if node is None:
                continue
            
            # Simple health update based on global tension
            delta = 0.01 * (1 - T_norm.item())
            node.rho = torch.clamp(node.rho + delta, -5.0, 10.0)
    
    def get_max_tension_edge(self) -> Optional[Tuple[int, int]]:
        """Get edge with highest tension."""
        if not self.edge_tensions:
            return None
        return max(self.edge_tensions.items(), key=lambda x: x[1])[0]
    
    def get_stats(self) -> dict:
        return {
            "mode": "hybrid",
            "edge_tensions": len(self.edge_tensions),
            "goodness_tracked": len(self.node_goodness),
        }
