"""
MCT5 Unified Forward Executor

Phase 1: Depth-layer batched forward execution with anytime property.

Key features:
- Topological depth-layer processing for parallelism
- Anytime threshold τ(t) = 0.5 · exp(λ_τ · (t - t_budget))
- Holographic residue boost: ρ_active = ρ + ⟨S, V_in⟩ + Re(⟨S, R⟩)
- Ghost injection for near-miss activations
- Inbox decay for asynchronous signal propagation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

from .types import GraphState, Node, NodeType, ActiveRecord, BatchActiveRecord
from .primitives import Primitive, apply_primitive
from .residue import HolographicResidue


class ForwardExecutor(nn.Module):
    """
    Forward execution engine for MCT5.
    
    Executes the compute graph in topological order with:
    - Depth-layer batching for efficiency
    - Anytime property via exponential threshold
    - Holographic context integration
    - Comprehensive activity recording for learning
    """
    
    def __init__(
        self,
        state: GraphState,
        residue: HolographicResidue,
        lambda_async: float = 0.08,
        residue_boost_scale: float = 1.0
    ):
        super().__init__()
        
        self.state = state
        self.residue = residue
        self.lambda_async = lambda_async
        self.residue_boost_scale = residue_boost_scale
        
        # Cached depth layers
        self._depth_layers: Dict[int, List[int]] = {}
        self._layers_valid: bool = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN FORWARD PASS
    # ═══════════════════════════════════════════════════════════════════════
    
    def forward(
        self,
        X: torch.Tensor,
        t_budget: int = 15,
        lambda_tau: float = 0.12
    ) -> torch.Tensor:
        """
        Execute forward pass with input X.
        
        Args:
            X: Input tensor (D,) for single sample or (B, D) for batch
            t_budget: Maximum hop count (anytime deadline)
            lambda_tau: Threshold steepness
        
        Returns:
            Output tensor from output nodes (D,) or (B, D)
        """
        # Ensure 2D
        if X.dim() == 1:
            X = X.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        B = X.size(0)
        D = self.state.D
        
        # Reset for new pass
        self.state.reset_active_path()
        self._invalidate_layers()
        
        # Get topological order and build depth layers
        topo_order = self.state.get_topo_order()
        layers = self._build_depth_layers(topo_order)
        
        if not layers:
            # Empty graph - return zeros
            if single_sample:
                return torch.zeros(D, device=self.state._device)
            return torch.zeros(B, D, device=self.state._device)
        
        # Node outputs buffer: node_id -> (B, D) tensor
        node_outputs: Dict[int, torch.Tensor] = {}
        
        # Inbox for message passing: dst_id -> {src_id: tensor}
        inbox: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        inbox_time: Dict[int, Dict[int, int]] = defaultdict(dict)
        
        max_depth = max(layers.keys())
        
        # ═══════════════════════════════════════════════════════════════════
        # PROCESS LAYERS
        # ═══════════════════════════════════════════════════════════════════
        
        for depth in range(min(max_depth + 1, t_budget)):
            layer_nodes = layers.get(depth, [])
            if not layer_nodes:
                continue
            
            # Compute threshold for this depth
            tau = self._compute_threshold(depth, t_budget, lambda_tau)
            
            for node_id in layer_nodes:
                node = self.state.get_node(node_id)
                if node is None:
                    continue
                
                # ────────────────────────────────────────────────────────────
                # INPUT NODES - Always fire with external input
                # ────────────────────────────────────────────────────────────
                
                if node.node_type == NodeType.INPUT:
                    # Pad/truncate input to D dimensions
                    V_out = self._process_input(X, node_id)
                    node_outputs[node_id] = V_out
                    node.steps_idle = 0
                    
                    # Route to downstream
                    self._route(node_id, V_out, depth, inbox, inbox_time)
                    
                    # Record
                    record = ActiveRecord(
                        node_id=node_id,
                        hop=depth,
                        V_in=X if X.size(1) == D else self._pad_input(X),
                        V_weighted=V_out,
                        V_out=V_out,
                        inbox_contributions={},
                        goodness=float((V_out ** 2).sum(dim=-1).mean().item())
                    )
                    self.state.active_path.append(record)
                    continue
                
                # ────────────────────────────────────────────────────────────
                # AGGREGATE INPUT FROM INBOX
                # ────────────────────────────────────────────────────────────
                
                contribs = inbox.get(node_id, {})
                
                if not contribs:
                    # No input - node is idle
                    node.steps_idle += 1
                    continue
                
                # Apply decay and aggregate
                V_in = self._aggregate_inbox(contribs, inbox_time[node_id], depth, node)
                
                if V_in is None:
                    node.steps_idle += 1
                    continue
                
                # ────────────────────────────────────────────────────────────
                # ACTIVATION POTENTIAL & FIRING DECISION
                # ────────────────────────────────────────────────────────────
                
                # Compute activation potential
                s_dot = (V_in * node.S.unsqueeze(0)).sum(dim=-1)  # (B,)
                residue_boost = self.residue.decode(node_id) * self.residue_boost_scale
                
                # For batch: average potential across batch
                rho_active = node.rho + s_dot.mean() + residue_boost
                
                # Firing decision
                if rho_active < tau:
                    # Ghost injection - near miss
                    self.residue.inject_ghost(node_id, float(rho_active.item()), float(depth))
                    node.steps_idle += 1
                    continue
                
                # ────────────────────────────────────────────────────────────
                # NODE FIRES
                # ────────────────────────────────────────────────────────────
                
                node.steps_idle = 0
                
                # Linear transform: V_weighted = W @ V_in + bias
                V_weighted = node.apply_W_batch(V_in) + node.bias  # (B, D)
                
                # Primitive nonlinearity
                V_out = self._apply_primitive(node.primitive, V_weighted, contribs, node_outputs)
                
                # Guard against NaN/Inf
                if not torch.isfinite(V_out).all():
                    V_out = torch.zeros_like(V_out)
                
                # Store output
                node_outputs[node_id] = V_out
                
                # Route to downstream
                self._route(node_id, V_out, depth, inbox, inbox_time)
                
                # Record activity
                goodness = (V_out ** 2).sum(dim=-1).mean()
                record = ActiveRecord(
                    node_id=node_id,
                    hop=depth,
                    V_in=V_in,
                    V_weighted=V_weighted,
                    V_out=V_out,
                    inbox_contributions=contribs.copy(),
                    goodness=float(goodness.item())
                )
                self.state.active_path.append(record)
        
        # ═══════════════════════════════════════════════════════════════════
        # COLLECT OUTPUT
        # ═══════════════════════════════════════════════════════════════════
        
        output_nodes = self.state.output_nodes()
        fired_outputs = [
            node_outputs[n.id] for n in output_nodes 
            if n.id in node_outputs
        ]
        
        if not fired_outputs:
            # No output fired - return mean of last layer
            if self.state.active_path:
                last_record = self.state.active_path[-1]
                result = last_record.V_out
            else:
                result = torch.zeros(B, D, device=self.state._device)
        else:
            # Average output node activations
            result = torch.stack(fired_outputs, dim=0).mean(dim=0)
        
        # End of pass - update residue
        self.residue.end_of_pass()
        
        # Update idle counters for non-firing hidden nodes
        fired_ids = {r.node_id for r in self.state.active_path}
        for node in self.state.hidden_nodes():
            if node.id not in fired_ids:
                node.steps_idle += 1
        
        if single_sample:
            return result.squeeze(0)
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # BATCH FORWARD
    # ═══════════════════════════════════════════════════════════════════════
    
    def forward_batch(
        self,
        X_batch: torch.Tensor,
        t_budget: int = 15,
        lambda_tau: float = 0.12
    ) -> Tuple[torch.Tensor, List[ActiveRecord]]:
        """
        Forward pass for batch with activity recording.
        
        Returns:
            output: (B, D) tensor
            records: List of activity records for learning
        """
        output = self.forward(X_batch, t_budget, lambda_tau)
        return output, self.state.active_path.copy()
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def reset_context(self):
        """Reset holographic residue at sequence boundary."""
        self.residue.reset()
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════════════════════
    
    def _build_depth_layers(
        self,
        topo_order: List[int]
    ) -> Dict[int, List[int]]:
        """Build depth -> [node_ids] mapping."""
        if self._layers_valid:
            return self._depth_layers
        
        layers: Dict[int, List[int]] = defaultdict(list)
        
        for nid in topo_order:
            node = self.state.get_node(nid)
            if node:
                layers[node.topo_depth].append(nid)
        
        self._depth_layers = dict(layers)
        self._layers_valid = True
        return self._depth_layers
    
    def _invalidate_layers(self):
        """Invalidate cached layers."""
        self._layers_valid = False
    
    def _compute_threshold(
        self,
        t: int,
        t_budget: int,
        lambda_tau: float
    ) -> float:
        """
        Compute anytime threshold τ(t).
        
        τ(t) = 0.5 · exp(λ_τ · (t - t_budget))
        
        Early in pass (t << t_budget): τ ≈ 0 (permissive)
        At deadline (t = t_budget): τ = 0.5
        After deadline: τ → ∞ (restrictive)
        """
        return 0.5 * np.exp(lambda_tau * (t - t_budget))
    
    def _pad_input(self, X: torch.Tensor) -> torch.Tensor:
        """Pad input to D dimensions."""
        B, current_dim = X.shape
        D = self.state.D
        
        if current_dim >= D:
            return X[:, :D]
        
        padding = torch.zeros(B, D - current_dim, device=X.device)
        return torch.cat([X, padding], dim=1)
    
    def _process_input(
        self,
        X: torch.Tensor,
        node_id: int
    ) -> torch.Tensor:
        """Process input node - pad to D dimensions."""
        return self._pad_input(X)
    
    def _route(
        self,
        src_id: int,
        V_out: torch.Tensor,
        t: int,
        inbox: Dict[int, Dict[int, torch.Tensor]],
        inbox_time: Dict[int, Dict[int, int]]
    ):
        """Route output to downstream nodes."""
        for dst_id in self.state.edges_out.get(src_id, []):
            if str(dst_id) in self.state.nodes:
                inbox[dst_id][src_id] = V_out.clone()
                inbox_time[dst_id][src_id] = t
    
    def _aggregate_inbox(
        self,
        contribs: Dict[int, torch.Tensor],
        times: Dict[int, int],
        t_now: int,
        node: Node
    ) -> Optional[torch.Tensor]:
        """
        Aggregate inbox messages with decay.
        
        For multi-input primitives, returns list of tensors.
        For others, returns mean.
        """
        if not contribs:
            return None
        
        vectors: List[torch.Tensor] = []
        
        for src_id, vec in contribs.items():
            t_arrived = times.get(src_id, t_now)
            decay = np.exp(-self.lambda_async * (t_now - t_arrived))
            vectors.append(vec * decay)
        
        if not vectors:
            return None
        
        # Multi-input primitives need the list
        if node.primitive in [Primitive.GATE, Primitive.BILINEAR, 
                               Primitive.PRODUCT, Primitive.MAX,
                               Primitive.CONCAT_PROJECT, Primitive.ATTENTION_LITE]:
            # Return first vector for binary primitives with single input
            if len(vectors) == 1 and node.primitive != Primitive.ADD:
                return vectors[0]
            # For multi-input, we'll handle in apply_primitive
            return torch.stack(vectors, dim=0).mean(dim=0)
        
        # Default: mean aggregation
        return torch.stack(vectors, dim=0).mean(dim=0)
    
    def _apply_primitive(
        self,
        primitive: Primitive,
        V_weighted: torch.Tensor,
        contribs: Dict[int, torch.Tensor],
        node_outputs: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply primitive with proper handling for multi-input cases.
        
        For binary primitives, reconstruct original inputs.
        """
        # Check if this is a multi-input primitive with multiple contributions
        if primitive in [Primitive.GATE, Primitive.BILINEAR, Primitive.PRODUCT, 
                         Primitive.MAX] and len(contribs) >= 2:
            # Reconstruct weighted inputs
            weighted_inputs = []
            for src_id, V_in_src in contribs.items():
                src_node = self.state.get_node(src_id)
                if src_node:
                    # Apply source node's weight transform
                    V_w = src_node.apply_W_batch(V_in_src) + src_node.bias
                    weighted_inputs.append(V_w)
            
            if len(weighted_inputs) >= 2:
                return apply_primitive(primitive, weighted_inputs)
        
        # Standard unary application
        return apply_primitive(primitive, V_weighted)
    
    def extra_repr(self) -> str:
        return f"lambda_async={self.lambda_async}, boost_scale={self.residue_boost_scale}"
