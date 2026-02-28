"""
MCT5 Unified Structural Evolution

Phase 3: Intelligent graph structure adaptation.

Key innovations:
- Adaptive mutation rate based on loss landscape curvature
- Topology-aware spawning along high-tension edges
- Progressive complexity (starts simple, grows as needed)
- Pruning with memory (preserves useful substructures)
- Lateral wiring for shortcut connections
- Nonlinearity enforcement for expressive power
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import random

from .types import GraphState, Node, NodeType
from .primitives import Primitive, select_primitive_for_task
from .residue import HolographicResidue


# Mutable primitives (can change during evolution)
_MUTABLE_PRIMITIVES = [
    Primitive.RELU, Primitive.GELU, Primitive.TANH, Primitive.SWISH, Primitive.SILU,
    Primitive.LEAKY_RELU, Primitive.GATE, Primitive.BILINEAR, Primitive.QUADRATIC,
    Primitive.PRODUCT, Primitive.ADD, Primitive.MAX,
]

# Nonlinear primitives (important for expressivity)
_NONLINEAR_PRIMITIVES = [
    Primitive.QUADRATIC, Primitive.PRODUCT, Primitive.GATE, Primitive.BILINEAR,
]


class StructuralEvolution(nn.Module):
    """
    Intelligent structural evolution for MCT5.
    
    Features:
    - Adaptive mutation based on loss landscape
    - Topology-aware node insertion
    - Health-based pruning
    - Lateral shortcut wiring
    - Nonlinearity enforcement
    """
    
    def __init__(
        self,
        state: GraphState,
        residue: HolographicResidue,
        sigma_mut: float = 0.04,
        sigma_mut_min: float = 0.01,
        sigma_mut_max: float = 0.15,
        K: int = 2,
        tau_lateral: float = 0.25,
        quadratic_spawn_bias: float = 0.35,
        adaptive_mutation: bool = True,
        adaptation_sensitivity: float = 0.3,
        prune_threshold: float = 0.0,
        min_nodes: int = 4,
        lateral_wiring: bool = True,
        ensure_nonlinearity: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.state = state
        self.residue = residue
        
        # Mutation parameters
        self.sigma_mut = sigma_mut
        self.sigma_mut_min = sigma_mut_min
        self.sigma_mut_max = sigma_mut_max
        self.K = K
        
        # Adaptive mutation
        self.adaptive_mutation = adaptive_mutation
        self.adaptation_sensitivity = adaptation_sensitivity
        self._loss_history: List[float] = []
        self._curvature_estimate: float = 0.0
        
        # Lateral wiring
        self.tau_lateral = tau_lateral
        self.lateral_wiring = lateral_wiring
        
        # Spawn bias
        self.quad_bias = quadratic_spawn_bias
        self.ensure_nonlinearity = ensure_nonlinearity
        
        # Pruning
        self.prune_threshold = prune_threshold
        self.min_nodes = min_nodes
        
        # Device
        self.device = device
        
        # Track if graph has nonlinear capacity
        self._has_quadratic = False
        self._has_product = False
        
        # Statistics
        self.total_pruned = 0
        self.total_spawned = 0
        self.total_lateral_edges = 0
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN EVOLUTION STEP
    # ═══════════════════════════════════════════════════════════════════════
    
    def evolve(
        self,
        pruned: List[int],
        max_tension_edge: Optional[Tuple[int, int]],
        ema_loss: float,
        ema_loss_prev: float,
        force: bool = False
    ) -> dict:
        """
        Execute structural evolution step.
        
        Args:
            pruned: List of node IDs that were pruned
            max_tension_edge: Edge with highest tension (for spawning)
            ema_loss: Current EMA loss
            ema_loss_prev: Previous EMA loss
            force: Force evolution regardless of interval
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "pruned": len(pruned),
            "spawned": 0,
            "lateral_edges": 0,
            "sigma_mut": self.sigma_mut,
        }
        
        # Update adaptive mutation rate
        if self.adaptive_mutation:
            self._update_adaptive_mutation(ema_loss, ema_loss_prev)
        
        # Always ensure nonlinearity early
        if self.ensure_nonlinearity and not self._has_quadratic:
            self._ensure_nonlinearity()
            stats["spawned"] += 1
        
        # Spawn on pruned locations
        if pruned:
            spawned = self._spawn_on_pruned(pruned)
            stats["spawned"] += len(spawned)
        
        # Spawn on high-tension edge
        if max_tension_edge is not None:
            spawned = self._insert_capacity(max_tension_edge)
            stats["spawned"] += len(spawned)
        
        # Stagnation-based spawning
        loss_delta = abs(ema_loss - ema_loss_prev)
        if loss_delta < 0.005 and ema_loss > 0.1 and len(self.state.nodes) < 50:
            # Loss is stagnant, add capacity
            if max_tension_edge:
                spawned = self._insert_capacity(max_tension_edge)
                stats["spawned"] += len(spawned)
        
        # Lateral wiring
        if self.lateral_wiring:
            lateral_count = self._lateral_wiring()
            stats["lateral_edges"] = lateral_count
        
        # Update nonlinearity tracking
        self._update_nonlinearity_tracking()
        
        return stats
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRUNING
    # ═══════════════════════════════════════════════════════════════════════
    
    def prune(self) -> List[int]:
        """
        Remove nodes with health below threshold.
        
        Returns:
            List of pruned node IDs
        """
        pruned = []
        
        # Sort by health (lowest first)
        nodes_by_health = sorted(
            self.state.hidden_nodes(),
            key=lambda n: n.rho.item()
        )
        
        for node in nodes_by_health:
            # Don't prune if we're at minimum size
            if len(self.state.nodes) <= self.min_nodes:
                break
            
            if node.rho.item() < self.prune_threshold:
                nid = node.id
                
                # Release residue basis
                self.residue.release_node(nid)
                
                # Remove from graph
                self.state.remove_node(nid)
                
                pruned.append(nid)
                self.state.pruning_events_this_pass += 1
                self.total_pruned += 1
        
        return pruned
    
    # ═══════════════════════════════════════════════════════════════════════
    # SPAWNING
    # ═══════════════════════════════════════════════════════════════════════
    
    def _spawn_on_pruned(self, pruned_ids: List[int]) -> List[int]:
        """Spawn new nodes at pruned locations."""
        spawned = []
        
        for nid in pruned_ids:
            # Find neighbors of pruned node
            # (This info is lost after removal, so we spawn along existing edges)
            pass
        
        # Instead, spawn along random high-activity edges
        if self.state.active_path:
            # Pick edges from active path
            active_edges = []
            for record in self.state.active_path:
                for src_id in record.inbox_contributions.keys():
                    active_edges.append((src_id, record.node_id))
            
            if active_edges:
                # Pick top tension edge
                edge = random.choice(active_edges)
                new_nodes = self._insert_capacity(edge)
                spawned.extend(new_nodes)
        
        return spawned
    
    def _insert_capacity(
        self,
        edge: Tuple[int, int],
        K: Optional[int] = None
    ) -> List[int]:
        """
        Insert K new nodes along an edge.
        
        Args:
            edge: (src, dst) tuple
            K: Number of nodes to insert (default: self.K)
        
        Returns:
            List of new node IDs
        """
        if K is None:
            K = self.K
        
        src_id, dst_id = edge
        
        # Validate edge
        src_node = self.state.get_node(src_id)
        dst_node = self.state.get_node(dst_id)
        
        if src_node is None or dst_node is None:
            return []
        
        # Remove existing edge
        self.state.remove_edge(src_id, dst_id)
        
        new_ids = []
        
        for k in range(K):
            # Choose primitive
            primitive = self._choose_primitive(src_node.primitive)
            
            # Create new node
            new_node = self.state.create_node(
                node_type=NodeType.HIDDEN,
                primitive=primitive,
                rho=1.5  # Start with healthy rho
            )
            
            # Register residue basis
            self.residue.register_node(new_node.id)
            
            # Inherit and perturb weights
            self._inherit_weights(new_node, src_node)
            
            # Wire: src -> new -> dst
            self.state.add_edge(src_id, new_node.id)
            
            if not self.state.add_edge(new_node.id, dst_id):
                # If can't connect to dst, connect to output
                for out_node in self.state.output_nodes():
                    if self.state.add_edge(new_node.id, out_node.id):
                        break
            
            new_ids.append(new_node.id)
            self.total_spawned += 1
        
        return new_ids
    
    def _inherit_weights(self, new_node: Node, parent: Node):
        """Inherit weights from parent with perturbation."""
        with torch.no_grad():
            # Perturbation scaled by adaptive sigma_mut
            noise_A = torch.randn_like(parent.A) * self.sigma_mut
            noise_B = torch.randn_like(parent.B) * self.sigma_mut
            noise_S = torch.randn_like(parent.S) * self.sigma_mut
            
            new_node.A.copy_(parent.A + noise_A)
            new_node.B.copy_(parent.B + noise_B)
            new_node.S.copy_(parent.S + noise_S)
            
            # Re-normalize signature
            new_node.S.div_(new_node.S.norm() + 1e-9)
            
            # Copy bias with small perturbation
            new_node.bias.copy_(parent.bias + torch.randn_like(parent.bias) * 0.01)
    
    # ═══════════════════════════════════════════════════════════════════════
    # LATERAL WIRING
    # ═══════════════════════════════════════════════════════════════════════
    
    def _lateral_wiring(self) -> int:
        """
        Add shortcut edges from high-tension nodes to downstream.
        
        Nodes with persistent high tension grow lateral connections
        to skip layers and improve gradient flow.
        
        Returns:
            Number of new edges added
        """
        new_edges = 0
        
        for node in self.state.hidden_nodes():
            # Check for high tension persistence
            if node.tension_trace < self.tau_lateral:
                continue
            
            nid = node.id
            
            # Find downstream nodes
            downstream = self._get_downstream(nid)
            existing = set(self.state.edges_out.get(nid, []))
            
            # Candidates: downstream nodes not already connected
            candidates = [
                d for d in downstream
                if d not in existing
                and self.state.get_node(d) is not None
                and self.state.get_node(d).node_type != NodeType.INPUT
            ]
            
            if candidates:
                # Add shortcut to random downstream node
                target = random.choice(candidates)
                if self.state.add_edge(nid, target):
                    new_edges += 1
            
            # Reset tension count after wiring attempt
            node.high_tension_count = 0
        
        self.total_lateral_edges += new_edges
        return new_edges
    
    def _get_downstream(self, node_id: int) -> List[int]:
        """Get all downstream node IDs via BFS."""
        visited = set()
        queue = list(self.state.edges_out.get(node_id, []))
        
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(self.state.edges_out.get(nid, []))
        
        return list(visited)
    
    # ═══════════════════════════════════════════════════════════════════════
    # NONLINEARITY ENFORCEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def _ensure_nonlinearity(self):
        """Ensure graph has nonlinear primitives for XOR-type problems."""
        if self._has_quadratic or self._has_product:
            return
        
        # Find input edges
        input_ids = {n.id for n in self.state.input_nodes()}
        
        best_edge = None
        best_tension = 0
        
        for (u, v), tension in self.state.edge_tensions.items():
            if u in input_ids and tension > best_tension:
                best_edge = (u, v)
                best_tension = tension
        
        if best_edge is None and self.state.edge_tensions:
            best_edge = max(self.state.edge_tensions.items(), key=lambda x: x[1])[0]
        
        if best_edge is None:
            # No tension info, pick random input edge
            for inp in self.state.input_nodes():
                for out_id in self.state.edges_out.get(inp.id, []):
                    best_edge = (inp.id, out_id)
                    break
                if best_edge:
                    break
        
        if best_edge:
            src_id, dst_id = best_edge
            
            # Create nonlinear node
            primitive = random.choice([Primitive.QUADRATIC, Primitive.PRODUCT])
            new_node = self.state.create_node(
                node_type=NodeType.HIDDEN,
                primitive=primitive,
                rho=1.5
            )
            
            self.residue.register_node(new_node.id)
            
            # Wire
            self.state.remove_edge(src_id, dst_id)
            self.state.add_edge(src_id, new_node.id)
            self.state.add_edge(new_node.id, dst_id)
            
            # Inherit weights
            src_node = self.state.get_node(src_id)
            if src_node:
                self._inherit_weights(new_node, src_node)
            
            self._update_nonlinearity_tracking()
    
    def _update_nonlinearity_tracking(self):
        """Update tracking of nonlinear primitives in graph."""
        all_nodes = self.state.hidden_nodes() + self.state.input_nodes() + self.state.output_nodes()
        self._has_quadratic = any(
            n.primitive == Primitive.QUADRATIC
            for n in all_nodes
        )
        self._has_product = any(
            n.primitive == Primitive.PRODUCT
            for n in all_nodes
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # ADAPTIVE MUTATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def _update_adaptive_mutation(self, ema_loss: float, ema_loss_prev: float):
        """
        Adapt mutation rate based on loss landscape curvature.
        
        High curvature (oscillating loss) → reduce mutation
        Low curvature (steady loss) → increase mutation for exploration
        """
        self._loss_history.append(ema_loss)
        
        # Keep last 20 losses
        if len(self._loss_history) > 20:
            self._loss_history.pop(0)
        
        if len(self._loss_history) < 5:
            return
        
        # Estimate curvature from loss variance
        recent = self._loss_history[-5:]
        loss_std = np.std(recent)
        loss_mean = np.mean(recent)
        
        # Coefficient of variation as curvature proxy
        if loss_mean > 0:
            cv = loss_std / loss_mean
        else:
            cv = 0
        
        # Update curvature estimate with EMA
        self._curvature_estimate = (
            self.adaptation_sensitivity * cv +
            (1 - self.adaptation_sensitivity) * self._curvature_estimate
        )
        
        # Adjust sigma_mut inversely to curvature
        # High curvature → low mutation (stability)
        # Low curvature → high mutation (exploration)
        target_sigma = self.sigma_mut_max - (
            self._curvature_estimate * (self.sigma_mut_max - self.sigma_mut_min)
        )
        
        # Smooth transition
        self.sigma_mut = (
            self.adaptation_sensitivity * target_sigma +
            (1 - self.adaptation_sensitivity) * self.sigma_mut
        )
        
        # Clamp
        self.sigma_mut = np.clip(
            self.sigma_mut,
            self.sigma_mut_min,
            self.sigma_mut_max
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIMITIVE SELECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def _choose_primitive(self, parent_primitive: Primitive) -> Primitive:
        """
        Choose primitive for new node.
        
        Biases:
        - 35% chance of nonlinear if not present
        - 30% chance of random mutable primitive
        - Otherwise inherit from parent
        """
        # Bias toward nonlinearity if missing
        if not self._has_quadratic and random.random() < self.quad_bias:
            return random.choice([Primitive.QUADRATIC, Primitive.PRODUCT])
        
        # Random mutation
        if random.random() < 0.3:
            return random.choice(_MUTABLE_PRIMITIVES)
        
        # Inherit
        return parent_primitive
    
    # ═══════════════════════════════════════════════════════════════════════
    # GRAPH INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def initialize_graph(self, primitive_hidden: str = "GELU"):
        """
        Build minimal starter graph.

        Structure:
        ```
        Input ──→ [GELU, QUADRATIC, PRODUCT, GATE] ──→ ADD ──→ Output
           └─────────────────────────────────────────────↑
        ```
        """
        # Clear existing
        self.state.nodes.clear()
        self.state._nodes_dict.clear()
        self.state.edges_out.clear()
        self.state.edges_in.clear()
        self.state.next_id = 0
        self.state.edge_tensions.clear()
        self.state._topo_valid = False
        
        # Parse primitive
        try:
            hidden_prim = getattr(Primitive, primitive_hidden)
        except AttributeError:
            hidden_prim = Primitive.GELU
        
        # Create input node
        inp = self.state.create_node(
            node_type=NodeType.INPUT,
            primitive=Primitive.FORK,
            rho=3.0
        )
        self.residue.register_node(inp.id)
        
        # Create diverse hidden nodes
        hidden_primitives = [
            Primitive.GELU,
            Primitive.QUADRATIC,
            Primitive.PRODUCT,
            Primitive.GATE,
        ]
        
        hidden_nodes = []
        for prim in hidden_primitives:
            h = self.state.create_node(
                node_type=NodeType.HIDDEN,
                primitive=prim,
                rho=1.5
            )
            self.residue.register_node(h.id)
            hidden_nodes.append(h)
        
        # Create aggregator
        agg = self.state.create_node(
            node_type=NodeType.HIDDEN,
            primitive=Primitive.ADD,
            rho=1.5
        )
        self.residue.register_node(agg.id)
        
        # Create output node
        out = self.state.create_node(
            node_type=NodeType.OUTPUT,
            primitive=Primitive.FORK,
            rho=2.0
        )
        self.residue.register_node(out.id)
        
        # Wire: Input → Hidden
        for h in hidden_nodes:
            self.state.add_edge(inp.id, h.id)
        
        # Wire: Hidden → ADD
        for h in hidden_nodes:
            self.state.add_edge(h.id, agg.id)
        
        # Wire: Input → ADD (skip connection)
        self.state.add_edge(inp.id, agg.id)
        
        # Wire: ADD → Output
        self.state.add_edge(agg.id, out.id)
        
        # Wire: Input → Output (direct skip)
        self.state.add_edge(inp.id, out.id)
        
        # Update tracking
        self._update_nonlinearity_tracking()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> dict:
        """Get evolution statistics."""
        return {
            "total_pruned": self.total_pruned,
            "total_spawned": self.total_spawned,
            "total_lateral_edges": self.total_lateral_edges,
            "current_sigma_mut": self.sigma_mut,
            "curvature_estimate": self._curvature_estimate,
            "has_quadratic": self._has_quadratic,
            "has_product": self._has_product,
        }
