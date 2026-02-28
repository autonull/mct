"""
MCT5 Unified Core Data Types

Node, GraphState, and execution records with full PyTorch integration.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

from .primitives import Primitive


class NodeType(Enum):
    """Node type in the compute graph."""
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


@dataclass
class NodeConfig:
    """Configuration for creating a new node."""
    node_type: NodeType = NodeType.HIDDEN
    primitive: Primitive = Primitive.GELU
    rho: float = 1.5


class Node(nn.Module):
    """
    A parametric node in the MCT5 compute graph.
    
    Uses low-rank weight factorization: W = A @ B.T
    where A, B ∈ ℝ^(D×r), giving 2Dr parameters instead of D².
    
    Features:
    - Learnable routing signature S for holographic addressing
    - Health scalar ρ for structural evolution
    - Adam optimizer state per node
    - Tension tracking for edge importance
    """
    
    def __init__(
        self,
        node_id: int,
        D: int,
        r: int,
        node_type: NodeType = NodeType.HIDDEN,
        primitive: Primitive = Primitive.GELU,
        rho: float = 1.5,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.id = node_id
        self.D = D
        self.r = r
        self.node_type = node_type
        self.primitive = primitive
        
        # Device
        self.device = device
        self._device = torch.device(device)
        
        # ═══════════════════════════════════════════════════════════════════
        # LEARNABLE PARAMETERS
        # ═══════════════════════════════════════════════════════════════════
        
        # Routing signature - unit sphere embedding for holographic addressing
        S_init = torch.randn(D, device=self._device)
        S_init = S_init / (S_init.norm() + 1e-9)
        self.S = nn.Parameter(S_init)
        
        # Low-rank weight factors: W = A @ B.T
        # He initialization scaled for low-rank
        scale = np.sqrt(2.0 / (D + r))
        self.A = nn.Parameter(torch.randn(D, r, device=self._device) * scale)
        self.B = nn.Parameter(torch.randn(D, r, device=self._device) * scale)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(D, device=self._device))
        
        # ═══════════════════════════════════════════════════════════════════
        # BUFFERS (state, not optimized)
        # ═══════════════════════════════════════════════════════════════════
        
        # Health scalar - determines firing and survival
        self.register_buffer("rho", torch.tensor(rho, device=self._device))
        
        # Adam optimizer state (per-node for online learning)
        self.register_buffer("m_A", torch.zeros(D, r, device=self._device))
        self.register_buffer("v_A", torch.zeros(D, r, device=self._device))
        self.register_buffer("m_B", torch.zeros(D, r, device=self._device))
        self.register_buffer("v_B", torch.zeros(D, r, device=self._device))
        self.register_buffer("adam_step", torch.tensor(0.0, device=self._device))
        
        # Structural health tracking
        self.register_buffer("tension_trace", torch.tensor(0.0, device=self._device))
        self.register_buffer("tension_ema", torch.tensor(0.0, device=self._device))
        
        # Non-buffer state (reset on structural changes)
        self.steps_idle: int = 0
        self.high_tension_count: int = 0
        self.last_attributed_tension: float = 0.0
        self.topo_depth: int = 0
        
        # Primitive-specific parameters (optional)
        self._init_primitive_params()
    
    def _init_primitive_params(self):
        """Initialize primitive-specific learnable parameters."""
        # Could add parameters for parametrized activations (e.g., PReLU, Swish beta)
        pass
    
    @property
    def W(self) -> torch.Tensor:
        """Reconstruct full D×D weight matrix."""
        return self.A @ self.B.T
    
    def apply_W(self, x: torch.Tensor) -> torch.Tensor:
        """Efficient low-rank matmul: W @ x = A @ (B.T @ x)."""
        return self.A @ (self.B.T @ x)
    
    def apply_W_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Batch apply: X ∈ ℝ^(B×D) → X @ W.T."""
        # X @ B @ A.T is more efficient than X @ (A @ B.T).T
        return (X @ self.B) @ self.A.T
    
    def spectral_norm(self) -> float:
        """
        Estimate spectral norm of W = A @ B.T.
        
        Uses the fact that σ_max(A @ B.T) ≤ σ_max(A) × σ_max(B).
        Fast approximation via power iteration would be more accurate but slower.
        """
        try:
            # SVD is expensive; use Frobenius as upper bound proxy
            # For better estimate, could do 1-2 power iterations
            norm_A = torch.linalg.norm(self.A, ord=2)
            norm_B = torch.linalg.norm(self.B, ord=2)
            return float((norm_A * norm_B / np.sqrt(self.r)))
        except:
            return 1.0
    
    def clamp_weights(self, max_norm: float):
        """Clamp weight factors to prevent explosion."""
        with torch.no_grad():
            for factor in [self.A, self.B]:
                fnorm = factor.norm(p='fro')
                if fnorm > max_norm:
                    factor.mul_(max_norm / fnorm)
    
    def reset_adam_state(self):
        """Reset Adam momentum - useful after structural changes."""
        with torch.no_grad():
            self.m_A.zero_()
            self.v_A.zero_()
            self.m_B.zero_()
            self.v_B.zero_()
            self.adam_step.zero_()
    
    def extra_repr(self) -> str:
        return f"id={self.id}, type={self.node_type.name}, prim={self.primitive.name}, ρ={self.rho:.2f}"


@dataclass
class ActiveRecord:
    """
    Records a node's activity during a forward pass.
    
    Used for:
    - Learning phase (computing gradients/updates)
    - Structural evolution (tension attribution)
    - Debugging/inspection
    """
    node_id: int
    hop: int  # Topological depth
    
    # Forward pass values
    V_in: torch.Tensor       # Aggregated input (D,) or (B, D) for batch
    V_weighted: torch.Tensor # After W transform, before primitive
    V_out: torch.Tensor      # After primitive (output)
    
    # For learning
    inbox_contributions: Dict[int, torch.Tensor] = field(default_factory=dict)
    goodness: float = 0.0    # ||V_out||²
    
    # Cached computations
    input_norm: float = 0.0
    output_norm: float = 0.0
    
    def __post_init__(self):
        """Compute cached norms."""
        if isinstance(self.V_in, torch.Tensor):
            if self.V_in.dim() == 1:
                self.input_norm = float(self.V_in.norm().item())
                self.output_norm = float(self.V_out.norm().item())
                self.goodness = float((self.V_out ** 2).sum().item())
            else:
                # Batch mode
                self.input_norm = float(self.V_in.norm(dim=-1).mean().item())
                self.output_norm = float(self.V_out.norm(dim=-1).mean().item())
                self.goodness = float((self.V_out ** 2).sum(dim=-1).mean().item())


@dataclass 
class BatchActiveRecord:
    """Batch version of ActiveRecord for efficient processing."""
    node_id: int
    hop: int
    
    V_in: torch.Tensor       # (B, D)
    V_weighted: torch.Tensor # (B, D)
    V_out: torch.Tensor      # (B, D)
    
    inbox_contributions: Dict[int, torch.Tensor] = field(default_factory=dict)
    goodness: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))  # (B,)
    
    def __post_init__(self):
        if self.goodness.numel() == 0:
            self.goodness = (self.V_out ** 2).sum(dim=-1)


class GraphState(nn.Module):
    """
    Global graph state management.
    
    Maintains:
    - Node registry (nn.ModuleDict for proper parameter registration)
    - Adjacency lists (DAG structure)
    - Topological ordering (cached)
    - Edge tensions (for structural evolution)
    """
    
    def __init__(
        self,
        D: int = 96,
        r: int = 24,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.D = D
        self.r = r
        self.device = device
        self._device = torch.device(device)
        
        # Node registry - ModuleDict for proper parameter tracking
        self.nodes = nn.ModuleDict()
        # Also keep a regular dict for fast lookup (ModuleDict doesn't have .get())
        self._nodes_dict: Dict[int, Node] = {}
        
        # Adjacency lists
        self.edges_out: Dict[int, List[int]] = {}
        self.edges_in: Dict[int, List[int]] = {}
        
        # Topological ordering cache
        self._topo_order: List[int] = []
        self._topo_valid: bool = False
        
        # Edge tensions (for structural evolution)
        self.edge_tensions: Dict[Tuple[int, int], float] = {}
        
        # Active path from last forward pass
        self.active_path: List[ActiveRecord] = []
        
        # Convergence tracking
        self.kappa: int = 0  # Passes since last structural change
        self.kappa_thresh: int = 150
        self.step_count: int = 0
        self.pruning_events_this_pass: int = 0
        
        # Node ID counter
        self.next_id: int = 0
    
    # ═══════════════════════════════════════════════════════════════════════
    # NODE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def create_node(
        self,
        node_type: NodeType = NodeType.HIDDEN,
        primitive: Primitive = Primitive.GELU,
        rho: float = 1.5
    ) -> Node:
        """Create and register a new node."""
        node = Node(
            node_id=self.next_id,
            D=self.D,
            r=self.r,
            node_type=node_type,
            primitive=primitive,
            rho=rho,
            device=self.device
        )
        
        self.nodes[str(self.next_id)] = node
        self._nodes_dict[self.next_id] = node
        self.edges_out[self.next_id] = []
        self.edges_in[self.next_id] = []
        
        self.next_id += 1
        self._topo_valid = False
        
        return node
    
    def remove_node(self, node_id: int):
        """Remove a node and all its edges."""
        if node_id not in self._nodes_dict:
            return
        
        # Remove all edges involving this node
        for dst in list(self.edges_out.get(node_id, [])):
            self.remove_edge(node_id, dst)
        for src in list(self.edges_in.get(node_id, [])):
            self.remove_edge(src, node_id)
        
        # Remove from registries
        del self._nodes_dict[node_id]
        del self.nodes[str(node_id)]
        self.edges_out.pop(node_id, None)
        self.edges_in.pop(node_id, None)
        
        self._topo_valid = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # EDGE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def add_edge(self, src: int, dst: int) -> bool:
        """
        Add edge src → dst if it doesn't create a cycle.
        
        Returns True if edge was added, False if rejected (cycle or exists).
        """
        if str(src) not in self.nodes or str(dst) not in self.nodes:
            return False
        
        if dst in self.edges_out.get(src, []):
            return True  # Already exists
        
        # Temporarily add and check acyclicity
        self.edges_out.setdefault(src, []).append(dst)
        self.edges_in.setdefault(dst, []).append(src)
        
        if not self.is_acyclic():
            # Revert
            self.edges_out[src].remove(dst)
            self.edges_in[dst].remove(src)
            return False
        
        self._topo_valid = False
        return True
    
    def remove_edge(self, src: int, dst: int):
        """Remove edge src → dst."""
        if dst in self.edges_out.get(src, []):
            self.edges_out[src].remove(dst)
        if src in self.edges_in.get(dst, []):
            self.edges_in[dst].remove(src)
        
        # Clear tension tracking for this edge
        self.edge_tensions.pop((src, dst), None)
        
        self._topo_valid = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # TOPOLOGICAL ORDERING
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_topo_order(self) -> List[int]:
        """
        Get cached topological order using Kahn's algorithm.
        
        Invalidated on structural changes.
        """
        if self._topo_valid:
            return self._topo_order
        
        # Compute in-degrees
        in_degree = {nid: len(self.edges_in.get(nid, [])) for nid in self.nodes.keys()}
        in_degree = {int(k): v for k, v in in_degree.items()}
        
        # Start with input nodes (in-degree 0)
        queue = sorted([nid for nid, deg in in_degree.items() if deg == 0])
        order: List[int] = []
        
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            
            for nxt in sorted(self.edges_out.get(curr, [])):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
        
        self._topo_order = order
        self._topo_valid = True
        
        # Update depth metadata
        self._update_depths(order)
        
        return order
    
    def _update_depths(self, topo_order: List[int]):
        """Update topological depth for each node."""
        depth: Dict[int, int] = {}

        for nid in topo_order:
            parents = self.edges_in.get(nid, [])
            if not parents:
                depth[nid] = 0
            else:
                depth[nid] = max(depth.get(p, 0) for p in parents) + 1

            if nid in self._nodes_dict:
                self._nodes_dict[nid].topo_depth = depth[nid]
    
    def is_acyclic(self) -> bool:
        """Check if current graph is a valid DAG using Kahn's algorithm."""
        in_degree = {nid: len(self.edges_in.get(nid, [])) for nid in self._nodes_dict.keys()}
        
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            n = queue.pop()
            visited += 1
            for child in self.edges_out.get(n, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return visited == len(self._nodes_dict)
    
    # ═══════════════════════════════════════════════════════════════════════
    # NODE QUERIES
    # ═══════════════════════════════════════════════════════════════════════

    def input_nodes(self) -> List[Node]:
        return [n for n in self._nodes_dict.values() if n.node_type == NodeType.INPUT]

    def output_nodes(self) -> List[Node]:
        return [n for n in self._nodes_dict.values() if n.node_type == NodeType.OUTPUT]

    def hidden_nodes(self) -> List[Node]:
        return [n for n in self._nodes_dict.values() if n.node_type == NodeType.HIDDEN]

    def get_node(self, node_id: int) -> Optional[Node]:
        return self._nodes_dict.get(node_id)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONVERGENCE & STATE
    # ═══════════════════════════════════════════════════════════════════════
    
    def tick(self):
        """Call at end of each training step."""
        if self.pruning_events_this_pass == 0:
            self.kappa += 1
        else:
            self.kappa = 0
        self.pruning_events_this_pass = 0
        self.step_count += 1
    
    def is_converged(self) -> bool:
        return self.kappa > self.kappa_thresh
    
    def reset_active_path(self):
        """Clear active path for new forward pass."""
        self.active_path.clear()
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        nodes = list(self._nodes_dict.values())
        prim_counts: Dict[str, int] = {}

        for n in nodes:
            name = n.primitive.name
            prim_counts[name] = prim_counts.get(name, 0) + 1

        return {
            "total_nodes": len(nodes),
            "input_nodes": sum(1 for n in nodes if n.node_type == NodeType.INPUT),
            "hidden_nodes": sum(1 for n in nodes if n.node_type == NodeType.HIDDEN),
            "output_nodes": sum(1 for n in nodes if n.node_type == NodeType.OUTPUT),
            "total_edges": sum(len(e) for e in self.edges_out.values()),
            "avg_rho": float(torch.tensor([n.rho.item() for n in nodes]).mean()) if nodes else 0.0,
            "avg_tension": float(torch.tensor([n.tension_trace.item() for n in nodes]).mean()) if nodes else 0.0,
            "primitives": prim_counts,
            "converged": self.is_converged(),
            "kappa": self.kappa,
        }
    
    def extra_repr(self) -> str:
        return f"nodes={len(self.nodes)}, edges={sum(len(e) for e in self.edges_out.values())}"
