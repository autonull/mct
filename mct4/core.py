"""
MCT4 Core Data Types

Node, Context, and Graph data structures as per the MCT4 specification.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import heapq

from .primitives import Primitive, apply_primitive


class NodeType(Enum):
    """Node role classification."""
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


@dataclass
class Node:
    """
    A node in the MCT4 graph.
    
    Each node is an asynchronous, self-contained processing unit with full parametric capacity.
    """
    id: int
    D: int  # Vector dimensionality
    
    # Routing signature: geometric embedding used for activation potential
    S: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    # Health scalar: survival fitness and routing priority
    rho_base: float = 0.0
    
    # Learnable weight matrix W ∈ ℝᴰˣᴰ, initialized to identity × ε
    W: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    W_factored: bool = False  # Whether using low-rank factorization
    A: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # For low-rank: W = A @ B.T
    B: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    rank: int = 64  # Low-rank dimension
    
    # Primitive nonlinearity
    primitive: Primitive = Primitive.GELU
    
    # Inbox: sender_id -> [(vector, time_arrived)]
    inbox: Dict[int, List[Tuple[np.ndarray, int]]] = field(default_factory=dict)
    
    # Outgoing edges: list of target node IDs
    edges_out: List[int] = field(default_factory=list)
    
    # Incoming edges: list of source node IDs (for retrograde flow)
    edges_in: List[int] = field(default_factory=list)
    
    # Steps since last firing
    steps_idle: int = 0
    
    # Exponential moving average of ||T|| observed at this node
    tension_trace: float = 0.0
    
    # Node type
    node_type: NodeType = NodeType.HIDDEN
    
    # Consecutive high-tension counter (for lateral wiring)
    high_tension_count: int = 0
    
    # Hop depth at which this node last fired (for execution ordering)
    last_hop: int = -1
    
    # Batch output storage
    batch_outputs: List[np.ndarray] = field(default_factory=list)
    
    # Track attributed tension from retrograde flow (for capacity insertion)
    last_attributed_tension: float = 0.0
    
    def __post_init__(self):
        """Initialize node arrays after creation."""
        if self.D > 0:
            if len(self.S) == 0:
                # Random signature on unit sphere for geometric routing
                self.S = np.random.randn(self.D)
                self.S /= np.linalg.norm(self.S) + 1e-9
            
            if self.W.shape == (0, 0):
                # Initialize W to identity × ε
                epsilon = 0.01
                if self.W_factored and self.rank < self.D:
                    # Low-rank initialization
                    self.A = np.random.randn(self.D, self.rank) * 0.01
                    self.B = np.random.randn(self.D, self.rank) * 0.01
                else:
                    self.W = np.eye(self.D) * epsilon
    
    def get_weight_matrix(self) -> np.ndarray:
        """Get the effective weight matrix (full or reconstructed from factors)."""
        if self.W_factored:
            return self.A @ self.B.T
        return self.W
    
    def apply_weight(self, x: np.ndarray) -> np.ndarray:
        """Apply weight matrix to input vector."""
        if self.W_factored:
            return self.A @ (self.B.T @ x)
        return self.W @ x
    
    def update_weight_full(self, delta: np.ndarray):
        """Update full-rank weight matrix."""
        self.W = self.W + delta
    
    def update_weight_factored(self, T_local: np.ndarray, V_in: np.ndarray, eta: float):
        """Update low-rank factors via rank-1 outer products."""
        # A ← A + η · T_local ⊗ (Bᵀ V_in)
        # B ← B + η · V_in ⊗ (Aᵀ T_local)
        BtV = self.B.T @ V_in
        AtT = self.A.T @ T_local
        self.A = self.A + eta * np.outer(T_local, BtV)
        self.B = self.B + eta * np.outer(V_in, AtT)
    
    def clamp_weight_norm(self, W_max: float):
        """Clamp weight matrix Frobenius norm via spectral rescaling."""
        if self.W_factored:
            # For factored mode, clamp individual factors
            for factor in [self.A, self.B]:
                norm = np.linalg.norm(factor, 'fro')
                if norm > W_max:
                    factor *= W_max / norm
        else:
            norm = np.linalg.norm(self.W, 'fro')
            if norm > W_max:
                self.W *= W_max / norm
    
    def fire(self, t: int, V_in: np.ndarray) -> np.ndarray:
        """
        Execute node: apply weight matrix, then primitive, return output.
        
        Args:
            t: Current hop count
            V_in: Aggregated input vector
            
        Returns:
            Output vector V_out
        """
        # Apply weight matrix then primitive
        V_weighted = self.apply_weight(V_in)
        V_out = apply_primitive(self.primitive, V_weighted)
        
        self.steps_idle = 0
        self.last_hop = t
        
        return V_out


@dataclass
class Context:
    """
    Global context vector C ∈ ℝᴰ.
    
    Persists across forward passes within a sequence, functioning as learned routing memory.
    Carries forward temporal context; resets only at sequence boundaries.
    """
    D: int
    C: np.ndarray = field(default_factory=lambda: np.zeros(0))
    decay_c: float = 0.95
    
    def __post_init__(self):
        if len(self.C) == 0:
            self.C = np.zeros(self.D)
    
    def reset(self):
        """Reset context to zero (at sequence boundaries)."""
        self.C = np.zeros(self.D)
    
    def add_ghost(self, rho: float, S: np.ndarray):
        """
        Add ghost signal from a node that failed to fire.
        
        C ← C · decay_c + (ρᵢ / D) · Sᵢ
        """
        self.C = self.C * self.decay_c + (rho / self.D) * S
    
    def decay(self):
        """Apply decay without adding signal."""
        self.C = self.C * self.decay_c


@dataclass
class InboxMessage:
    """A message in a node's inbox."""
    sender_id: int
    vector: np.ndarray
    time_arrived: int


@dataclass 
class ActivePathRecord:
    """Record of nodes that fired during a forward pass (for learning)."""
    node_id: int
    V_in: np.ndarray  # Pre-activation input
    V_out: np.ndarray  # Post-activation output
    V_weighted: np.ndarray  # Post-weight, pre-primitive
    hop: int  # Hop depth when fired
    inbox_sources: Dict[int, np.ndarray]  # sender_id -> their output that arrived


@dataclass
class GraphState:
    """
    Global MCT4 graph state.
    
    Manages node registry, execution queue, and convergence monitoring.
    """
    D: int = 512  # Vector dimensionality
    t_budget: int = 20  # Maximum hop count
    lambda_tau: float = 0.1  # Latency threshold steepness
    kappa: int = 0  # Convergence counter
    kappa_thresh: int = 100  # Threshold for convergence dampening
    N: int = 32  # Batch size
    
    # Node registry: id -> Node
    nodes: Dict[int, Node] = field(default_factory=dict)
    
    # Context vector
    context: Optional[Context] = None
    
    # Next node ID for creation
    next_node_id: int = 0
    
    # Track edges for retrograde flow: (src, dst) -> attributed_tension
    edge_tensions: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # Nodes that fired in current pass (for learning phase)
    active_path: List[ActivePathRecord] = field(default_factory=list)
    
    # Pruning events counter (for convergence)
    pruning_events_this_pass: int = 0
    
    def __post_init__(self):
        if self.context is None:
            self.context = Context(D=self.D)
    
    def get_tau(self, t: int) -> float:
        """
        Compute dynamic activation threshold.
        
        τ(t) = exp(λ_τ · (t − t_budget))
        
        As t → t_budget, τ → ∞, collapsing routing to shortest output path.
        
        For early hops (t << t_budget), tau is very small allowing easy activation.
        """
        # Use a scaled version that keeps tau reasonable
        # At t=0 with t_budget=20, lambda_tau=0.1: tau = exp(-2) ≈ 0.135
        # At t=t_budget: tau = exp(0) = 1
        # At t>t_budget: tau grows exponentially
        return np.exp(self.lambda_tau * (t - self.t_budget)) * 0.5
    
    def create_node(self, node_type: NodeType = NodeType.HIDDEN,
                    primitive: Primitive = Primitive.GELU,
                    rho_base: float = 0.5) -> Node:
        """Create and register a new node.
        
        Args:
            node_type: Type of node (INPUT, HIDDEN, OUTPUT)
            primitive: Primitive operator for this node
            rho_base: Initial health/base activation bias (default 0.5 for easier firing)
        """
        node = Node(
            id=self.next_node_id,
            D=self.D,
            rho_base=rho_base,
            primitive=primitive,
            node_type=node_type
        )
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node
    
    def add_edge(self, src_id: int, dst_id: int):
        """Add directed edge from src to dst."""
        if src_id not in self.nodes or dst_id not in self.nodes:
            raise ValueError(f"Invalid edge: {src_id} -> {dst_id}")
        
        if dst_id not in self.nodes[src_id].edges_out:
            self.nodes[src_id].edges_out.append(dst_id)
        
        if src_id not in self.nodes[dst_id].edges_in:
            self.nodes[dst_id].edges_in.append(src_id)
    
    def remove_edge(self, src_id: int, dst_id: int):
        """Remove directed edge from src to dst."""
        if src_id in self.nodes:
            if dst_id in self.nodes[src_id].edges_out:
                self.nodes[src_id].edges_out.remove(dst_id)
        
        if dst_id in self.nodes:
            if src_id in self.nodes[dst_id].edges_in:
                self.nodes[dst_id].edges_in.remove(src_id)
        
        # Clear tension tracking for this edge
        self.edge_tensions.pop((src_id, dst_id), None)
    
    def remove_node(self, node_id: int):
        """Remove a node and all its edges."""
        if node_id not in self.nodes:
            return
        
        # Remove all outgoing edges
        for dst_id in list(self.nodes[node_id].edges_out):
            self.remove_edge(node_id, dst_id)
        
        # Remove all incoming edges
        for src_id in list(self.nodes[node_id].edges_in):
            self.remove_edge(src_id, node_id)
        
        del self.nodes[node_id]
    
    def get_input_nodes(self) -> List[Node]:
        """Get all input nodes."""
        return [n for n in self.nodes.values() if n.node_type == NodeType.INPUT]
    
    def get_output_nodes(self) -> List[Node]:
        """Get all output nodes."""
        return [n for n in self.nodes.values() if n.node_type == NodeType.OUTPUT]
    
    def is_acyclic(self) -> bool:
        """Check if graph is a DAG using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: int) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in self.nodes[node_id].edges_out:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return False
        
        return True
    
    def increment_kappa(self):
        """Increment convergence counter if no pruning occurred."""
        if self.pruning_events_this_pass == 0:
            self.kappa += 1
        else:
            self.kappa = 0
        
        self.pruning_events_this_pass = 0
    
    def is_converged(self) -> bool:
        """Check if graph has converged (stable structure)."""
        return self.kappa > self.kappa_thresh
