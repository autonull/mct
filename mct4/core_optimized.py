"""
MCT4 Core Data Types - OPTIMIZED VERSION

High-performance implementations with vectorized operations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import heapq

# Import from original core to ensure enum compatibility
from .core import NodeType
from .primitives import Primitive, apply_primitive


@dataclass
class Node:
    """
    Optimized node with vectorized batch support.
    """
    id: int
    D: int
    
    # Routing signature
    S: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    # Health scalar
    rho_base: float = 0.5
    
    # Weight matrix (full or factored)
    W: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    W_factored: bool = False
    A: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    B: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    rank: int = 64
    
    # Primitive
    primitive: Primitive = Primitive.GELU
    
    # Optimized inbox: list of (sender_id, vector_array, time_array) for batch
    inbox_senders: List[int] = field(default_factory=list)
    inbox_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (n_messages, D)
    inbox_times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    
    # Edges
    edges_out: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    edges_in: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    
    # State
    steps_idle: int = 0
    tension_trace: float = 0.0
    node_type: NodeType = NodeType.HIDDEN
    high_tension_count: int = 0
    last_hop: int = -1
    
    # Cached values
    _S_norm_sq: float = 0.0
    
    def __post_init__(self):
        if self.D > 0:
            if len(self.S) == 0:
                self.S = np.random.randn(self.D)
                self.S /= np.linalg.norm(self.S) + 1e-9
            self._S_norm_sq = np.dot(self.S, self.S)
            
            if self.W.shape == (0, 0):
                epsilon = 0.01
                if self.W_factored and self.rank < self.D:
                    self.A = np.random.randn(self.D, self.rank) * 0.01
                    self.B = np.random.randn(self.D, self.rank) * 0.01
                else:
                    self.W = np.eye(self.D) * epsilon
    
    def get_weight_matrix(self) -> np.ndarray:
        if self.W_factored:
            return self.A @ self.B.T
        return self.W
    
    def apply_weight(self, x: np.ndarray) -> np.ndarray:
        if self.W_factored:
            return self.A @ (self.B.T @ x)
        return self.W @ x
    
    def apply_weight_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply weight to batch of vectors (N, D) -> (N, D)."""
        if self.W_factored:
            return (X @ self.B) @ self.A.T
        return X @ self.W.T
    
    def update_weight_full(self, delta: np.ndarray):
        self.W = self.W + delta
    
    def update_weight_factored(self, T_local: np.ndarray, V_in: np.ndarray, eta: float):
        BtV = self.B.T @ V_in
        AtT = self.A.T @ T_local
        self.A = self.A + eta * np.outer(T_local, BtV)
        self.B = self.B + eta * np.outer(V_in, AtT)
    
    def clamp_weight_norm(self, W_max: float):
        if self.W_factored:
            for factor in [self.A, self.B]:
                norm = np.linalg.norm(factor, 'fro')
                if norm > W_max:
                    factor *= W_max / norm
        else:
            norm = np.linalg.norm(self.W, 'fro')
            if norm > W_max:
                self.W *= W_max / norm
    
    def clear_inbox(self):
        """Clear inbox for new forward pass."""
        self.inbox_senders = []
        self.inbox_vectors = np.zeros((0, self.D))
        self.inbox_times = np.zeros(0, dtype=np.int32)
    
    def add_to_inbox(self, sender_id: int, vectors: np.ndarray, time: int):
        """Add vectors to inbox (batched)."""
        n_new = len(vectors)
        if n_new == 0:
            return
        
        # Append to inbox
        self.inbox_senders.append(sender_id)
        self.inbox_vectors = np.vstack([self.inbox_vectors, vectors]) if len(self.inbox_vectors) > 0 else vectors
        self.inbox_times = np.append(self.inbox_times, np.full(n_new, time, dtype=np.int32))
    
    def aggregate_inbox(self, current_t: int, lambda_async: float, N: int) -> np.ndarray:
        """
        Aggregate inbox messages with decay.
        Returns (N, D) array for batch.
        """
        if len(self.inbox_vectors) == 0:
            return np.zeros((N, self.D))
        
        # Compute decay for all messages
        ages = current_t - self.inbox_times
        decays = np.exp(-lambda_async * ages)
        
        # Apply decay and aggregate
        decayed = self.inbox_vectors * decays[:, np.newaxis]
        
        # Mean aggregate
        aggregated = np.mean(decayed, axis=0)
        
        # Return batch (same for all in simple mode)
        return np.tile(aggregated, (N, 1))
    
    def fire(self, t: int, V_in: np.ndarray) -> np.ndarray:
        """Execute node on batch input (N, D) -> (N, D)."""
        # Apply weight matrix then primitive
        V_weighted = self.apply_weight_batch(V_in)
        
        # Apply primitive to each sample
        V_out = np.array([apply_primitive(self.primitive, v) for v in V_weighted])
        
        self.steps_idle = 0
        self.last_hop = t
        
        return V_out


@dataclass
class Context:
    """Optimized context vector."""
    D: int
    C: np.ndarray = field(default_factory=lambda: np.zeros(0))
    decay_c: float = 0.95
    
    def __post_init__(self):
        if len(self.C) == 0:
            self.C = np.zeros(self.D)
    
    def reset(self):
        self.C = np.zeros(self.D)
    
    def add_ghost_batch(self, rhos: np.ndarray, S: np.ndarray):
        """Add ghost signals from batch."""
        avg_rho = np.mean(rhos)
        self.C = self.C * self.decay_c + (avg_rho / self.D) * S
    
    def decay(self):
        self.C = self.C * self.decay_c


@dataclass 
class ActivePathRecord:
    """Record of nodes that fired during a forward pass."""
    node_id: int
    V_in: np.ndarray  # (N, D) batch
    V_out: np.ndarray  # (N, D) batch
    V_weighted: np.ndarray  # (N, D) batch
    hop: int
    inbox_sources: Dict[int, np.ndarray]


@dataclass
class GraphState:
    """Optimized graph state."""
    D: int = 512
    t_budget: int = 20
    lambda_tau: float = 0.1
    kappa: int = 0
    kappa_thresh: int = 100
    N: int = 32
    
    nodes: Dict[int, Node] = field(default_factory=dict)
    context: Optional[Context] = None
    next_node_id: int = 0
    edge_tensions: Dict[Tuple[int, int], float] = field(default_factory=dict)
    active_path: List[ActivePathRecord] = field(default_factory=list)
    pruning_events_this_pass: int = 0
    
    # Cached tau values
    _tau_cache: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    def __post_init__(self):
        if self.context is None:
            self.context = Context(D=self.D)
        self._update_tau_cache()
    
    def _update_tau_cache(self):
        """Pre-compute tau values for all hops."""
        self._tau_cache = np.exp(self.lambda_tau * (np.arange(self.t_budget + 2) - self.t_budget)) * 0.5
    
    def get_tau(self, t: int) -> float:
        if t < len(self._tau_cache):
            return self._tau_cache[t]
        return np.exp(self.lambda_tau * (t - self.t_budget)) * 0.5
    
    def create_node(self, node_type: NodeType = NodeType.HIDDEN,
                    primitive: Primitive = Primitive.GELU,
                    rho_base: float = 0.5) -> Node:
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
        if src_id not in self.nodes or dst_id not in self.nodes:
            return
        
        node = self.nodes[src_id]
        if dst_id not in node.edges_out:
            node.edges_out = np.append(node.edges_out, dst_id)
        
        node = self.nodes[dst_id]
        if src_id not in node.edges_in:
            node.edges_in = np.append(node.edges_in, src_id)
    
    def remove_edge(self, src_id: int, dst_id: int):
        if src_id in self.nodes:
            self.nodes[src_id].edges_out = self.nodes[src_id].edges_out[
                self.nodes[src_id].edges_out != dst_id
            ]
        if dst_id in self.nodes:
            self.nodes[dst_id].edges_in = self.nodes[dst_id].edges_in[
                self.nodes[dst_id].edges_in != src_id
            ]
        self.edge_tensions.pop((src_id, dst_id), None)
    
    def remove_node(self, node_id: int):
        if node_id not in self.nodes:
            return
        
        for dst_id in list(self.nodes[node_id].edges_out):
            self.remove_edge(node_id, dst_id)
        for src_id in list(self.nodes[node_id].edges_in):
            self.remove_edge(src_id, node_id)
        
        del self.nodes[node_id]
    
    def get_input_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.INPUT]
    
    def get_output_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.OUTPUT]
    
    def is_acyclic(self) -> bool:
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
        if self.pruning_events_this_pass == 0:
            self.kappa += 1
        else:
            self.kappa = 0
        self.pruning_events_this_pass = 0
    
    def is_converged(self) -> bool:
        return self.kappa > self.kappa_thresh
