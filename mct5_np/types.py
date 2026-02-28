"""
MCT5 Core Data Types

Node, GraphState, and associated records.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

from .primitives import Primitive


class NodeType(Enum):
    INPUT  = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


@dataclass
class Node:
    """
    A node in the MCT5 graph.

    Parameters are stored as low-rank factors:  W  =  A @ B.T
    where  A, B ∈ ℝᴰˣʳ.  This gives 2Dr parameters per node instead of D².
    """
    id: int
    D: int   # vector dimension
    r: int   # low-rank dimension

    node_type: NodeType = NodeType.HIDDEN
    primitive: Primitive = Primitive.GELU

    # Routing signature — learnable geometric embedding
    S: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Low-rank weight factors.  Effective W = A @ B.T
    A: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    B: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Health scalar
    rho: float = 1.0

    # Adam momentum/variance buffers for A, B
    m_A: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    v_A: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    m_B: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    v_B: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    adam_t: int = 0  # per-node Adam step counter

    # Health EMA — used for routing signature LR scaling
    tension_trace: float = 0.0
    high_tension_count: int = 0

    # Topo metadata
    topo_depth: int = 0       # Layer depth in the current topo ordering
    steps_idle: int = 0       # Hops since last firing
    last_attributed_tension: float = 0.0

    def __post_init__(self):
        if self.D > 0 and len(self.S) == 0:
            self._init_arrays()

    def _init_arrays(self):
        D, r = self.D, self.r
        # Random unit-sphere signature
        self.S = np.random.randn(D)
        self.S /= np.linalg.norm(self.S) + 1e-9

        # Low-rank init: A, B ≈ He-style
        scale = np.sqrt(2.0 / (D + r))
        self.A = np.random.randn(D, r) * scale
        self.B = np.random.randn(D, r) * scale

        # Bias near zero
        self.bias = np.zeros(D)

        # Adam buffers
        self.m_A = np.zeros((D, r))
        self.v_A = np.zeros((D, r))
        self.m_B = np.zeros((D, r))
        self.v_B = np.zeros((D, r))

    # ── Weight helpers ────────────────────────────────────────────────────────

    def get_W(self) -> np.ndarray:
        """Reconstruct full weight matrix (D×D)."""
        return self.A @ self.B.T

    def apply_W(self, x: np.ndarray) -> np.ndarray:
        """Efficient low-rank matmul:  W @ x  =  A @ (Bᵀ x)."""
        return self.A @ (self.B.T @ x)

    def apply_W_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch apply: X ∈ ℝᴺˣᴰ → ℝᴺˣᴰ."""
        # X @ B  (N×r)  →  * A.T  → (N×D)
        return (X @ self.B) @ self.A.T

    def spectral_norm(self) -> float:
        """Approximate spectral norm of W = A @ B.T via SVD of A."""
        s_A = np.linalg.svd(self.A, compute_uv=False)
        s_B = np.linalg.svd(self.B, compute_uv=False)
        return float(s_A[0] * s_B[0])


@dataclass
class ActiveRecord:
    """Records one node's activity during a forward pass."""
    node_id:  int
    V_in:     np.ndarray   # Aggregated pre-activation input  (D,)
    V_weighted: np.ndarray # Post-W, pre-primitive             (D,)
    V_out:    np.ndarray   # Post-primitive output             (D,)
    hop:      int
    # Sender contributions for blame attribution
    inbox_contributions: Dict[int, np.ndarray] = field(default_factory=dict)
    # ||V_out||² goodness
    goodness: float = 0.0


@dataclass
class GraphState:
    """
    Global graph state.

    Maintains the node registry, adjacency, topo ordering, and the
    Holographic Residue (imported lazily to avoid circular imports).
    """
    D: int = 64
    r: int = 16
    t_budget: int = 15
    lambda_tau: float = 0.15
    kappa: int = 0
    kappa_thresh: int = 100

    # Node registry
    nodes: Dict[int, Node] = field(default_factory=dict)
    next_id: int = 0

    # Adjacency (ahead-of-node-lists for cache efficiency)
    edges_out: Dict[int, List[int]] = field(default_factory=dict)  # src → [dst]
    edges_in:  Dict[int, List[int]] = field(default_factory=dict)  # dst → [src]

    # Cached topological order (invalidated on structural change)
    _topo_order: List[int] = field(default_factory=list)
    _topo_valid: bool = False

    # Active-path records from last forward pass
    active_path: List[ActiveRecord] = field(default_factory=list)

    # Edge tension from last learning pass (for structural evolution)
    edge_tensions: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Convergence
    pruning_events_this_pass: int = 0
    step_count: int = 0

    # ── Constructors ──────────────────────────────────────────────────────────

    def create_node(self,
                    node_type: NodeType = NodeType.HIDDEN,
                    primitive: Primitive = Primitive.GELU,
                    rho: float = 1.0) -> Node:
        node = Node(
            id=self.next_id,
            D=self.D,
            r=self.r,
            node_type=node_type,
            primitive=primitive,
            rho=rho,
        )
        self.nodes[self.next_id] = node
        self.edges_out[self.next_id] = []
        self.edges_in[self.next_id] = []
        self.next_id += 1
        self._topo_valid = False
        return node

    # ── Edge management ───────────────────────────────────────────────────────

    def add_edge(self, src: int, dst: int) -> bool:
        """Add edge src→dst; return True if acyclicity check passes."""
        if src not in self.nodes or dst not in self.nodes:
            return False
        if dst in self.edges_out.get(src, []):
            return True  # already exists
        self.edges_out[src].append(dst)
        self.edges_in[dst].append(src)
        self._topo_valid = False
        if not self.is_acyclic():
            # Revert
            self.edges_out[src].remove(dst)
            self.edges_in[dst].remove(src)
            self._topo_valid = False
            return False
        return True

    def remove_edge(self, src: int, dst: int):
        if src in self.edges_out and dst in self.edges_out[src]:
            self.edges_out[src].remove(dst)
        if dst in self.edges_in and src in self.edges_in[dst]:
            self.edges_in[dst].remove(src)
        self.edge_tensions.pop((src, dst), None)
        self._topo_valid = False

    def remove_node(self, node_id: int):
        if node_id not in self.nodes:
            return
        for dst in list(self.edges_out.get(node_id, [])):
            self.remove_edge(node_id, dst)
        for src in list(self.edges_in.get(node_id, [])):
            self.remove_edge(src, node_id)
        del self.nodes[node_id]
        self.edges_out.pop(node_id, None)
        self.edges_in.pop(node_id, None)
        self._topo_valid = False

    # ── Topological ordering ──────────────────────────────────────────────────

    def topo_order(self) -> List[int]:
        """Kahn's algorithm.  Cached; recomputed only after structural changes."""
        if self._topo_valid:
            return self._topo_order
        in_deg = {nid: len(self.edges_in.get(nid, [])) for nid in self.nodes}
        queue = [nid for nid, d in in_deg.items() if d == 0]
        order: List[int] = []
        while queue:
            # stable sort for determinism
            queue.sort()
            n = queue.pop(0)
            order.append(n)
            for child in self.edges_out.get(n, []):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        self._topo_order = order
        self._topo_valid = True
        # Update depth metadata
        depth: Dict[int, int] = {}
        for nid in order:
            parents = self.edges_in.get(nid, [])
            depth[nid] = (max(depth[p] for p in parents) + 1) if parents else 0
            self.nodes[nid].topo_depth = depth[nid]
        return self._topo_order

    def is_acyclic(self) -> bool:
        in_deg = {nid: len(self.edges_in.get(nid, [])) for nid in self.nodes}
        queue = [nid for nid, d in in_deg.items() if d == 0]
        visited = 0
        while queue:
            n = queue.pop()
            visited += 1
            for child in self.edges_out.get(n, []):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        return visited == len(self.nodes)

    # ── Activation threshold ──────────────────────────────────────────────────

    def get_tau(self, t: int) -> float:
        """τ(t) = 0.5 · exp(λ_τ · (t − t_budget))"""
        return 0.5 * np.exp(self.lambda_tau * (t - self.t_budget))

    # ── Node selectors ────────────────────────────────────────────────────────

    def input_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.INPUT]

    def output_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.OUTPUT]

    def hidden_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.HIDDEN]

    # ── Convergence ───────────────────────────────────────────────────────────

    def tick(self):
        """Call once per training step."""
        if self.pruning_events_this_pass == 0:
            self.kappa += 1
        else:
            self.kappa = 0
        self.pruning_events_this_pass = 0
        self.step_count += 1

    def is_converged(self) -> bool:
        return self.kappa > self.kappa_thresh
