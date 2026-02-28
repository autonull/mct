"""
MCT5 Phase 3 — Structural Evolution

Continuous online topology mutation:
  - Pruning (Lysis):    Remove nodes with rho < 0 (I/O nodes immortal)
  - Capacity insertion: Spawn K new nodes on the highest-tension edge
  - Lateral wiring:    Add skip connections for chronically high-tension nodes
  - Auto-depth growth: Insert a depth layer when loss is stagnating

DAG acyclicity is always enforced via GraphState.add_edge's built-in check.
"""

from __future__ import annotations
import numpy as np
import random
from typing import Dict, List, Optional, Set, Tuple

from .types import GraphState, Node, NodeType
from .primitives import Primitive
from .residue import HolographicResidue


# Primitives allowed during random mutation
_MUTABLE_PRIMITIVES = [
    Primitive.RELU, Primitive.GELU, Primitive.TANH, Primitive.SWISH,
    Primitive.GATE, Primitive.BILINEAR, Primitive.QUADRATIC, Primitive.ADD,
    Primitive.PRODUCT,
]


class StructuralEvolution:
    """Manages structural mutations of the MCT5 graph."""

    def __init__(self,
                 state: GraphState,
                 residue: HolographicResidue,
                 sigma_mut: float = 0.05,
                 K: int = 2,
                 tau_lateral: float = 0.3,
                 quadratic_spawn_bias: float = 0.3,
                 kappa_thresh: int = 100):
        self.state = state
        self.residue = residue
        self.sigma_mut = sigma_mut
        self.K = K
        self.tau_lateral = tau_lateral
        self.quad_bias = quadratic_spawn_bias
        self.kappa_thresh = kappa_thresh

        # Track whether graph has any QUADRATIC node
        self._has_quadratic = False

    # ── Public API ────────────────────────────────────────────────────────────

    def prune(self) -> List[int]:
        """Remove hidden nodes with rho < 0. Returns pruned IDs."""
        pruned: List[int] = []
        for nid, node in list(self.state.nodes.items()):
            if node.node_type != NodeType.HIDDEN:
                continue
            if node.rho < 0.0:
                self.residue.release_node(nid)
                self.state.remove_node(nid)
                pruned.append(nid)
                self.state.pruning_events_this_pass += 1
        self._has_quadratic = any(
            n.primitive == Primitive.QUADRATIC
            for n in self.state.nodes.values()
        )
        return pruned

    def insert_capacity(self, edge: Optional[Tuple[int, int]] = None):
        """
        Spawn K new nodes along the highest-tension edge (or given edge).
        Each inherits from the upstream donor with Gaussian perturbation.
        """
        if edge is None:
            if not self.state.edge_tensions:
                return
            edge = max(self.state.edge_tensions.items(), key=lambda kv: kv[1])[0]

        u, v = edge
        if u not in self.state.nodes or v not in self.state.nodes:
            return

        node_u = self.state.nodes[u]
        sigma = self.sigma_mut
        if self.state.is_converged():
            sigma *= 0.5

        # Remove original edge (new nodes slot in between)
        self.state.remove_edge(u, v)

        new_ids: List[int] = []
        for _ in range(self.K):
            prim = self._choose_primitive(node_u.primitive)
            new_node = self.state.create_node(
                node_type=NodeType.HIDDEN,
                primitive=prim,
                rho=0.8,
            )
            self.residue.register_node(new_node.id)

            # Inherit weights + perturbation
            new_node.A = node_u.A + np.random.randn(*node_u.A.shape) * sigma
            new_node.B = node_u.B + np.random.randn(*node_u.B.shape) * sigma
            new_node.S = node_u.S + np.random.randn(node_u.D) * sigma
            new_node.S /= np.linalg.norm(new_node.S) + 1e-9
            new_node.tension_trace = node_u.tension_trace

            # Wire: u → new → v (acyclicity checked by add_edge)
            if self.state.add_edge(u, new_node.id):
                if not self.state.add_edge(new_node.id, v):
                    # If second edge fails, connect directly to an output instead
                    for out_node in self.state.output_nodes():
                        if self.state.add_edge(new_node.id, out_node.id):
                            break
            new_ids.append(new_node.id)

        return new_ids

    def lateral_wiring(self):
        """
        Nodes persistent under high tension (> 20 consecutive passes above threshold)
        grow one shortcut edge to a random downstream node.
        """
        for node in list(self.state.nodes.values()):
            if node.node_type != NodeType.HIDDEN:
                continue
            if node.high_tension_count > 20:
                downstream = self._downstream_of(node.id)
                existing = set(self.state.edges_out.get(node.id, []))
                candidates = [nid for nid in downstream if nid not in existing
                              and self.state.nodes[nid].node_type != NodeType.INPUT]
                if candidates:
                    target = random.choice(candidates)
                    # add_edge handles acyclicity
                    self.state.add_edge(node.id, target)
                node.high_tension_count = 0

    def auto_depth_growth(self, ema_loss_delta: float):
        """
        If loss is stagnating and max active depth ≤ 2, insert a new hidden layer.
        """
        if abs(ema_loss_delta) > 0.005:
            return  # Loss is still improving
        active_path = self.state.active_path
        if not active_path:
            return
        max_depth = max(r.hop for r in active_path)
        if max_depth > 2:
            return
        # Find edge with highest tension and insert capacity there
        self.insert_capacity()

    def ensure_nonlinearity(self):
        """
        If graph has no QUADRATIC node, inject one at the first opportunity
        (highest-tension edge from input).  Solves XOR-class problems.
        """
        if self._has_quadratic:
            return
        # Find highest-tension edge from any input node
        input_ids = {n.id for n in self.state.input_nodes()}
        best_edge = None
        best_val = -1.0
        for (u, v), t in self.state.edge_tensions.items():
            if u in input_ids and t > best_val:
                best_edge, best_val = (u, v), t
        if best_edge is None and self.state.edge_tensions:
            best_edge = max(self.state.edge_tensions.items(), key=lambda kv: kv[1])[0]
        if best_edge is None:
            return
        u, v = best_edge
        if u not in self.state.nodes or v not in self.state.nodes:
            return
        node_u = self.state.nodes[u]
        new_node = self.state.create_node(NodeType.HIDDEN, Primitive.QUADRATIC, rho=1.0)
        self.residue.register_node(new_node.id)
        new_node.A = node_u.A.copy()
        new_node.B = node_u.B.copy()
        new_node.S = node_u.S + np.random.randn(node_u.D) * 0.05
        new_node.S /= np.linalg.norm(new_node.S) + 1e-9
        self.state.remove_edge(u, v)
        self.state.add_edge(u, new_node.id)
        self.state.add_edge(new_node.id, v)
        self._has_quadratic = True

    def evolve(self, pruned: List[int],
               max_tension_edge: Optional[Tuple[int, int]],
               ema_loss: float,
               ema_loss_prev: float):
        """Main evolution step called by engine after each learning pass."""
        # Capacity insertion for pruned nodes
        if pruned:
            self.insert_capacity(max_tension_edge)

        # Ensure non-linear capacity for compositional tasks (XOR, Circles)
        self.ensure_nonlinearity()

        # Lateral wiring for chronically high-tension nodes
        self.lateral_wiring()

        # Auto-depth growth if stagnating
        delta = ema_loss - ema_loss_prev
        self.auto_depth_growth(delta)

    # ── Graph initialisation ──────────────────────────────────────────────────

    def initialize_graph(self, primitive_hidden: Primitive = Primitive.GELU):
        """
        Build the minimal starter graph.

        Input → GELU → ADD (aggregator) → SOFTMAX output
              ↗ GELU ↗
              ↗ QUADRATIC ↗  (non-linear capacity from birth)
        """
        self.state.nodes.clear()
        self.state.edges_out.clear()
        self.state.edges_in.clear()
        self.state.next_id = 0
        self.state.edge_tensions.clear()
        self.state._topo_valid = False

        # Input node
        inp = self.state.create_node(NodeType.INPUT, Primitive.FORK, rho=3.0)
        self.residue.register_node(inp.id)

        # Hidden layer: two standard + one quadratic for non-linear capacity
        h1 = self.state.create_node(NodeType.HIDDEN, Primitive.GELU, rho=1.5)
        h2 = self.state.create_node(NodeType.HIDDEN, primitive_hidden, rho=1.5)
        h3 = self.state.create_node(NodeType.HIDDEN, Primitive.QUADRATIC, rho=1.5)
        agg = self.state.create_node(NodeType.HIDDEN, Primitive.ADD, rho=1.5)

        for h in [h1, h2, h3]:
            self.residue.register_node(h.id)
        self.residue.register_node(agg.id)

        # Output node
        out = self.state.create_node(NodeType.OUTPUT, Primitive.SOFTMAX, rho=2.0)
        self.residue.register_node(out.id)

        # Wire
        for h in [h1, h2, h3]:
            self.state.add_edge(inp.id, h.id)
            self.state.add_edge(h.id, agg.id)

        # Skip connections from hidden to output
        self.state.add_edge(inp.id, agg.id)
        self.state.add_edge(agg.id, out.id)
        # Direct skip: inp → out
        self.state.add_edge(inp.id, out.id)

        self._has_quadratic = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _choose_primitive(self, parent_prim: Primitive) -> Primitive:
        """Select primitive for a new spawned node."""
        if not self._has_quadratic and random.random() < self.quad_bias:
            return Primitive.QUADRATIC
        if random.random() < 0.3:
            return random.choice(_MUTABLE_PRIMITIVES)
        return parent_prim

    def _downstream_of(self, node_id: int) -> List[int]:
        """BFS to collect all nodes reachable from node_id."""
        visited: Set[int] = set()
        queue = list(self.state.edges_out.get(node_id, []))
        while queue:
            nid = queue.pop(0)
            if nid in visited or nid not in self.state.nodes:
                continue
            visited.add(nid)
            queue.extend(self.state.edges_out.get(nid, []))
        return list(visited)
