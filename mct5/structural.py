import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random

from .types import GraphState, Node, NodeType
from .primitives import Primitive
from .residue import HolographicResidue

_MUTABLE_PRIMITIVES = [
    Primitive.RELU, Primitive.GELU, Primitive.TANH, Primitive.SWISH,
    Primitive.GATE, Primitive.BILINEAR, Primitive.QUADRATIC, Primitive.ADD,
    Primitive.PRODUCT,
]

class StructuralEvolution:
    """Manages structural mutations of the MCT5 PyTorch graph."""

    def __init__(self,
                 state: GraphState,
                 residue: HolographicResidue,
                 sigma_mut: float = 0.05,
                 K: int = 2,
                 tau_lateral: float = 0.3,
                 quadratic_spawn_bias: float = 0.3):
        self.state = state
        self.residue = residue
        self.sigma_mut = sigma_mut
        self.K = K
        self.tau_lateral = tau_lateral
        self.quad_bias = quadratic_spawn_bias
        
        # In PyTorch, checking for quadratic is just a loop over ModuleDict
        self._has_quadratic = False

    # ── Public API ────────────────────────────────────────────────────────────

    def prune(self) -> List[int]:
        """Remove hidden nodes with rho < 0. Returns pruned IDs."""
        pruned: List[int] = []
        # Need to iterate over keys since we cannot modify dict while iterating
        for nid_str in list(self.state.nodes.keys()):
            node = self.state.nodes[nid_str]
            if node.node_type != NodeType.HIDDEN:
                continue
            if node.rho.item() < 0.0:
                nid = int(nid_str)
                self.residue.release_node(nid) # Assuming nid maps to internal residue index for simplicity here
                self.state.remove_node(nid)
                pruned.append(nid)
                self.state.pruning_events_this_pass += 1
                
        self._has_quadratic = any(n.primitive == Primitive.QUADRATIC for n in self.state.nodes.values())
        return pruned

    def insert_capacity(self, edge: Optional[Tuple[int, int]] = None) -> List[int]:
        """Spawn K new nodes along the highest-tension edge (or given edge)."""
        if edge is None:
            if not self.state.edge_tensions:
                return []
            edge = max(self.state.edge_tensions.items(), key=lambda kv: kv[1])[0]

        u, v = edge
        str_u, str_v = str(u), str(v)
        if str_u not in self.state.nodes or str_v not in self.state.nodes:
            return []

        node_u = self.state.nodes[str_u]
        sigma = self.sigma_mut
        if self.state.is_converged():
            sigma *= 0.5

        self.state.remove_edge(u, v)

        new_ids: List[int] = []
        for _ in range(self.K):
            prim = self._choose_primitive(node_u.primitive)
            new_node = self.state.create_node(node_type=NodeType.HIDDEN, primitive=prim, rho=0.8)
            self.residue.register_node(new_node.id)

            # Inherit weights + perturbation safely (without tracking gradients)
            with torch.no_grad():
                new_node.A.copy_(node_u.A + torch.randn_like(node_u.A) * sigma)
                new_node.B.copy_(node_u.B + torch.randn_like(node_u.B) * sigma)
                new_node.S.copy_(node_u.S + torch.randn_like(node_u.S) * sigma)
                new_node.S.div_(torch.norm(new_node.S) + 1e-9)
            
            new_node.tension_trace = node_u.tension_trace

            if self.state.add_edge(u, new_node.id):
                if not self.state.add_edge(new_node.id, v):
                    for out_node in self.state.output_nodes():
                        if self.state.add_edge(new_node.id, out_node.id):
                            break
            new_ids.append(new_node.id)
            
        return new_ids

    def lateral_wiring(self):
        """Nodes persistent under high tension grow one shortcut edge downstream."""
        for nid_str, node in self.state.nodes.items():
            if node.node_type != NodeType.HIDDEN:
                continue
            if node.high_tension_count > 20:
                nid = int(nid_str)
                downstream = self._downstream_of(nid)
                existing = set(self.state.edges_out.get(nid, []))
                candidates = [d for d in downstream if d not in existing 
                              and self.state.nodes[str(d)].node_type != NodeType.INPUT]
                if candidates:
                    target = random.choice(candidates)
                    self.state.add_edge(nid, target)
                node.high_tension_count = 0

    def auto_depth_growth(self, ema_loss_delta: float):
        """If loss is stagnating and depth is shallow, insert capacity."""
        if abs(ema_loss_delta) > 0.005:
            return
        
        # Max active depth approximated by topological order length
        # In a generic way we just trigger capacity insertion
        self.insert_capacity()

    def ensure_nonlinearity(self):
        """Inject QUADRATIC or PRODUCT primitive on high-tension input edges to enable non-linear composition."""
        if self._has_quadratic:
            return
            
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
        str_u, str_v = str(u), str(v)
        if str_u not in self.state.nodes or str_v not in self.state.nodes:
            return
            
        node_u = self.state.nodes[str_u]
        # Choose randomly between QUADRATIC and PRODUCT
        prim = random.choice([Primitive.QUADRATIC, Primitive.PRODUCT])
        
        new_node = self.state.create_node(NodeType.HIDDEN, prim, rho=1.0)
        self.residue.register_node(new_node.id)
        
        with torch.no_grad():
            new_node.A.copy_(node_u.A)
            new_node.B.copy_(node_u.B)
            new_node.S.copy_(node_u.S + torch.randn_like(node_u.S) * 0.05)
            new_node.S.div_(torch.norm(new_node.S) + 1e-9)
            
        self.state.remove_edge(u, v)
        self.state.add_edge(u, new_node.id)
        self.state.add_edge(new_node.id, v)
        self._has_quadratic = True

    def evolve(self, ema_loss: float, ema_loss_prev: float):
        """Main evolution step."""
        pruned = self.prune()
        if pruned:
            self.insert_capacity()
        
        self.ensure_nonlinearity()
        self.lateral_wiring()
        
        delta = ema_loss - ema_loss_prev
        self.auto_depth_growth(delta)

    # ── Graph Initialisation ──────────────────────────────────────────────────
    
    def initialize_graph(self, primitive_hidden: Primitive = Primitive.GELU):
        self.state.nodes.clear()
        self.state.edges_out.clear()
        self.state.edges_in.clear()
        self.state.next_id = 0
        self.state.edge_tensions.clear()
        self.state._topo_valid = False

        inp = self.state.create_node(NodeType.INPUT, Primitive.FORK, rho=3.0)
        self.residue.register_node(inp.id)

        h1 = self.state.create_node(NodeType.HIDDEN, Primitive.GELU, rho=1.5)
        h2 = self.state.create_node(NodeType.HIDDEN, primitive_hidden, rho=1.5)
        h3 = self.state.create_node(NodeType.HIDDEN, Primitive.QUADRATIC, rho=1.5)
        h4 = self.state.create_node(NodeType.HIDDEN, Primitive.PRODUCT, rho=1.5)
        agg = self.state.create_node(NodeType.HIDDEN, Primitive.ADD, rho=1.5)

        for h in [h1, h2, h3, h4]:
            self.residue.register_node(h.id)
        self.residue.register_node(agg.id)

        out = self.state.create_node(NodeType.OUTPUT, Primitive.FORK, rho=2.0)
        self.residue.register_node(out.id)

        for h in [h1, h2, h3, h4]:
            self.state.add_edge(inp.id, h.id)
            self.state.add_edge(h.id, agg.id)

        self.state.add_edge(inp.id, agg.id)
        self.state.add_edge(agg.id, out.id)
        self.state.add_edge(inp.id, out.id)

        self._has_quadratic = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _choose_primitive(self, parent_prim: Primitive) -> Primitive:
        if not self._has_quadratic and random.random() < self.quad_bias:
            return random.choice([Primitive.QUADRATIC, Primitive.PRODUCT])
        if random.random() < 0.3:
            return random.choice(_MUTABLE_PRIMITIVES)
        return parent_prim

    def _downstream_of(self, node_id: int) -> List[int]:
        visited: set = set()
        queue = list(self.state.edges_out.get(node_id, []))
        while queue:
            nid = queue.pop(0)
            if nid in visited or str(nid) not in self.state.nodes:
                continue
            visited.add(nid)
            queue.extend(self.state.edges_out.get(nid, []))
        return list(visited)
