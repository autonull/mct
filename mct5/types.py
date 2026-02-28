import torch
import torch.nn as nn
from enum import Enum
from typing import Dict, List, Optional
from collections import defaultdict
from .primitives import Primitive

class NodeType(Enum):
    INPUT  = 0
    HIDDEN = 1
    OUTPUT = 2

class Node(nn.Module):
    """
    Lean parameterized node in the MCT5 PyTorch graph.
    """
    def __init__(self, node_id: int, node_type: NodeType, primitive: Primitive,
                 D: int, r: int, rho: float, device: str = "cpu"):
        super().__init__()
        self.id = node_id
        self.node_type = node_type
        self.primitive = primitive
        self.D = D
        self.r = r
        
        # ── Learnable Parameters ────────────────────────────────────────────────
        # S: routing signature (D)
        self.S = nn.Parameter(torch.randn(D, device=device) / torch.sqrt(torch.tensor(D, dtype=torch.float32)))
        
        # Low-rank factors: W = A @ B^T
        # Init A, B with small numbers
        self.A = nn.Parameter(torch.randn(D, r, device=device) * 0.05)
        self.B = nn.Parameter(torch.randn(D, r, device=device) * 0.05)
        
        self.bias = nn.Parameter(torch.zeros(D, device=device))

        # ── Buffers (State) ─────────────────────────────────────────────────────
        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float32, device=device))
        
        # Structural health tracking (not autograd)
        self.steps_idle = 0
        self.tension_trace = 0.0
        self.high_tension_count = 0
        self.last_attributed_tension = 0.0

    @property
    def W(self) -> torch.Tensor:
        """Assembles the full DxD weight matrix."""
        return self.A @ self.B.T

class GraphState(nn.Module):
    """
    The dynamic MCT5 topology.
    """
    def __init__(self, D: int, r: int, device: str = "cpu"):
        super().__init__()
        self.D = D
        self.r = r
        self.device = device
        # PyTorch requires parameters to be in ModuleDict/ModuleList to register
        self.nodes = nn.ModuleDict()
        # DAG structure (adjacency lists)
        self.edges_out: Dict[int, List[int]] = defaultdict(list)
        self.edges_in:  Dict[int, List[int]] = defaultdict(list)
        
        self.next_id = 0
        self._topo_valid = False
        self.topo_order: List[int] = []
        
        # Tensions computed during backward pass (manual tracking for structure)
        self.edge_tensions: Dict[tuple, float] = defaultdict(float)
        
        self.passes_since_mutation = 0
        self.pruning_events_this_pass = 0

    def create_node(self, node_type: NodeType, primitive: Primitive, rho: float = 1.0) -> Node:
        nid = self.next_id
        self.next_id += 1
        node = Node(nid, node_type, primitive, self.D, self.r, rho, self.device)
        self.nodes[str(nid)] = node
        self._topo_valid = False
        return node

    def add_edge(self, u: int, v: int) -> bool:
        """Adds u->v if valid DAG, else returns False."""
        if str(u) not in self.nodes or str(v) not in self.nodes:
            return False
        if v in self.edges_out[u]:
            return True # already exists
        if u == v:
            return False # no self-loops

        self.edges_out[u].append(v)
        self.edges_in[v].append(u)
        
        if not self.is_acyclic():
            self.edges_out[u].remove(v)
            self.edges_in[v].remove(u)
            return False
            
        self._topo_valid = False
        return True

    def remove_edge(self, u: int, v: int):
        if v in self.edges_out.get(u, []):
            self.edges_out[u].remove(v)
        if u in self.edges_in.get(v, []):
            self.edges_in[v].remove(u)
        self._topo_valid = False
        self.edge_tensions.pop((u, v), None)

    def remove_node(self, nid: int):
        str_id = str(nid)
        if str_id not in self.nodes:
            return
        
        for dst in list(self.edges_out.get(nid, [])):
            self.remove_edge(nid, dst)
        for src in list(self.edges_in.get(nid, [])):
            self.remove_edge(src, nid)
            
        del self.nodes[str_id]
        if nid in self.edges_out: del self.edges_out[nid]
        if nid in self.edges_in:  del self.edges_in[nid]
        self._topo_valid = False

    def is_acyclic(self) -> bool:
        visited = set()
        path = set()
        
        def visit(n: int) -> bool:
            if n in path: return False
            if n in visited: return True
            visited.add(n)
            path.add(n)
            for m in self.edges_out.get(n, []):
                if not visit(m):
                    return False
            path.remove(n)
            return True
            
        return all(visit(int(n)) for n in self.nodes.keys())

    def get_topo_order(self) -> List[int]:
        if self._topo_valid:
            return self.topo_order
            
        in_degree = {int(n): len(self.edges_in.get(int(n), [])) for n in self.nodes.keys()}
        queue = [n for n, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for nxt in self.edges_out.get(curr, []):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
                    
        self.topo_order = order
        self._topo_valid = True
        return order

    def input_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.INPUT]

    def output_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.OUTPUT]

    def is_converged(self) -> bool:
        return self.passes_since_mutation > 100
