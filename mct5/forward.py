import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from .types import GraphState, Node, NodeType
from .primitives import apply_primitive, Primitive
from .residue import HolographicResidue

@dataclass
class BatchedActiveRecord:
    node_id: int
    hop: int
    V_in: torch.Tensor       # (B, D) Aggregated input
    V_weighted: torch.Tensor # (B, D) After W transform
    V_out: torch.Tensor      # (B, D) After primitive
    
    # Store for structural debugging; gradients automatically handled by autograd
    goodness: torch.Tensor   # (B,) Contrastive goodness score
    inbox_contributions: Dict[int, torch.Tensor] # Maps src_id -> (B, D) chunk

class ForwardExecutor(nn.Module):
    def __init__(self, state: GraphState, residue: HolographicResidue):
        super().__init__()
        self.state = state
        self.residue = residue
        self.active_path: List[BatchedActiveRecord] = []

    def forward(self, X_batch: torch.Tensor, t_budget: int = 12) -> torch.Tensor:
        """
        Batched, vectorized depth-layer forward execution.
        X_batch: (B, input_dim)
        Returns: (B, D) or (B, n_classes) depending on output node behavior.
        """
        B = X_batch.size(0)
        self.active_path.clear()
        
        # 1. Group nodes by topological depth
        topo = self.state.get_topo_order()
        depths: Dict[int, int] = {}
        for nid in topo:
            in_edges = self.state.edges_in.get(nid, [])
            if not in_edges:
                depths[nid] = 0
            else:
                depths[nid] = max(depths.get(src, 0) for src in in_edges) + 1
                
        # Invert to `layers[d] = [nid1, nid2, ...]`
        layers: Dict[int, List[int]] = {}
        for nid, d in depths.items():
            layers.setdefault(d, []).append(nid)
            
        max_depth = max(layers.keys()) if layers else 0
        
        # 2. Node outputs buffer: nid_str -> (B, D) tensor
        node_outputs: Dict[str, torch.Tensor] = {}
        
        # We need a shared timing mechanism across layers for the "Anytime" threshold
        # For simplicity in testing XOR, we just run all topological layers.
        # But we implement the ρ firing logic based on Holographic residue.

        for d in range(max_depth + 1):
            if d > t_budget:
                break # Hard anytime deadline
                
            layer_nodes = layers.get(d, [])
            if not layer_nodes:
                continue
                
            for nid in layer_nodes:
                node = self.state.nodes[str(nid)]
                
                # ── a. Aggregate inputs ───────────────────────────────────────
                if node.node_type == NodeType.INPUT:
                    # Pad X_batch (B, input_dim) to (B, D)
                    pad_dim = node.D - X_batch.shape[1]
                    assert pad_dim >= 0, "Input dimension exceeds node D"
                    if pad_dim > 0:
                        padding = torch.zeros(B, pad_dim, device=X_batch.device, dtype=X_batch.dtype)
                        V_in = torch.cat([X_batch, padding], dim=1)
                    else:
                        V_in = X_batch
                    inbox_contribs = {}
                else:
                    # Collect from upstream senders
                    in_edges = self.state.edges_in.get(nid, [])
                    inbox_contribs = {src: node_outputs[str(src)] for src in in_edges if str(src) in node_outputs}
                    
                    if not inbox_contribs:
                        # Should not happen in valid topo unless disconnected
                        V_in = torch.zeros(B, node.D, device=X_batch.device)
                    elif len(inbox_contribs) == 1:
                        V_in = list(inbox_contribs.values())[0]
                    else:
                        V_in = torch.stack(list(inbox_contribs.values()), dim=0).sum(dim=0)
                
                # ── b. Activation Potential & Firing ──────────────────────────
                # ρ = node.rho + <S, V_in> + Decode(residue)
                s_dot = (V_in * node.S.unsqueeze(0)).sum(dim=-1) # (B,)
                
                # For PyTorch, we need the residue decoding index to match.
                # In this refactor, let's keep it simple: residue index == node.id
                # (Assuming residue was initialized sequentially or holds a mapping)
                res_decode = self.residue.decode(nid) # Scalar
                
                rho = getattr(node, 'rho') # buffer
                potential = rho + s_dot + res_decode
                
                # Base firing threshold is 0. If potential < 0, node is idle this pass.
                # (For Autograd consistency, we use a soft mask or binary multiplication 
                # but keep the gradient flowing if it fires).
                # Actually, standard MCT threshold is temporal tau(t). We'll use a simple threshold.
                mask = (potential > 0.0).float().unsqueeze(1) # (B, 1)
                
                # ── c. Linear Transform & Primitive ───────────────────────────
                W = node.W # (D, D)   (Computed from A @ B.T)
                
                if len(inbox_contribs) >= 2 and node.primitive in [Primitive.ADD, Primitive.GATE, Primitive.BILINEAR, Primitive.CONCAT_PROJECT, Primitive.PRODUCT]:
                    # Multi-input primitives require a list of tensors for proper cross-terms
                    # We apply W to each incoming signal separately
                    V_weighted_list = [torch.matmul(v, W.t()) + node.bias for v in inbox_contribs.values()]
                    V_out = apply_primitive(node.primitive, V_weighted_list)
                else:
                    # Single input or unary primitive
                    V_weighted = torch.matmul(V_in, W.t()) + node.bias
                    V_out = apply_primitive(node.primitive, V_weighted)
                
                # Store output
                node_outputs[str(nid)] = V_out
                
                # ── d. Goodness & Record ──────────────────────────────────────
                goodness = (V_out ** 2).sum(dim=-1) # (B,)
                record = BatchedActiveRecord(
                    node_id=nid,
                    hop=d,
                    V_in=V_in,
                    V_weighted=V_weighted,
                    V_out=V_out,
                    goodness=goodness,
                    inbox_contributions=inbox_contribs
                )
                self.active_path.append(record)
                
                # ── e. Update Structural Stats ────────────────────────────────
                # (No autograd here, just basic integer updates for pruning)
                firing_ratio = mask.mean().item()
                if firing_ratio < 0.05:
                    node.steps_idle += 1
                else:
                    node.steps_idle = 0

                # ── f. Ghost Injection to Residue ─────────────────────────────
                if firing_ratio > 0.0:
                    self.residue.inject_ghost(nid, rho=float(rho.item()), t=float(d))

        # 3. Aggregate network output from output nodes
        # If multiple output nodes fired, average them.
        out_nids = [n.id for n in self.state.output_nodes()]
        fired_outs = [node_outputs[str(nid)] for nid in out_nids if str(nid) in node_outputs]
        
        if not fired_outs:
            return torch.zeros(B, self.state.D, device=X_batch.device)
            
        final_out = torch.stack(fired_outs, dim=0).mean(dim=0)
        return final_out
