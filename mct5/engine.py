import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from .config import MCT5Config
from .types import GraphState, NodeType, Node
from .residue import HolographicResidue
from .forward import ForwardExecutor
from .structural import StructuralEvolution

class MCT5(nn.Module):
    """
    MCT5 PyTorch Top-Level API.
    Handles exact credit assignment via autograd.
    """
    def __init__(self, config: MCT5Config):
        super().__init__()
        self.cfg = config
        self.device = torch.device(config.device)
        
        self.residue = HolographicResidue(max_nodes=config.max_nodes, D=config.D, 
                                          phi_max=config.phi_max, device=self.device)
        self.state = GraphState(D=config.D, r=config.r, device=self.device)
        self.forward_executor = ForwardExecutor(self.state, self.residue)
        self.evolver = StructuralEvolution(
            self.state, self.residue,
            sigma_mut=config.sigma_mut, K=config.K,
            tau_lateral=config.tau_lateral,
            quadratic_spawn_bias=config.quadratic_spawn_bias
        )
        
        self.opt_W = None
        self.opt_S = None
        self.step_count = 0
        self.ema_loss = 1.0
        self.prev_ema_loss = 1.0

        # Classification output projection
        # MCT5 produces an output in R^D from the topology. We project this to n_classes.
        self.classifier = nn.Linear(config.D, config.n_classes, device=self.device)

    def initialize(self):
        """Builds the minimal starter graph."""
        self.evolver.initialize_graph()
        self._rebuild_optimizers()
        
    def _rebuild_optimizers(self):
        """Creates or updates the optimizers after a structural mutation."""
        # W parameters: A, B across all nodes + classifier
        w_params = [self.classifier.weight, self.classifier.bias]
        s_params = []
        for node in self.state.nodes.values():
            w_params.extend([node.A, node.B, node.bias])
            s_params.append(node.S)
            
        # Using AdamW for better regularization of the low-rank factors
        self.opt_W = torch.optim.AdamW(w_params, lr=self.cfg.eta_W, weight_decay=self.cfg.weight_decay)
        self.opt_S = torch.optim.AdamW(s_params, lr=self.cfg.eta_S, weight_decay=self.cfg.weight_decay)

    # ── API ───────────────────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        X: (B, input_dim) -> Returns logits (B, n_classes)
        """
        # Ensure correct shape and device
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.to(self.device)
        
        graph_out = self.forward_executor(X, t_budget=self.cfg.t_budget)
        logits = self.classifier(graph_out)
        return logits

    def train_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Takes a batch, runs forward/backward, and steps optimizers.
        If X, y are numpy, they are converted to tensors.
        """
        self.train()
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        
        self.opt_W.zero_grad()
        self.opt_S.zero_grad()
        
        logits = self.forward(X_t)
        loss = F.cross_entropy(logits, y_t)
        
        loss.backward()
        
        # Clip gradients to prevent explosion in dynamic architecture
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        
        self.opt_W.step()
        self.opt_S.step()
        
        loss_val = loss.item()
        self.ema_loss = 0.95 * self.ema_loss + 0.05 * loss_val
        self.step_count += 1
        
        # Structural Evolution (Every interval steps)
        if self.step_count % self.cfg.evolve_interval == 0:
            # We don't have exact per-edge tension anymore natively, 
            # but we can evolve based on the generic stagnant loss logic for now.
            # (In a full implementation, we could hook autograd to get edge flow).
            
            # Simple workaround: just assume a random input->hidden edge is max tension 
            # or default to auto-growth
            pruned = self.evolver.prune()
            if pruned or (abs(self.ema_loss - self.prev_ema_loss) < 0.005 and self.ema_loss > 0.1):
                self.evolver.ensure_nonlinearity()
                self.evolver.insert_capacity()
                self._rebuild_optimizers() # Crucial: params changed!
                
            self.prev_ema_loss = self.ema_loss

        return loss_val

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.forward(X_t)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.forward(X_t)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def get_stats(self) -> Dict[str, Any]:
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_nodes": len(self.state.nodes),
            "total_edges": sum(len(e) for e in self.state.edges_out.values()),
            "total_params": params,
            "ema_loss": self.ema_loss,
            "primitives": {
                p.name: sum(1 for n in self.state.nodes.values() if n.primitive == p)
                for p in Primitive
            }
        }
