"""
MCT5 Optimized - Fast batched implementation

Key optimizations:
1. True batched forward pass (no Python loops over samples)
2. Vectorized learning updates
3. Efficient low-rank weights (r << D)
4. Better initialization (Xavier/He)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

class Primitive(Enum):
    GELU = auto()
    RELU = auto()
    TANH = auto()
    SWISH = auto()
    QUADRATIC = auto()
    PRODUCT = auto()
    GATE = auto()
    ADD = auto()
    FORK = auto()

def apply_primitive(prim: Primitive, x: torch.Tensor) -> torch.Tensor:
    if prim == Primitive.GELU:
        return F.gelu(x)
    elif prim == Primitive.RELU:
        return F.relu(x)
    elif prim == Primitive.TANH:
        return torch.tanh(x)
    elif prim == Primitive.SWISH:
        return x * torch.sigmoid(x)
    elif prim == Primitive.QUADRATIC:
        return x * x + x
    elif prim == Primitive.FORK:
        return x
    elif prim == Primitive.PRODUCT:
        return x * x  # Self-product for single input
    return x


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZED NODE - Batched, Low-Rank
# ═══════════════════════════════════════════════════════════════════════════

class OptimizedNode(nn.Module):
    """
    Optimized node with:
    - True low-rank weights: W = A @ B.T where r << D
    - Batched operations
    - He initialization
    """
    def __init__(self, node_id: int, D: int, r: int, primitive: Primitive, 
                 device: str = "cpu"):
        super().__init__()
        self.id = node_id
        self.D = D
        self.r = r
        self.primitive = primitive
        
        # Low-rank weights with He init
        scale = np.sqrt(2.0 / (D + r))
        self.A = nn.Parameter(torch.randn(D, r, device=device) * scale)
        self.B = nn.Parameter(torch.randn(D, r, device=device) * scale)
        self.bias = nn.Parameter(torch.zeros(D, device=device))
        
        # Routing signature
        self.S = nn.Parameter(torch.randn(D, device=device))
        self.S.data = self.S.data / self.S.data.norm()
        
        # Health
        self.register_buffer("rho", torch.tensor(1.5, device=device))
    
    def forward(self, V_in: torch.Tensor) -> torch.Tensor:
        """
        V_in: (B, D) batched input
        Returns: (B, D) batched output
        """
        # Low-rank matmul: (B, D) @ (D, r) @ (r, D) = (B, D)
        # More efficient: ((V_in @ B) @ A.T)
        V_weighted = (V_in @ self.B) @ self.A.T + self.bias
        return apply_primitive(self.primitive, V_weighted)


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZED GRAPH - Depth-layer batched execution
# ═══════════════════════════════════════════════════════════════════════════

class OptimizedMCT5(nn.Module):
    """
    Optimized MCT5 with:
    - Batched forward pass
    - Efficient depth-layer execution
    - Minimal Python overhead
    """
    
    def __init__(self, D: int = 64, r: int = 16, n_classes: int = 2, 
                 input_dim: int = 2, device: str = "cpu"):
        super().__init__()
        self.D = D
        self.r = r
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.device = device
        
        # Create nodes
        self.nodes = nn.ModuleDict()
        self.edges_out: Dict[int, List[int]] = {}
        self.edges_in: Dict[int, List[int]] = {}
        
        # Build initial graph
        self._build_graph()
        
        # Output classifier
        self.classifier = nn.Linear(D, n_classes, device=device)
        
        # Optimizer
        self.optimizer = None
    
    def _build_graph(self):
        """Build efficient starter graph."""
        # Input node
        inp = OptimizedNode(0, self.D, self.r, Primitive.FORK, self.device)
        self.nodes["0"] = inp
        self.edges_out[0] = []
        self.edges_in[0] = []
        
        # Hidden nodes with diverse primitives
        primitives = [Primitive.GELU, Primitive.QUADRATIC, Primitive.GATE]
        hidden_ids = []
        
        for i, prim in enumerate(primitives):
            nid = i + 1
            node = OptimizedNode(nid, self.D, self.r, prim, self.device)
            self.nodes[str(nid)] = node
            self.edges_out[nid] = []
            self.edges_in[nid] = []
            hidden_ids.append(nid)
        
        # Aggregator
        agg_id = len(primitives) + 1
        agg = OptimizedNode(agg_id, self.D, self.r, Primitive.ADD, self.device)
        self.nodes[str(agg_id)] = agg
        self.edges_out[agg_id] = []
        self.edges_in[agg_id] = []
        
        # Output
        out_id = agg_id + 1
        out = OptimizedNode(out_id, self.D, self.r, Primitive.FORK, self.device)
        self.nodes[str(out_id)] = out
        self.edges_out[out_id] = []
        self.edges_in[out_id] = []
        
        # Wire: Input -> Hidden
        for hid in hidden_ids:
            self.edges_out[0].append(hid)
            self.edges_in[hid].append(0)
        
        # Wire: Hidden -> Agg
        for hid in hidden_ids:
            self.edges_out[hid].append(agg_id)
            self.edges_in[agg_id].append(hid)
        
        # Wire: Input -> Agg (skip)
        self.edges_out[0].append(agg_id)
        self.edges_in[agg_id].append(0)
        
        # Wire: Agg -> Output
        self.edges_out[agg_id].append(out_id)
        self.edges_in[out_id].append(agg_id)
        
        # Wire: Input -> Output (skip)
        self.edges_out[0].append(out_id)
        self.edges_in[out_id].append(0)
    
    def _get_topo_layers(self) -> List[List[int]]:
        """Get nodes grouped by topological depth."""
        # Simple BFS for this fixed graph
        in_degree = {nid: len(self.edges_in.get(nid, [])) for nid in self.nodes.keys()}
        in_degree = {int(k): v for k, v in in_degree.items()}
        
        layers = []
        queue = sorted([nid for nid, deg in in_degree.items() if deg == 0])
        
        while queue:
            layers.append(queue.copy())
            next_queue = []
            for nid in queue:
                for child in self.edges_out.get(nid, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            queue = sorted(next_queue)
        
        return layers
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass.
        X: (B, input_dim)
        Returns: (B, n_classes) logits
        """
        B = X.size(0)
        
        # Pad input to D
        if X.size(1) < self.D:
            padding = torch.zeros(B, self.D - X.size(1), device=self.device)
            X = torch.cat([X, padding], dim=1)
        
        # Node outputs buffer
        node_outputs: Dict[int, torch.Tensor] = {}
        
        # Execute by layers
        layers = self._get_topo_layers()
        
        for layer in layers:
            for nid in layer:
                node = self.nodes[str(nid)]
                
                if nid == 0:  # Input node
                    node_outputs[0] = X
                else:
                    # Aggregate inputs from parents
                    parents = self.edges_in.get(nid, [])
                    if not parents:
                        continue
                    
                    # Stack and mean (simple aggregation)
                    parent_outputs = torch.stack([node_outputs[p] for p in parents], dim=0)
                    V_in = parent_outputs.mean(dim=0)
                    
                    # Node forward
                    node_outputs[nid] = node(V_in)
        
        # Get output node
        out_id = max(self.nodes.keys())
        graph_out = node_outputs.get(int(out_id), torch.zeros(B, self.D, device=self.device))
        
        return self.classifier(graph_out)
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def train_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Train on a batch with proper vectorization.
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.01, weight_decay=0.01)
        
        self.train()
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        
        self.optimizer.zero_grad()
        logits = self.forward(X_t)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.forward(X_t)
            preds = logits.argmax(dim=-1)
        return preds.cpu().numpy()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("=" * 60)
    print("  Optimized MCT5 vs MLP Comparison")
    print("=" * 60)
    
    # Dataset
    X, y = make_classification(n_samples=500, n_features=15, n_informative=12,
                               n_classes=5, n_clusters_per_class=2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_dim, n_classes = X_train.shape[1], 5
    
    # MLP baseline
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, n_classes)
            )
        def forward(self, x): return self.net(x)
    
    mlp = MLP()
    mlp_params = sum(p.numel() for p in mlp.parameters())
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    
    t0 = time.time()
    for epoch in range(50):
        mlp_opt.zero_grad()
        loss = F.cross_entropy(mlp(X_t), y_t)
        loss.backward()
        mlp_opt.step()
    mlp_time = time.time() - t0
    
    mlp.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    with torch.no_grad():
        mlp_acc = (mlp(X_test_t).argmax(-1) == y_test_t).float().mean().item()
    
    print(f"\nMLP (32->16): {mlp_params:,} params, {mlp_time:.2f}s, Acc={mlp_acc:.1%}")
    
    # Optimized MCT5
    model = OptimizedMCT5(D=48, r=8, n_classes=n_classes, input_dim=input_dim)
    mct5_params = model.count_params()
    
    t0 = time.time()
    for epoch in range(50):
        model.train_batch(X_train, y_train)
    mct5_time = time.time() - t0
    
    mct5_acc = model.score(X_test, y_test)
    
    print(f"OptMCT5 (D=48,r=8): {mct5_params:,} params, {mct5_time:.2f}s, Acc={mct5_acc:.1%}")
    
    print(f"\nDelta: {mct5_acc - mlp_acc:+.1%} accuracy, {mlp_time/mct5_time:.1f}x speed")
    print(f"Params ratio: {mct5_params/mlp_params:.1f}x")
    
    print("\n" + "=" * 60)
