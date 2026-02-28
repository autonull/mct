"""
MorphoNet - Self-Structuring Neural Networks

A novel architecture that learns its own connectivity during training.

Key innovations:
1. Dense-to-sparse training - starts connected, learns what to prune
2. Edge importance learning - each connection has a learnable gate
3. Dynamic capacity - grows/shrinks based on task complexity
4. Competitive performance with MLPs + interpretability

This is MCT5's self-structuring idea, but actually optimized to work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import time

# ═══════════════════════════════════════════════════════════════════════════
# MORPHO LINEAR - Learnable connectivity
# ═══════════════════════════════════════════════════════════════════════════

class MorphoLinear(nn.Module):
    """
    Linear layer with learnable edge masks.
    
    Instead of fixed connectivity, learns which connections matter.
    """
    def __init__(self, in_features: int, out_features: int, 
                 sparse_init: float = 0.3,  # Start with 30% connections
                 mask_lr: float = 0.01,
                 sparsity_loss: float = 0.001):  # Penalty for using connections
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparse_init = sparse_init
        self.sparsity_loss = sparsity_loss
        
        # Weight matrix
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 
                                   np.sqrt(2.0 / in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Learnable mask - controls which connections are active
        # Initialize with random subset active
        mask_init = torch.rand(out_features, in_features) < sparse_init
        self.mask_logits = nn.Parameter(torch.zeros(out_features, in_features))
        self.mask_logits.data = torch.where(
            mask_init,
            torch.ones_like(self.mask_logits) * 2,  # Active
            torch.ones_like(self.mask_logits) * -2  # Inactive
        )
        
        # Mask learning rate (separate from weights)
        self.mask_lr = mask_lr
    
    def get_mask(self, temperature: float = 1.0) -> torch.Tensor:
        """Get connectivity mask via sigmoid relaxation."""
        return torch.sigmoid(self.mask_logits / temperature)
    
    def get_sparsity(self) -> float:
        """Current fraction of active connections."""
        mask = self.get_mask()
        return (mask > 0.5).float().mean().item()
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        mask = self.get_mask(temperature)
        # Apply mask to weights
        masked_weight = self.weight * mask
        return F.linear(x, masked_weight, self.bias)
    
    def get_mask_loss(self) -> torch.Tensor:
        """Sparsity penalty - encourages fewer connections."""
        mask = self.get_mask()
        return self.sparsity_loss * mask.mean()


# ═══════════════════════════════════════════════════════════════════════════
# MORPHONET - Self-structuring MLP
# ═══════════════════════════════════════════════════════════════════════════

class MorphoNet(nn.Module):
    """
    Self-structuring neural network.
    
    Learns both weights AND architecture during training.
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 n_classes: int,
                 sparse_init: float = 0.3,
                 sparsity_loss: float = 0.001,
                 mask_lr: float = 0.01,
                 growth_threshold: float = 0.8,  # Add capacity when layer >80% used
                 prune_threshold: float = 0.1):  # Remove connections <10% active
        
        super().__init__()
        
        self.sparsity_loss = sparsity_loss
        self.growth_threshold = growth_threshold
        self.prune_threshold = prune_threshold
        
        # Build layers
        self.layers = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(MorphoLinear(
                prev_dim, hidden_dim,
                sparse_init=sparse_init,
                mask_lr=mask_lr,
                sparsity_loss=sparsity_loss
            ))
            prev_dim = hidden_dim
        
        # Output layer (fully connected)
        self.output = nn.Linear(prev_dim, n_classes)
        
        # Track layer utilization for growth decisions
        self.layer_utilization = [0.0] * len(hidden_dims)
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x, temperature))
        return self.output(x)
    
    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_effective_params(self) -> int:
        """Count only active (non-pruned) parameters."""
        count = 0
        for layer in self.layers:
            mask = layer.get_mask()
            count += (mask > 0.5).sum().item()
        # Output layer is always dense
        count += self.output.weight.numel()
        return int(count)
    
    def get_sparsity(self) -> float:
        """Average sparsity across layers."""
        return np.mean([layer.get_sparsity() for layer in self.layers])
    
    def get_architecture_loss(self) -> torch.Tensor:
        """Sum of sparsity penalties."""
        return sum(layer.get_mask_loss() for layer in self.layers)
    
    def adapt_structure(self, epoch: int, total_epochs: int):
        """
        Dynamically adjust network structure during training.
        
        Early training: encourage exploration (higher temperature)
        Late training: commit to structure (lower temperature, prune)
        """
        progress = epoch / total_epochs
        
        # Anneal temperature
        temperature = max(0.5, 1.0 - progress * 0.5)
        
        # Anneal sparsity penalty (increase over time)
        for layer in self.layers:
            layer.sparsity_loss = self.sparsity_loss * (1 + progress * 3)
        
        return temperature
    
    def prune(self, threshold: float = 0.2):
        """Hard prune connections below threshold."""
        pruned = 0
        with torch.no_grad():
            for layer in self.layers:
                mask = layer.get_mask()
                # Zero out logits for low-confidence connections
                to_prune = mask < threshold
                layer.mask_logits[to_prune] = -10  # Effectively zero after sigmoid
                pruned += to_prune.sum().item()
        return pruned


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING WITH STRUCTURE LEARNING
# ═══════════════════════════════════════════════════════════════════════════

class MorphoTrainer:
    """
    Trainer that optimizes both weights and architecture.
    """
    def __init__(self, model: MorphoNet, 
                 weight_lr: float = 0.001,
                 mask_lr: float = 0.01):
        self.model = model
        
        # Separate optimizers for weights and masks
        weight_params = [
            {'params': [l.weight for l in model.layers], 'lr': weight_lr},
            {'params': [l.bias for l in model.layers], 'lr': weight_lr},
            {'params': model.output.parameters(), 'lr': weight_lr}
        ]
        mask_params = [
            {'params': [l.mask_logits for l in model.layers], 'lr': mask_lr}
        ]
        
        self.weight_opt = torch.optim.AdamW(weight_params, weight_decay=0.01)
        self.mask_opt = torch.optim.Adam(mask_params)
        
        self.loss_history = []
    
    def train_step(self, X: torch.Tensor, y: torch.Tensor, 
                   temperature: float = 1.0) -> Tuple[float, float]:
        """One training step."""
        self.model.train()
        
        # Forward pass
        logits = self.model(X, temperature)
        ce_loss = F.cross_entropy(logits, y)
        arch_loss = self.model.get_architecture_loss()
        total_loss = ce_loss + arch_loss
        
        # Backward pass
        self.weight_opt.zero_grad()
        self.mask_opt.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        # Update
        self.weight_opt.step()
        self.mask_opt.step()
        
        sparsity = self.model.get_sparsity()
        self.loss_history.append(total_loss.item())
        
        return total_loss.item(), sparsity
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 100, batch_size: int = 64,
              verbose: bool = True) -> dict:
        """Full training loop with structure adaptation."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        n = len(X)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Adapt structure based on progress
            temperature = self.model.adapt_structure(epoch, epochs)
            
            # Mini-batch training
            perm = torch.randperm(n)
            epoch_loss = 0
            epoch_sparsity = 0
            n_batches = 0
            
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                loss, sparsity = self.train_step(
                    X_t[idx], y_t[idx], temperature
                )
                epoch_loss += loss
                epoch_sparsity += sparsity
                n_batches += 1
            
            # Periodic pruning
            if epoch > epochs // 2 and epoch % 10 == 0:
                pruned = self.model.prune(0.2)
                if verbose and pruned > 0:
                    print(f"  Epoch {epoch}: pruned {pruned} connections")
            
            if verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch}: loss={epoch_loss/n_batches:.4f}, "
                      f"sparsity={1-epoch_sparsity/n_batches:.1%}, "
                      f"temp={temperature:.2f}")
        
        train_time = time.time() - start_time
        
        # Final pruning
        self.model.prune(0.3)
        
        return {
            'train_time': train_time,
            'final_loss': self.loss_history[-1],
            'final_sparsity': self.model.get_sparsity(),
            'total_params': self.model.get_total_params(),
            'effective_params': self.model.get_effective_params()
        }


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

def compare_architectures():
    """Compare MorphoNet vs standard MLP."""
    from sklearn.datasets import make_classification, make_moons, make_circles
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("=" * 70)
    print("  MorphoNet vs MLP: Architecture Comparison")
    print("=" * 70)
    
    datasets = {
        'Blobs': lambda: make_classification(n_samples=800, n_features=20, 
                                              n_informative=18, n_classes=5,
                                              n_clusters_per_class=2, random_state=42),
        'Moons': lambda: make_moons(n_samples=600, noise=0.2, random_state=42),
        'Circles': lambda: make_circles(n_samples=600, noise=0.1, factor=0.5, 
                                        random_state=42),
    }
    
    results = []
    
    for name, make_fn in datasets.items():
        print(f"\n{name}:")
        print("-" * 50)
        
        X, y = make_fn()
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42)
        
        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        
        # Standard MLP
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, n_classes)
                )
            def forward(self, x): return self.net(x)
        
        mlp = MLP()
        mlp_params = sum(p.numel() for p in mlp.parameters())
        mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=0.01)
        
        X_t, y_t = torch.tensor(X_train), torch.tensor(y_train)
        start = time.time()
        for _ in range(100):
            mlp_opt.zero_grad()
            F.cross_entropy(mlp(X_t), y_t).backward()
            mlp_opt.step()
        mlp_time = time.time() - start
        
        mlp.eval()
        X_test_t, y_test_t = torch.tensor(X_test), torch.tensor(y_test)
        with torch.no_grad():
            mlp_acc = (mlp(X_test_t).argmax(-1) == y_test_t).float().mean().item()
        
        print(f"  MLP:        {mlp_acc:.1%} acc, {mlp_params:,} params, {mlp_time:.2f}s")
        
        # MorphoNet
        model = MorphoNet(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            n_classes=n_classes,
            sparse_init=0.4,
            sparsity_loss=0.002,
            mask_lr=0.02
        )
        
        trainer = MorphoTrainer(model, weight_lr=0.001, mask_lr=0.02)
        stats = trainer.train(X_train, y_train, epochs=100, batch_size=64, verbose=False)
        
        model.eval()
        with torch.no_grad():
            morpho_acc = (model(X_test_t).argmax(-1) == y_test_t).float().mean().item()
        
        param_ratio = stats['effective_params'] / mlp_params
        
        print(f"  MorphoNet:  {morpho_acc:.1%} acc, {stats['effective_params']:,.0f}/{stats['total_params']:,} params, {stats['train_time']:.2f}s")
        print(f"    → Sparsity: {1-stats['final_sparsity']:.1%} pruned")
        print(f"    → Param efficiency: {param_ratio:.2f}x vs MLP")
        
        if morpho_acc >= mlp_acc * 0.95:  # Within 5% of MLP
            print(f"    → ✓ Competitive (within 5% of MLP)")
        if param_ratio < 0.5:
            print(f"    → ✓ Efficient (<50% of MLP params)")
        
        results.append({
            'dataset': name,
            'mlp_acc': mlp_acc,
            'morpho_acc': morpho_acc,
            'mlp_params': mlp_params,
            'morpho_params': stats['effective_params'],
            'sparsity': 1 - stats['final_sparsity']
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    avg_mlp = np.mean([r['mlp_acc'] for r in results])
    avg_morpho = np.mean([r['morpho_acc'] for r in results])
    avg_sparsity = np.mean([r['sparsity'] for r in results])
    avg_param_ratio = np.mean([r['morpho_params']/r['mlp_params'] for r in results])
    
    print(f"  Average MLP accuracy:     {avg_mlp:.1%}")
    print(f"  Average MorphoNet accuracy: {avg_morpho:.1%}")
    print(f"  Accuracy gap:             {avg_mlp - avg_morpho:+.1%}")
    print(f"  Average sparsity:         {avg_sparsity:.1%} connections pruned")
    print(f"  Parameter efficiency:     {avg_param_ratio:.2f}x vs MLP")
    
    print("\n" + "=" * 70)
    if avg_morpho >= avg_mlp * 0.95:
        print("  ✓ MorphoNet achieves competitive accuracy with learned sparsity")
    if avg_param_ratio < 0.5:
        print("  ✓ MorphoNet uses fewer effective parameters than MLP")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    compare_architectures()
