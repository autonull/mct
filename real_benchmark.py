#!/usr/bin/env python3
"""
MCT5 vs MLP: Real Benchmark Comparison
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mct5_unified import MCT5, MCT5Config, LearningMode

print("=" * 70)
print("  MCT5 vs MLP: Real Benchmark Comparison")
print("=" * 70)

def make_dataset(name, n_samples=1000, n_features=20, n_classes=5):
    if name == "blobs":
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, 
            n_informative=n_features-2, n_redundant=2,
            n_classes=n_classes, n_clusters_per_class=2,
            random_state=42
        )
    else:  # 10-class
        X, y = make_classification(
            n_samples=n_samples, n_features=30,
            n_informative=25, n_redundant=5,
            n_classes=n_classes, n_clusters_per_class=3,
            random_state=42
        )
    return X.astype(np.float32), y.astype(int)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

def train_mlp(X_train, y_train, X_test, y_test, input_dim, n_classes, 
              hidden_dims, epochs, batch_size=32, lr=0.01):
    model = MLP(input_dim, hidden_dims, n_classes)
    total_params = model.count_params()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    n_batches = max(1, len(X_train) // batch_size)
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(n_batches):
            idx = perm[i*batch_size:(i+1)*batch_size]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]
            
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    
    train_time = time.time() - t0
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=-1)
        acc = (preds == y_test_t).float().mean().item()
    
    return acc, train_time, total_params

def train_mct5(X_train, y_train, X_test, y_test, input_dim, n_classes,
               D, r, epochs, evolve_interval=8):
    config = MCT5Config(
        D=D, r=r, n_classes=n_classes, input_dim=input_dim,
        learning_mode=LearningMode.HYBRID,
        eta_W=0.02, eta_S=0.005,
        evolve_interval=evolve_interval,
        adaptive_mutation=True,
        ensure_nonlinearity=True,
        device='cpu', seed=42, verbose=False
    )
    model = MCT5(config)
    model.initialize()
    
    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    t0 = time.time()
    n = len(X_train)
    for epoch in range(epochs):
        lr_scale = 1.0 / (1.0 + 0.01 * epoch)
        model.cfg.eta_W = 0.02 * lr_scale
        for i in np.random.permutation(n):
            model.train_step(X_train[i], int(y_train[i]))
    
    train_time = time.time() - t0
    
    final_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acc = model.score(X_test, y_test)
    
    return acc, train_time, initial_params, final_params

# Run benchmarks
results = []

for dataset_name, n_samples, n_classes in [
    ("Blobs-5C", 800, 5),
    ("10-Class", 1000, 10),
]:
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name} ({n_samples} samples, {n_classes} classes)")
    print(f"{'='*70}")
    
    X, y = make_dataset(dataset_name.split('-')[0].lower(), n_samples, 
                        n_features=20 if n_classes == 5 else 30, n_classes=n_classes)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_dim = X_train.shape[1]
    
    # MLP configurations
    mlp_configs = [
        ("MLP-Small", [16]),
        ("MLP-Medium", [32, 16]),
        ("MLP-Large", [64, 32, 16]),
    ]
    
    print(f"\n  MLP Results (80 epochs):")
    print(f"  {'Model':<15} {'Params':>10} {'Time':>8} {'Acc':>10}")
    print(f"  {'-'*50}")
    
    mlp_results = []
    for name, hidden in mlp_configs:
        acc, ttime, params = train_mlp(
            X_train, y_train, X_test, y_test, 
            input_dim, n_classes, hidden, epochs=80, batch_size=32, lr=0.01
        )
        mlp_results.append((name, params, ttime, acc))
        print(f"  {name:<15} {params:>10,} {ttime:>7.2f}s {acc:>9.1%}")
    
    # MCT5 configurations
    mct5_configs = [
        ("MCT5-Small", 32, 8),
        ("MCT5-Medium", 48, 12),
        ("MCT5-Large", 64, 16),
    ]
    
    print(f"\n  MCT5 Results (80 epochs):")
    print(f"  {'Model':<15} {'Params':>10} {'Time':>8} {'Acc':>10}")
    print(f"  {'-'*50}")
    
    mct5_results = []
    for name, D, r in mct5_configs:
        acc, ttime, init_params, final_params = train_mct5(
            X_train, y_train, X_test, y_test,
            input_dim, n_classes, D, r, epochs=80
        )
        mct5_results.append((name, init_params, final_params, ttime, acc))
        print(f"  {name:<15} {init_params:>10,}->{final_params:<10,} {ttime:>7.2f}s {acc:>9.1%}")
    
    results.append((dataset_name, mlp_results, mct5_results))

# Summary
print(f"\n{'='*70}")
print("  SUMMARY: Best MCT5 vs Best MLP")
print(f"{'='*70}")

for dataset_name, mlp_res, mct5_res in results:
    best_mlp = max(mlp_res, key=lambda x: x[3])
    best_mct5 = max(mct5_res, key=lambda x: x[4])
    
    print(f"\n  {dataset_name}:")
    print(f"    Best MLP:  {best_mlp[0]:<12} Params={best_mlp[1]:,}  Time={best_mlp[2]:.1f}s  Acc={best_mlp[3]:.1%}")
    print(f"    Best MCT5: {best_mct5[0]:<12} Params={best_mct5[1]:,}->{best_mct5[2]:,}  Time={best_mct5[3]:.1f}s  Acc={best_mct5[4]:.1%}")
    
    delta = best_mct5[4] - best_mlp[3]
    if delta >= 0:
        print(f"    -> MCT5 leads by +{delta:.1%}")
    else:
        print(f"    -> MLP leads by {-delta:.1%}")

print(f"\n{'='*70}")
