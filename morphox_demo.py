#!/usr/bin/env python3
"""MorphoX Demonstration"""
import torch
import torch.nn.functional as F
import numpy as np
from morphox import MorphoX, MorphoXTrainer, create_morphox
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print('=' * 80)
print('  MorphoX: Next-Generation Dynamic Architecture')
print('=' * 80)

X, y = make_classification(n_samples=500, n_features=20, n_informative=15, 
                           n_classes=5, n_clusters_per_class=2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_morphox({
    'input_dim': 20, 'hidden_dims': [64, 32], 'n_classes': 5,
    'use_dynamic_mask': True, 'use_learnable_primitive': True,
    'use_early_exit': True, 'exit_threshold': 0.5,
    'device': 'cpu'
})

print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

trainer = MorphoXTrainer(model, sparsity_weight=0.01, compute_weight=0.1)
stats = trainer.train(X_train, y_train, X_val=X_test, y_val=y_test, 
                      epochs=80, batch_size=32, verbose=True)

print('\n  Results:')
model.eval()
with torch.no_grad():
    logits, info = model(torch.tensor(X_test), return_all=True)
    acc = (logits.argmax(-1).numpy() == y_test).mean()
print(f'  Accuracy: {acc:.1%}')
print(f'  Sparsity: {info.get("total_sparsity", 0):.1%}')
print(f'  Early exits: {info.get("early_exits", 0)}/{len(X_test)}')

print('\n  Learned Primitives:')
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'primitive'):
        w = F.softmax(layer.primitive.primitive_logits / layer.primitive.temperature, dim=-1)
        dom = layer.primitive.primitives[w.argmax().item()]
        print(f'  Layer {i}: {dom}')

print('\n  MORPHOX: NOVEL ARCHITECTURE')
print('  - Input-dependent masks (dynamic computation)')
print('  - Learnable primitives (chooses activation per layer)')
print('  - Early exiting (adaptive depth)')
print('  - Compute budget awareness')
print('=' * 80)
