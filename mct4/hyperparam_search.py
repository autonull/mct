#!/usr/bin/env python3
"""
MCT4 Hyperparameter Optimization

Systematic search for optimal configurations.
"""

import numpy as np
import time
from itertools import product
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, '/home/me/mct')

from mct4 import MCT4, MCT4Config, Primitive


def create_xor_data(n_samples, D):
    X_2d = np.random.randn(n_samples, 2)
    y = ((X_2d[:, 0] > 0) != (X_2d[:, 1] > 0)).astype(float)
    X = np.zeros((n_samples, D))
    X[:, :2] = X_2d
    Y = np.zeros((n_samples, D))
    for i in range(n_samples):
        if y[i] == 0:
            Y[i, :D//2] = 1.0 / (D//2)
        else:
            Y[i, D//2:] = 1.0 / (D - D//2)
    return X, Y


def create_circular_data(n_samples, D):
    n_half = n_samples // 2
    r1 = np.random.uniform(0, 1, n_half)
    theta1 = np.random.uniform(0, 2*np.pi, n_half)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    r2 = np.random.uniform(1.5, 2.5, n_half)
    theta2 = np.random.uniform(0, 2*np.pi, n_half)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    X_2d = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_half), np.ones(n_half)])
    X = np.zeros((n_samples, D))
    X[:, :2] = X_2d
    Y = np.zeros((n_samples, D))
    for i in range(n_samples):
        if y[i] == 0:
            Y[i, :D//2] = 1.0 / (D//2)
        else:
            Y[i, D//2:] = 1.0 / (D - D//2)
    return X, Y


def compute_accuracy(predictions, targets):
    D = predictions.shape[1]
    half = D // 2
    pred_classes = (np.sum(predictions[:, half:], axis=1) > np.sum(predictions[:, :half], axis=1)).astype(float)
    true_classes = (np.sum(targets[:, half:], axis=1) > np.sum(targets[:, :half], axis=1)).astype(float)
    return np.mean(pred_classes == true_classes)


def evaluate_config(config: MCT4Config, task: str, n_epochs: int = 30) -> Dict:
    """Evaluate a configuration on a task."""
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    D = config.D
    if task == 'xor':
        X_train, Y_train = create_xor_data(300, D)
        X_test, Y_test = create_xor_data(100, D)
    else:  # circular
        X_train, Y_train = create_circular_data(300, D)
        X_test, Y_test = create_circular_data(100, D)
    
    start = time.time()
    best_acc = 0
    final_loss = 0
    
    for epoch in range(n_epochs):
        loss = model.train_batch(X_train, Y_train, evolve=True)
        final_loss = loss
        
        if epoch % 5 == 0:
            preds = np.array([model.predict(X_test[i]) for i in range(len(X_test))])
            acc = compute_accuracy(preds, Y_test)
            best_acc = max(best_acc, acc)
    
    elapsed = time.time() - start
    stats = model.get_stats()
    
    return {
        'config': config,
        'best_accuracy': best_acc,
        'final_loss': final_loss,
        'time': elapsed,
        'samples_per_sec': n_epochs * len(X_train) / elapsed,
        'final_nodes': stats['total_nodes'],
        'pruning_events': model.metrics.pruning_events,
        'kappa': stats['kappa'],
    }


def grid_search():
    """Run grid search over hyperparameters."""
    print("=" * 70)
    print("MCT4 Hyperparameter Optimization")
    print("=" * 70)
    
    # Parameter ranges
    params = {
        'eta': [0.005, 0.01, 0.02],
        'alpha': [0.01, 0.02, 0.03],
        'beta': [0.03, 0.05, 0.08],
        'gamma': [0.0005, 0.001, 0.002],
        'sigma_mut': [0.03, 0.05, 0.08],
        'K': [1, 2, 3],
    }
    
    D = 64
    task = 'xor'
    n_epochs = 30
    
    results = []
    best_result = None
    best_acc = 0
    
    # Test configurations
    configs_to_test = [
        # Baseline
        MCT4Config(D=D, eta=0.01, alpha=0.02, beta=0.05, gamma=0.001, sigma_mut=0.05, K=2, N=16, kappa_thresh=50),
        # High learning rate
        MCT4Config(D=D, eta=0.02, alpha=0.03, beta=0.05, gamma=0.001, sigma_mut=0.05, K=2, N=16, kappa_thresh=50),
        # Low learning rate
        MCT4Config(D=D, eta=0.005, alpha=0.01, beta=0.03, gamma=0.0005, sigma_mut=0.03, K=1, N=16, kappa_thresh=100),
        # Aggressive evolution
        MCT4Config(D=D, eta=0.01, alpha=0.02, beta=0.08, gamma=0.002, sigma_mut=0.08, K=3, N=16, kappa_thresh=30),
        # Stable learning
        MCT4Config(D=D, eta=0.005, alpha=0.01, beta=0.03, gamma=0.0003, sigma_mut=0.02, K=1, N=32, kappa_thresh=150),
        # Fast training
        MCT4Config(D=D, eta=0.02, alpha=0.03, beta=0.05, gamma=0.001, sigma_mut=0.05, K=2, N=32, kappa_thresh=30),
    ]
    
    config_names = [
        'Baseline',
        'High LR',
        'Low LR',
        'Aggressive Evo',
        'Stable',
        'Fast Training',
    ]
    
    print(f"\nTask: {task.upper()} (D={D})")
    print(f"Epochs: {n_epochs}")
    print("-" * 70)
    
    for name, config in zip(config_names, configs_to_test):
        print(f"\nEvaluating: {name}...")
        result = evaluate_config(config, task, n_epochs)
        result['name'] = name
        results.append(result)
        
        print(f"  Best Accuracy: {result['best_accuracy']:.1%}")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  Speed: {result['samples_per_sec']:.1f} samples/s")
        print(f"  Final Nodes: {result['final_nodes']} (pruning: {result['pruning_events']})")
        print(f"  κ={result['kappa']}")
        
        if result['best_accuracy'] > best_acc:
            best_acc = result['best_accuracy']
            best_result = result
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda r: -r['best_accuracy'])
    
    print("\nTop 3 Configurations:")
    for i, r in enumerate(results_sorted[:3]):
        print(f"\n{i+1}. {r['name']}: {r['best_accuracy']:.1%} accuracy")
        print(f"   η={r['config'].eta}, α={r['config'].alpha}, β={r['config'].beta}")
        print(f"   γ={r['config'].gamma}, σ={r['config'].sigma_mut}, K={r['config'].K}")
        print(f"   {r['samples_per_sec']:.1f} samples/s, {r['final_nodes']} nodes")
    
    print(f"\nBest configuration: {best_result['name']}")
    print(f"  Accuracy: {best_result['best_accuracy']:.1%}")
    print(f"  Speed: {best_result['samples_per_sec']:.1f} samples/s")
    
    return results


def demonstrate_capabilities():
    """Demonstrate MCT4 capabilities with optimal config."""
    print("\n" + "=" * 70)
    print("MCT4 Capabilities Demonstration")
    print("=" * 70)
    
    # Use best config from search
    config = MCT4Config(
        D=64, 
        eta=0.01, 
        alpha=0.02, 
        beta=0.05, 
        gamma=0.001, 
        sigma_mut=0.05, 
        K=2, 
        N=16,
        kappa_thresh=50,
        t_budget=20,
    )
    
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    print(f"\nConfiguration: D={config.D}, η={config.eta}, K={config.K}")
    print(f"Initial graph: {len(model.state.nodes)} nodes")
    
    # Train on XOR
    X_train, Y_train = create_xor_data(500, config.D)
    X_test, Y_test = create_xor_data(100, config.D)
    
    print("\nTraining on XOR...")
    accuracies = []
    
    for epoch in range(50):
        loss = model.train_batch(X_train, Y_train, evolve=True)
        
        if epoch % 10 == 0:
            preds = np.array([model.predict(X_test[i]) for i in range(len(X_test))])
            acc = compute_accuracy(preds, Y_test)
            accuracies.append(acc)
            stats = model.get_stats()
            print(f"  Epoch {epoch:2d}: acc={acc:.1%}, nodes={stats['total_nodes']}, edges={stats['total_edges']}, κ={stats['kappa']}")
    
    # Final evaluation
    preds = np.array([model.predict(X_test[i]) for i in range(len(X_test))])
    final_acc = compute_accuracy(preds, Y_test)
    
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {final_acc:.1%}")
    print(f"  Graph: {len(model.state.nodes)} nodes, {sum(len(n.edges_out) for n in model.state.nodes.values())} edges")
    
    stats = model.get_stats()
    print(f"  Primitives: {stats['primitives']}")
    print(f"  Avg health: {stats['avg_health']:.3f}")
    print(f"  Pruning events: {model.metrics.pruning_events}")
    
    return model


if __name__ == "__main__":
    # Run hyperparameter search
    results = grid_search()
    
    # Demonstrate with best config
    model = demonstrate_capabilities()
    
    print("\n" + "=" * 70)
    print("Optimization complete!")
    print("=" * 70)
