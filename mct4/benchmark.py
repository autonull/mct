#!/usr/bin/env python3
"""
MCT4 Comprehensive Benchmark Suite

Benchmarks performance across multiple tasks and configurations.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import sys

sys.path.insert(0, '/home/me/mct')

from mct4 import MCT4, MCT4Config, Primitive, NodeType


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    task_name: str
    config_name: str
    final_accuracy: float
    final_loss: float
    training_time: float
    samples_per_second: float
    node_count_final: int
    node_count_peak: int
    pruning_events: int
    convergence_passes: int
    config: MCT4Config = field(repr=False)


def create_xor_data(n_samples: int, D: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create XOR dataset."""
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


def create_moons_data(n_samples: int, D: int, noise: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Create two moons dataset."""
    n_half = n_samples // 2
    
    theta = np.linspace(0, np.pi, n_half)
    x1 = np.cos(theta)
    y1 = np.sin(theta)
    x2 = np.cos(theta) + 1
    y2 = np.sin(theta) - 0.5
    
    X_2d = np.vstack([
        np.column_stack([x1, y1]) + np.random.randn(n_half, 2) * noise,
        np.column_stack([x2, y2]) + np.random.randn(n_half, 2) * noise,
    ])
    
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


def create_circular_data(n_samples: int, D: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create circular dataset (inner vs outer ring)."""
    n_half = n_samples // 2
    
    # Inner circle
    r1 = np.random.uniform(0, 1, n_half)
    theta1 = np.random.uniform(0, 2*np.pi, n_half)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    
    # Outer ring
    r2 = np.random.uniform(1.5, 2.5, n_half)
    theta2 = np.random.uniform(0, 2*np.pi, n_half)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    
    X_2d = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2]),
    ])
    
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


def create_sequence_data(n_samples: int, D: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequence prediction task (predict next element)."""
    X_seq = []
    Y_seq = []
    
    for _ in range(n_samples):
        # Random walk sequence
        seq = np.zeros((seq_len, D))
        seq[0] = np.random.randn(D) * 0.5
        
        for t in range(1, seq_len):
            seq[t] = seq[t-1] + np.random.randn(D) * 0.3
        
        X_seq.append(seq[0])
        Y_seq.append(seq[-1])
    
    return np.array(X_seq), np.array(Y_seq)


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy."""
    D = predictions.shape[1]
    half = D // 2
    
    pred_sums_first = np.sum(predictions[:, :half], axis=1)
    pred_sums_second = np.sum(predictions[:, half:], axis=1)
    pred_classes = (pred_sums_second > pred_sums_first).astype(float)
    
    true_sums_first = np.sum(targets[:, :half], axis=1)
    true_sums_second = np.sum(targets[:, half:], axis=1)
    true_classes = (true_sums_second > true_sums_first).astype(float)
    
    return np.mean(pred_classes == true_classes)


def benchmark_task(
    task_name: str,
    create_data_fn,
    n_train: int,
    n_test: int,
    config: MCT4Config,
    n_epochs: int,
    batch_size: int,
    D: int,
    evolve: bool = True,
    **data_kwargs
) -> BenchmarkResult:
    """Run benchmark on a specific task."""
    
    # Create data
    X_train, Y_train = create_data_fn(n_train, D, **data_kwargs)
    X_test, Y_test = create_data_fn(n_test, D, **data_kwargs)
    
    # Create model
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    initial_nodes = len(model.state.nodes)
    peak_nodes = initial_nodes
    
    # Training
    start_time = time.time()
    total_samples = 0
    
    losses = []
    accuracies = []
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_train)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]
        
        epoch_losses = []
        
        for i in range(0, n_train, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            loss = model.train_batch(X_batch, Y_batch, evolve=evolve)
            epoch_losses.append(loss)
            total_samples += len(X_batch)
            
            # Track peak nodes
            current_nodes = len(model.state.nodes)
            peak_nodes = max(peak_nodes, current_nodes)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # Evaluate
        predictions = np.array([model.predict(X_test[i]) for i in range(n_test)])
        acc = compute_accuracy(predictions, Y_test)
        accuracies.append(acc)
    
    training_time = time.time() - start_time
    
    # Final evaluation
    predictions = np.array([model.predict(X_test[i]) for i in range(n_test)])
    final_acc = compute_accuracy(predictions, Y_test)
    final_loss = losses[-1] if losses else 0
    
    stats = model.get_stats()
    
    return BenchmarkResult(
        task_name=task_name,
        config_name=f"D{config.D}_eta{config.eta}_K{config.K}",
        final_accuracy=final_acc,
        final_loss=final_loss,
        training_time=training_time,
        samples_per_second=total_samples / training_time,
        node_count_final=stats['total_nodes'],
        node_count_peak=peak_nodes,
        pruning_events=model.metrics.pruning_events,
        convergence_passes=stats['kappa'],
        config=config
    )


def run_benchmarks():
    """Run comprehensive benchmark suite."""
    print("=" * 80)
    print("MCT4 Comprehensive Benchmark Suite")
    print("=" * 80)
    
    results = []
    
    # Configuration variants
    configs = {
        'small': MCT4Config(D=32, t_budget=15, eta=0.01, alpha=0.02, beta=0.03, 
                           gamma=0.001, sigma_mut=0.05, K=2, kappa_thresh=50, N=16),
        'medium': MCT4Config(D=64, t_budget=20, eta=0.005, alpha=0.015, beta=0.04,
                            gamma=0.0005, sigma_mut=0.03, K=2, kappa_thresh=100, N=32),
        'large': MCT4Config(D=128, t_budget=25, eta=0.002, alpha=0.01, beta=0.05,
                           gamma=0.0003, sigma_mut=0.02, K=3, kappa_thresh=150, N=64),
        'fast_evolve': MCT4Config(D=64, t_budget=20, eta=0.01, alpha=0.02, beta=0.05,
                                  gamma=0.002, sigma_mut=0.08, K=4, kappa_thresh=30, N=32),
        'stable': MCT4Config(D=64, t_budget=20, eta=0.002, alpha=0.01, beta=0.03,
                             gamma=0.0002, sigma_mut=0.02, K=1, kappa_thresh=200, N=32),
    }
    
    # Tasks
    tasks = [
        ('XOR', create_xor_data, 500, 100, {}, 50),
        ('Moons', create_moons_data, 500, 100, {'noise': 0.2}, 50),
        ('Circular', create_circular_data, 500, 100, {}, 50),
        ('Sequence', create_sequence_data, 300, 50, {'seq_len': 5}, 30),
    ]
    
    for task_name, create_fn, n_train, n_test, data_kwargs, n_epochs in tasks:
        print(f"\n{'='*80}")
        print(f"Task: {task_name}")
        print(f"{'='*80}")
        
        for config_name, config in configs.items():
            D = config.D
            
            # Adjust epochs for sequence task
            actual_epochs = n_epochs
            if task_name == 'Sequence':
                actual_epochs = min(n_epochs, 30)
            
            print(f"\n  Config: {config_name} (D={D})...")
            
            try:
                result = benchmark_task(
                    task_name=task_name,
                    create_data_fn=create_fn,
                    n_train=n_train,
                    n_test=n_test,
                    config=config,
                    n_epochs=actual_epochs,
                    batch_size=config.N,
                    D=D,
                    evolve=True,
                    **data_kwargs
                )
                results.append(result)
                
                print(f"    Accuracy: {result.final_accuracy:.1%}")
                print(f"    Loss: {result.final_loss:.4f}")
                print(f"    Time: {result.training_time:.2f}s")
                print(f"    Samples/s: {result.samples_per_second:.1f}")
                print(f"    Nodes: {result.node_count_final} (peak: {result.node_count_peak})")
                print(f"    Pruning events: {result.pruning_events}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    # Group by task
    tasks_run = set(r.task_name for r in results)
    
    for task in tasks_run:
        task_results = [r for r in results if r.task_name == task]
        best = max(task_results, key=lambda r: r.final_accuracy)
        
        print(f"\n{task}:")
        print(f"  Best: {best.config_name} -> {best.final_accuracy:.1%} accuracy")
        print(f"  ({best.samples_per_second:.1f} samples/s, {best.node_count_final} nodes)")
    
    return results


def profile_single_run():
    """Profile a single run to identify bottlenecks."""
    print("\n" + "=" * 80)
    print("PROFILING RUN")
    print("=" * 80)
    
    import cProfile
    import pstats
    from io import StringIO
    
    config = MCT4Config(D=64, t_budget=20, eta=0.01, N=32)
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    X, Y = create_xor_data(200, config.D)
    
    def run_training():
        for i in range(20):
            model.train_batch(X[:32], Y[:32], evolve=True)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    run_training()
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print("\nTop 20 functions by cumulative time:")
    print("-" * 80)
    stats.print_stats(20)
    
    return profiler


if __name__ == "__main__":
    # Run benchmarks
    results = run_benchmarks()
    
    # Profile
    profile_single_run()
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)
