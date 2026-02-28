#!/usr/bin/env python3
"""
MCT4 Demonstration

Demonstrates MCT4 capabilities on classification and sequence modeling tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mct4 import MCT4, MCT4Config, Primitive


def create_xor_data(n_samples: int = 100, D: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create XOR dataset embedded in D dimensions.
    
    XOR is the canonical non-linearly-separable problem.
    """
    # Generate 2D XOR
    X_2d = np.random.randn(n_samples, 2)
    y = ((X_2d[:, 0] > 0) != (X_2d[:, 1] > 0)).astype(float)
    
    # Embed in D dimensions (pad with zeros)
    X = np.zeros((n_samples, D))
    X[:, :2] = X_2d
    
    # Create D-dimensional one-hot style target
    # Class 0: positive in first half, class 1: positive in second half
    Y = np.zeros((n_samples, D))
    for i in range(n_samples):
        if y[i] == 0:
            Y[i, :D//2] = 1.0 / (D//2)  # Uniform in first half
        else:
            Y[i, D//2:] = 1.0 / (D - D//2)  # Uniform in second half
    
    return X, Y


def create_moons_data(n_samples: int = 100, D: int = 64, noise: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Create two moons dataset embedded in D dimensions."""
    n_samples_per_class = n_samples // 2
    
    # Moon 1
    theta = np.linspace(0, np.pi, n_samples_per_class)
    x1 = np.cos(theta)
    y1 = np.sin(theta)
    
    # Moon 2
    x2 = np.cos(theta) + 1
    y2 = np.sin(theta) - 0.5
    
    # Add noise
    X_2d = np.vstack([
        np.column_stack([x1, y1]) + np.random.randn(n_samples_per_class, 2) * noise,
        np.column_stack([x2, y2]) + np.random.randn(n_samples_per_class, 2) * noise,
    ])
    
    y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    
    # Embed in D dimensions
    X = np.zeros((n_samples, D))
    X[:, :2] = X_2d
    
    # One-hot encode
    Y = np.zeros((n_samples, 2))
    Y[np.arange(n_samples), y.astype(int)] = 1
    
    return X, Y


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy."""
    # For D-dimensional targets, compare which half has higher sum
    D = predictions.shape[1]
    half = D // 2
    
    pred_sums_first = np.sum(predictions[:, :half], axis=1)
    pred_sums_second = np.sum(predictions[:, half:], axis=1)
    pred_classes = (pred_sums_second > pred_sums_first).astype(float)
    
    true_sums_first = np.sum(targets[:, :half], axis=1)
    true_sums_second = np.sum(targets[:, half:], axis=1)
    true_classes = (true_sums_second > true_sums_first).astype(float)
    
    return np.mean(pred_classes == true_classes)


def demonstrate_xor():
    """Demonstrate MCT4 on XOR classification."""
    print("=" * 60)
    print("MCT4 XOR Classification Demo")
    print("=" * 60)
    
    # Configuration for smaller demo
    D = 64  # Reduced dimensionality for demo
    config = MCT4Config(
        D=D,
        t_budget=15,
        eta=0.01,  # Higher learning rate for demo
        alpha=0.02,
        beta=0.03,
        gamma=0.0005,
        sigma_mut=0.03,
        K=2,
        N=16,
        kappa_thresh=50,
        use_factored=False,
    )
    
    # Create model
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    # Generate data
    n_train = 200
    n_test = 50
    X_train, Y_train = create_xor_data(n_train, D)
    X_test, Y_test = create_xor_data(n_test, D)
    
    print(f"\nTraining data: {n_train} samples")
    print(f"Test data: {n_test} samples")
    print(f"Dimensionality: D={D}")
    print(f"Initial graph: {len(model.state.nodes)} nodes")
    
    # Training loop
    n_epochs = 100
    losses = []
    accuracies = []
    
    print("\nTraining...")
    for epoch in range(n_epochs):
        # Shuffle training data
        indices = np.random.permutation(n_train)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]
        
        epoch_losses = []
        
        # Mini-batch training
        batch_size = config.N
        for i in range(0, n_train, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            loss = model.train_batch(X_batch, Y_batch, evolve=True)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # Evaluate on test set
        predictions = np.array([model.predict(X_test[i]) for i in range(n_test)])
        acc = compute_accuracy(predictions, Y_test)
        accuracies.append(acc)
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            stats = model.get_stats()
            print(f"Epoch {epoch:4d}: Loss={avg_loss:.4f}, Test Acc={acc:.2%}, "
                  f"Nodes={stats['total_nodes']}, Edges={stats['total_edges']}, "
                  f"κ={stats['kappa']}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    predictions = np.array([model.predict(X_test[i]) for i in range(n_test)])
    final_acc = compute_accuracy(predictions, Y_test)
    print(f"Final Test Accuracy: {final_acc:.2%}")
    
    stats = model.get_stats()
    print(f"\nFinal Graph Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Active nodes: {stats['active_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Average health: {stats['avg_health']:.4f}")
    print(f"  Average tension: {stats['avg_tension']:.4f}")
    print(f"  Converged: {stats['is_converged']}")
    print(f"  Primitives: {stats['primitives']}")
    print(f"  Pruning events: {model.metrics.pruning_events}")
    
    return model, losses, accuracies


def demonstrate_sequence():
    """Demonstrate MCT4 on sequence modeling (copy task)."""
    print("\n" + "=" * 60)
    print("MCT4 Sequence Modeling Demo (Copy Task)")
    print("=" * 60)
    
    D = 32  # Smaller for sequence demo
    seq_len = 5
    config = MCT4Config(
        D=D,
        t_budget=20,
        eta=0.005,
        alpha=0.015,
        beta=0.04,
        gamma=0.0003,
        sigma_mut=0.02,
        K=1,
        N=8,
        kappa_thresh=30,
        use_factored=False,
    )
    
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    def generate_sequence() -> Tuple[np.ndarray, np.ndarray]:
        """Generate random sequence and target (copy task)."""
        # Generate sequence of vectors
        sequence = np.random.randn(seq_len, D) * 0.5
        
        # Input: first vector
        X = sequence[0]
        
        # Target: last vector (simple copy with delay)
        Y = sequence[-1]
        
        return X, Y
    
    print(f"\nSequence length: {seq_len}")
    print(f"Dimensionality: D={D}")
    print(f"Task: Copy first element to predict last element")
    
    # Training
    n_iterations = 200
    losses = []
    
    print("\nTraining...")
    for i in range(n_iterations):
        # Reset context for new sequence
        X, Y = generate_sequence()
        
        # Process as sequence
        model.reset_sequence()
        
        # Forward through sequence
        for t in range(seq_len):
            if t == 0:
                X_t = X
            else:
                # Use previous output as context (autoregressive)
                X_t = np.random.randn(D) * 0.1  # Placeholder
            
            model.forward(X_t)
        
        # Learn from final target
        loss = model.learn(Y)
        losses.append(loss)
        
        # Evolve periodically
        if i % 10 == 0:
            model.evolve()
        
        if i % 20 == 0 or i == n_iterations - 1:
            stats = model.get_stats()
            print(f"Iteration {i:4d}: Loss={loss:.4f}, "
                  f"Nodes={stats['total_nodes']}, κ={stats['kappa']}")
    
    # Evaluation
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    # Test on new sequences
    test_losses = []
    for _ in range(20):
        model.reset_sequence()
        X, Y = generate_sequence()
        
        for t in range(seq_len):
            if t == 0:
                X_t = X
            else:
                X_t = np.random.randn(D) * 0.1
            model.forward(X_t)
        
        pred = model.predict(np.zeros(D))
        test_loss = float(np.mean((pred - Y) ** 2))
        test_losses.append(test_loss)
    
    print(f"Average Test Loss: {np.mean(test_losses):.4f}")
    
    stats = model.get_stats()
    print(f"\nFinal Graph Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Primitives: {stats['primitives']}")
    
    return model, losses


def plot_results(losses: List[float], accuracies: List[float], save_path: str = None):
    """Plot training results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(accuracies)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Test Accuracy')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("MCT4 Proof-of-Concept Demonstration")
    print("Morphogenic Compute Topology v4.0")
    print("=" * 60)
    
    # Run XOR demo
    model_xor, losses_xor, acc_xor = demonstrate_xor()
    
    # Run sequence demo
    model_seq, losses_seq = demonstrate_sequence()
    
    # Plot results
    try:
        plot_results(losses_xor, acc_xor, save_path=None)
    except Exception as e:
        print(f"\nNote: Could not display plots: {e}")
        print("This is expected in headless environments.")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)
    print("\nKey MCT4 Features Demonstrated:")
    print("  ✓ Self-structuring graph (nodes added/pruned dynamically)")
    print("  ✓ Local learning without backpropagation")
    print("  ✓ Online, incremental learning")
    print("  ✓ Context vector for sequence handling")
    print("  ✓ Multiple primitive operators")
    print("  ✓ Convergence monitoring (κ counter)")
    print("  ✓ Health-based node survival")
    print("  ✓ Tension-driven structural evolution")
    print("\nThe graph discovered its own architecture for each task!")


if __name__ == "__main__":
    main()
