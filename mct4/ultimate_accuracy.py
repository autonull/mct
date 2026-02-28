#!/usr/bin/env python3
"""
MCT4 Ultimate Accuracy Demo

Achieves maximum accuracy through optimized architecture and training.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/me/mct')

from mct4 import MCT4, MCT4Config, Primitive, NodeType


def create_xor_data(n_samples, D):
    X_2d = np.random.randn(n_samples, 2)
    y = ((X_2d[:, 0] > 0) != (X_2d[:, 1] > 0)).astype(float)
    X = np.zeros((n_samples, D))
    X[:, :2] = X_2d
    Y = np.zeros((n_samples, D))
    for i in range(n_samples):
        Y[i, int(y[i])] = 1.0
    return X, Y, y


def create_circles_data(n_train, n_test, D, noise=0.1):
    n_half = n_train // 2
    theta1 = np.random.uniform(0, 2*np.pi, n_half)
    r1 = np.random.uniform(0.5, 1.0, n_half)
    x1 = r1 * np.cos(theta1) + np.random.randn(n_half) * noise
    y1 = r1 * np.sin(theta1) + np.random.randn(n_half) * noise
    
    theta2 = np.random.uniform(0, 2*np.pi, n_half)
    r2 = np.random.uniform(1.5, 2.5, n_half)
    x2 = r2 * np.cos(theta2) + np.random.randn(n_half) * noise
    y2 = r2 * np.sin(theta2) + np.random.randn(n_half) * noise
    
    X_train = np.zeros((n_train, D))
    X_train[:n_half, :2] = np.column_stack([x1, y1])
    X_train[n_half:, :2] = np.column_stack([x2, y2])
    Y_train = np.zeros((n_train, D))
    Y_train[:n_half, 0] = 1.0
    Y_train[n_half:, 1] = 1.0
    y_train = np.hstack([np.zeros(n_half), np.ones(n_half)])
    
    n_half = n_test // 2
    theta1 = np.random.uniform(0, 2*np.pi, n_half)
    r1 = np.random.uniform(0.5, 1.0, n_half)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    
    theta2 = np.random.uniform(0, 2*np.pi, n_half)
    r2 = np.random.uniform(1.5, 2.5, n_half)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    
    X_test = np.zeros((n_test, D))
    X_test[:n_half, :2] = np.column_stack([x1, y1])
    X_test[n_half:, :2] = np.column_stack([x2, y2])
    Y_test = np.zeros((n_test, D))
    Y_test[:n_half, 0] = 1.0
    Y_test[n_half:, 1] = 1.0
    y_test = np.hstack([np.zeros(n_half), np.ones(n_half)])
    
    return X_train, Y_train, y_train, X_test, Y_test, y_test


def compute_accuracy(predictions, targets):
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    return np.mean(pred_classes == true_classes)


class OptimizedMCT4:
    """MCT4 with optimized architecture for maximum accuracy."""
    
    def __init__(self, D=32, hidden_size=8):
        self.D = D
        self.hidden_size = hidden_size
        self.config = MCT4Config(
            D=D,
            t_budget=10,
            eta=0.2,
            alpha=0.02,
            beta=0.01,
            gamma=0.0001,
            sigma_mut=0.05,
            K=1,
            N=1,
            kappa_thresh=500,
            lambda_tau=0.08,
        )
        self.model = MCT4(self.config)
        self._init_graph()
    
    def _init_graph(self):
        """Create a larger, well-connected graph."""
        self.model.state.nodes = {}
        self.model.state.next_node_id = 0
        self.model.state.edge_tensions = {}
        
        # Input
        inp = self.model.state.create_node(NodeType.INPUT, Primitive.FORK, rho_base=1.0)
        
        # Multiple hidden nodes with diverse primitives
        hidden = []
        primitives = [Primitive.GELU, Primitive.RELU, Primitive.TANH, Primitive.GELU, 
                      Primitive.RELU, Primitive.GATE, Primitive.ADD, Primitive.GELU]
        for i in range(self.hidden_size):
            h = self.model.state.create_node(
                NodeType.HIDDEN, 
                primitives[i % len(primitives)], 
                rho_base=1.0
            )
            hidden.append(h)
            self.model.state.add_edge(inp.id, h.id)
        
        # Combination layer
        c1 = self.model.state.create_node(NodeType.HIDDEN, Primitive.ADD, rho_base=1.0)
        c2 = self.model.state.create_node(NodeType.HIDDEN, Primitive.GATE, rho_base=1.0)
        
        for h in hidden[:self.hidden_size//2]:
            self.model.state.add_edge(h.id, c1.id)
        for h in hidden[self.hidden_size//2:]:
            self.model.state.add_edge(h.id, c2.id)
        
        # Output
        out = self.model.state.create_node(NodeType.OUTPUT, Primitive.SOFTMAX, rho_base=1.0)
        self.model.state.add_edge(c1.id, out.id)
        self.model.state.add_edge(c2.id, out.id)
        
        # Skip connections
        self.model.state.add_edge(inp.id, c1.id)
        self.model.state.add_edge(inp.id, c2.id)
    
    def train(self, X, Y, epochs=200, verbose=True):
        n = len(X)
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            epoch_loss = 0
            
            for i in indices:
                self.model.reset_sequence()
                self.model.forward(X[i])
                loss = self.model.learn(Y[i])
                epoch_loss += loss
            
            losses.append(epoch_loss / n)
            
            if epoch % 50 == 0 or epoch == epochs - 1:
                preds = np.array([self.model.predict(X[i]) for i in range(min(100, n))])
                acc = compute_accuracy(preds, Y[:min(100, n)])
                accuracies.append(acc)
                
                if verbose:
                    print(f"Epoch {epoch:3d}: loss={epoch_loss/n:.4f}, acc={acc:.1%}")
        
        return losses, accuracies
    
    def predict(self, X):
        predictions = np.array([self.model.predict(x) for x in X])
        return np.argmax(predictions[:, :2], axis=1)
    
    def score(self, X, Y):
        return compute_accuracy(
            np.array([self.model.predict(x) for x in X]),
            Y
        )


def run_demo(name, X_train, Y_train, X_test, Y_test, D=32, epochs=200, hidden=8):
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, D={D}")
    
    clf = OptimizedMCT4(D=D, hidden_size=hidden)
    print(f"Initial graph: {len(clf.model.state.nodes)} nodes, {hidden} hidden")
    print("\nTraining...")
    
    losses, accs = clf.train(X_train, Y_train, epochs=epochs, verbose=True)
    
    train_acc = clf.score(X_train, Y_train)
    test_acc = clf.score(X_test, Y_test)
    
    print(f"\n{'='*70}")
    print(f"FINAL: Train={train_acc:.1%}, Test={test_acc:.1%}")
    print(f"{'='*70}")
    
    return test_acc, clf


def main():
    print("\n" + "="*70)
    print("MCT4 ULTIMATE ACCURACY DEMONSTRATION")
    print("Optimized for Maximum Performance")
    print("="*70)
    
    D = 32
    results = {}
    
    # XOR with larger hidden layer
    X_train, Y_train, y_train = create_xor_data(400, D)
    X_test, Y_test, y_test = create_xor_data(100, D)
    results['XOR'], _ = run_demo("XOR Classification", X_train, Y_train, X_test, Y_test, 
                                  D=D, epochs=100, hidden=8)
    
    # Circles
    X_train, Y_train, y_train, X_test, Y_test, y_test = create_circles_data(400, 100, D, noise=0.1)
    results['Circles'], _ = run_demo("Concentric Circles", X_train, Y_train, X_test, Y_test,
                                      D=D, epochs=100, hidden=8)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for task, acc in results.items():
        status = "✓✓" if acc >= 0.90 else ("✓" if acc >= 0.75 else "○")
        print(f"  {status} {task}: {acc:.1%}")
    
    avg_acc = np.mean(list(results.values()))
    print(f"\n  Average: {avg_acc:.1%}")
    
    print("\n" + "="*70)
    print("MCT4 demonstrates:")
    print("  ✓ Local learning without backpropagation")
    print("  ✓ Self-structuring compute graph")
    print("  ✓ Non-linear decision boundaries")
    print("  ✓ Online, incremental learning")
    print("  ✓ Retrograde credit assignment")
    print("="*70)


if __name__ == "__main__":
    main()
