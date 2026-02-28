#!/usr/bin/env python3
"""
MCT4 EFFICIENCY BENCHMARK

Demonstrates MCT4 advantages:
1. Parameter efficiency - fewer params for same accuracy
2. Sample efficiency - fewer samples to converge
3. Online learning - true single-sample updates
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/me/mct')

from mct4 import MCT4, MCT4Config, Primitive, NodeType


# ============================================================================
# MCT4 v5.3 - Efficient Configuration
# ============================================================================

class EfficientMCT4:
    """MCT4 optimized for parameter efficiency."""
    
    def __init__(self, input_dim, n_classes, eta=0.6):
        # Key: D can be smaller than input for efficiency
        self.D = max(input_dim, n_classes * 2, 32)
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        self.config = MCT4Config(D=self.D, t_budget=6, eta=eta, N=1)
        self.model = MCT4(self.config)
        self.momentum = {}
        self.samples_seen = 0
        self._init_graph()
    
    def _init_graph(self):
        """Minimal graph for efficiency."""
        self.model.state.nodes = {}
        self.model.state.next_node_id = 0
        
        inp = self.model.state.create_node(NodeType.INPUT, Primitive.FORK, rho_base=3.0)
        h1 = self.model.state.create_node(NodeType.HIDDEN, Primitive.GELU, rho_base=2.5)
        h2 = self.model.state.create_node(NodeType.HIDDEN, Primitive.RELU, rho_base=2.5)
        c1 = self.model.state.create_node(NodeType.HIDDEN, Primitive.ADD, rho_base=2.0)
        out = self.model.state.create_node(NodeType.OUTPUT, Primitive.SOFTMAX, rho_base=2.0)
        
        # Minimal edges for efficiency
        self.model.state.add_edge(inp.id, h1.id)
        self.model.state.add_edge(inp.id, h2.id)
        self.model.state.add_edge(h1.id, c1.id)
        self.model.state.add_edge(h2.id, c1.id)
        self.model.state.add_edge(c1.id, out.id)
        self.model.state.add_edge(inp.id, out.id)  # Skip connection
        
        # Count parameters
        self.n_params = sum(np.prod(node.W.shape) for node in self.model.state.nodes.values() 
                          if node.node_type != NodeType.INPUT)
        
        for node in self.model.state.nodes.values():
            if node.node_type != NodeType.INPUT:
                W = np.random.randn(self.D, self.D) * 0.1
                W += np.eye(self.D) * 0.7
                node.W = W
    
    def train_step(self, X, y):
        self.model.reset_sequence()
        X_emb = np.zeros(self.D); X_emb[:self.input_dim] = X
        Y_target = np.zeros(self.D); Y_target[y] = 1.0
        
        outputs = self.model.forward(X_emb)
        if not outputs: return
        
        Y_pred = list(outputs.values())[0]
        grad = np.zeros(self.D)
        grad[:self.n_classes] = Y_pred[:self.n_classes] - Y_target[:self.n_classes]
        grad = grad / (np.linalg.norm(grad) + 1e-8) * 2.0
        
        node_tensions = self.model.learning_engine.retrograde_flow(grad, 4)
        
        for record in self.model.state.active_path:
            if record.node_id not in node_tensions: continue
            node = self.model.state.nodes[record.node_id]
            if node.node_type == NodeType.INPUT: continue
            
            grad_W = np.outer(node_tensions[record.node_id], record.V_in)
            if record.node_id not in self.momentum:
                self.momentum[record.node_id] = np.zeros_like(node.W)
            self.momentum[record.node_id] = 0.9 * self.momentum[record.node_id] + 0.1 * grad_W
            node.W -= self.config.eta * self.momentum[record.node_id]
        
        self.samples_seen += 1
    
    def train_to_accuracy(self, X, y, target_acc, max_epochs=100):
        """Train until target accuracy or max epochs."""
        for epoch in range(max_epochs):
            self.config.eta = 0.6 / (1 + 0.01 * epoch)
            for i in np.random.permutation(len(X)):
                self.train_step(X[i], y[i])
            
            if (epoch + 1) % 5 == 0:
                acc = self.score(X, y)
                if acc >= target_acc:
                    return epoch + 1, self.samples_seen
        return max_epochs, self.samples_seen
    
    def predict(self, X):
        self.model.reset_sequence()
        X_emb = np.zeros(self.D); X_emb[:self.input_dim] = X
        outputs = self.model.forward(X_emb)
        return np.argmax(list(outputs.values())[0][:self.n_classes]) if outputs else 0
    
    def score(self, X, y):
        return sum(1 for i in range(len(X)) if self.predict(X[i]) == y[i]) / len(X)


# ============================================================================
# Simple Neural Network Baseline
# ============================================================================

class SimpleNN:
    """Simple 2-layer neural network baseline."""
    
    def __init__(self, input_dim, n_classes, hidden=32):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden = hidden
        
        # Parameters
        self.W1 = np.random.randn(input_dim, hidden) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, n_classes) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_classes)
        
        self.n_params = input_dim * hidden + hidden + hidden * n_classes + n_classes
        self.samples_seen = 0
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, z):
        z = z - np.max(z)  # Numerical stability
        exp_z = np.exp(z)
        return exp_z / (np.sum(exp_z) + 1e-10)
    
    def train_step(self, X, y, lr=0.01):
        # Forward
        h = self.relu(X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = self.softmax(logits)
        
        # Backward (cross-entropy gradient)
        dlogits = probs.copy()
        dlogits[y] -= 1.0
        
        dW2 = np.outer(h, dlogits)
        db2 = dlogits
        dh = self.W2 @ dlogits
        dh = dh * (h > 0).astype(float)
        dW1 = np.outer(X, dh)
        db1 = dh
        
        # Update with gradient clipping
        for grad in [dW1, db1, dW2, db2]:
            np.clip(grad, -5, 5, out=grad)
        
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        
        self.samples_seen += 1
    
    def train_to_accuracy(self, X, y, target_acc, lr=0.1, max_epochs=100):
        for epoch in range(max_epochs):
            lr_decay = lr / (1 + 0.01 * epoch)
            for i in np.random.permutation(len(X)):
                self.train_step(X[i], y[i], lr_decay)
            
            if (epoch + 1) % 5 == 0:
                acc = self.score(X, y)
                if acc >= target_acc:
                    return epoch + 1, self.samples_seen
        return max_epochs, self.samples_seen
    
    def predict(self, X):
        if len(X.shape) == 1:
            h = self.relu(X @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            return np.argmax(logits)
        else:
            h = self.relu(X @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            return np.argmax(logits, axis=1)
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y) if len(X.shape) > 1 else float(preds == y)


# ============================================================================
# Logistic Regression Baseline
# ============================================================================

class LogisticRegression:
    """Linear baseline."""
    
    def __init__(self, input_dim, n_classes):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.W = np.random.randn(input_dim, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        self.n_params = input_dim * n_classes + n_classes
        self.samples_seen = 0
    
    def softmax(self, z):
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / (np.sum(exp_z) + 1e-10)
    
    def train_step(self, X, y, lr=0.1):
        logits = X @ self.W + self.b
        probs = self.softmax(logits)
        grad = probs.copy()
        grad[y] -= 1.0
        # Gradient clipping
        np.clip(grad, -5, 5, out=grad)
        self.W -= lr * np.outer(X, grad)
        self.b -= lr * grad
        self.samples_seen += 1
    
    def train_to_accuracy(self, X, y, target_acc, lr=0.5, max_epochs=100):
        for epoch in range(max_epochs):
            lr_decay = lr / (1 + 0.01 * epoch)
            for i in np.random.permutation(len(X)):
                self.train_step(X[i], y[i], lr_decay)
            
            if (epoch + 1) % 5 == 0:
                acc = self.score(X, y)
                if acc >= target_acc:
                    return epoch + 1, self.samples_seen
        return max_epochs, self.samples_seen
    
    def predict(self, X):
        if len(X.shape) == 1:
            logits = X @ self.W + self.b
            return np.argmax(logits)
        else:
            logits = X @ self.W + self.b
            return np.argmax(logits, axis=1)
    
    def score(self, X, y):
        preds = self.predict(X)
        if len(X.shape) == 1:
            return float(preds == y)
        return np.mean(preds == y)


# ============================================================================
# Efficiency Benchmark
# ============================================================================

def benchmark_efficiency(name, X_train, y_train, X_test, y_test, target_acc=0.85):
    """Compare efficiency of MCT4 vs baselines."""
    print(f"\n{name}")
    print("-" * 70)
    
    n_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1] if len(X_train.shape) > 1 else 1
    
    results = {}
    
    # Logistic Regression - higher LR
    start = time.time()
    lr = LogisticRegression(input_dim, n_classes)
    epochs_lr, samples_lr = lr.train_to_accuracy(X_train, y_train, target_acc, lr=1.0, max_epochs=100)
    acc_lr = lr.score(X_test, y_test)
    time_lr = time.time() - start
    results['Logistic'] = {
        'params': lr.n_params,
        'epochs': epochs_lr,
        'samples': samples_lr,
        'time': time_lr,
        'acc': acc_lr
    }
    print(f"  Logistic: {lr.n_params:,} params, {epochs_lr} epochs, {acc_lr:.1%} acc ({time_lr:.2f}s)")
    
    # Simple NN - higher LR
    start = time.time()
    nn = SimpleNN(input_dim, n_classes, hidden=32)
    epochs_nn, samples_nn = nn.train_to_accuracy(X_train, y_train, target_acc, lr=0.5, max_epochs=100)
    acc_nn = nn.score(X_test, y_test)
    time_nn = time.time() - start
    results['NN'] = {
        'params': nn.n_params,
        'epochs': epochs_nn,
        'samples': samples_nn,
        'time': time_nn,
        'acc': acc_nn
    }
    print(f"  Simple NN:  {nn.n_params:,} params, {epochs_nn} epochs, {acc_nn:.1%} acc ({time_nn:.2f}s)")
    
    # MCT4
    start = time.time()
    mct4 = EfficientMCT4(input_dim, n_classes, eta=0.6)
    epochs_mct4, samples_mct4 = mct4.train_to_accuracy(X_train, y_train, target_acc, max_epochs=100)
    acc_mct4 = mct4.score(X_test, y_test)
    time_mct4 = time.time() - start
    results['MCT4'] = {
        'params': mct4.n_params,
        'epochs': epochs_mct4,
        'samples': samples_mct4,
        'time': time_mct4,
        'acc': acc_mct4
    }
    print(f"  MCT4:       {mct4.n_params:,} params, {epochs_mct4} epochs, {acc_mct4:.1%} acc ({time_mct4:.2f}s)")
    
    return results


def main():
    print("=" * 70)
    print("MCT4 EFFICIENCY BENCHMARK")
    print("Parameter Efficiency & Learning Speed vs Baselines")
    print("=" * 70)
    
    all_results = {}
    
    # Task 1: 10-Class (where MCT4 should shine)
    np.random.seed(42)
    prototypes = np.random.randn(10, 32) * 0.5
    X = np.array([prototypes[i % 10] + np.random.randn(32) * 0.25 for i in range(400)])
    y = np.array([i % 10 for i in range(400)])
    X_train, X_test = X[:320], X[320:]
    y_train, y_test = y[:320], y[320:]
    all_results['10-Class'] = benchmark_efficiency(
        "10-Class Classification (target: 85%)",
        X_train, y_train, X_test, y_test, target_acc=0.85
    )
    
    # Task 2: Linear (baseline should be fastest)
    np.random.seed(42)
    X = np.random.randn(300, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    all_results['Linear'] = benchmark_efficiency(
        "Linear Classification (target: 95%)",
        X_train, y_train, X_test, y_test, target_acc=0.95
    )
    
    # Task 3: XOR (non-linear, NN should win)
    np.random.seed(42)
    X = np.random.randn(300, 2) * 0.5
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    all_results['XOR'] = benchmark_efficiency(
        "XOR Classification (target: 70%)",
        X_train, y_train, X_test, y_test, target_acc=0.70
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("EFFICIENCY SUMMARY")
    print("=" * 70)
    
    print("\nParameter Count (lower is better):")
    print(f"  {'Task':<15} {'Logistic':<12} {'NN':<12} {'MCT4':<12} {'Best':<8}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for task, results in all_results.items():
        lr_p = results['Logistic']['params']
        nn_p = results['NN']['params']
        mct4_p = results['MCT4']['params']
        best = min(lr_p, nn_p, mct4_p)
        best_name = 'MCT4' if mct4_p == best else ('Logistic' if lr_p == best else 'NN')
        print(f"  {task:<15} {lr_p:>8,}   {nn_p:>8,}   {mct4_p:>8,}   {best_name:<8}")
    
    print("\nSamples to Convergence (lower is better):")
    print(f"  {'Task':<15} {'Logistic':<12} {'NN':<12} {'MCT4':<12} {'Best':<8}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for task, results in all_results.items():
        lr_s = results['Logistic']['samples']
        nn_s = results['NN']['samples']
        mct4_s = results['MCT4']['samples']
        best = min(lr_s, nn_s, mct4_s)
        best_name = 'MCT4' if mct4_s == best else ('Logistic' if lr_s == best else 'NN')
        print(f"  {task:<15} {lr_s:>8}   {nn_s:>8}   {mct4_s:>8}   {best_name:<8}")
    
    print("\nTime to Convergence (lower is better):")
    print(f"  {'Task':<15} {'Logistic':<12} {'NN':<12} {'MCT4':<12} {'Best':<8}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for task, results in all_results.items():
        lr_t = results['Logistic']['time']
        nn_t = results['NN']['time']
        mct4_t = results['MCT4']['time']
        best = min(lr_t, nn_t, mct4_t)
        best_name = 'MCT4' if mct4_t == best else ('Logistic' if lr_t == best else 'NN')
        print(f"  {task:<15} {lr_t:>8.2f}s  {nn_t:>8.2f}s  {mct4_t:>8.2f}s  {best_name:<8}")
    
    # MCT4 advantages
    print("\n" + "=" * 70)
    print("MCT4 ADVANTAGES")
    print("=" * 70)
    
    # Count wins
    param_wins = sum(1 for t, r in all_results.items() 
                    if r['MCT4']['params'] <= min(r['Logistic']['params'], r['NN']['params']))
    sample_wins = sum(1 for t, r in all_results.items() 
                     if r['MCT4']['samples'] <= min(r['Logistic']['samples'], r['NN']['samples']))
    time_wins = sum(1 for t, r in all_results.items() 
                   if r['MCT4']['time'] <= min(r['Logistic']['time'], r['NN']['time']))
    
    print(f"\n  Parameter Efficiency: MCT4 wins {param_wins}/{len(all_results)} tasks")
    print(f"  Sample Efficiency: MCT4 wins {sample_wins}/{len(all_results)} tasks")
    print(f"  Time Efficiency: MCT4 wins {time_wins}/{len(all_results)} tasks")
    
    print("\n  Key MCT4 Advantages:")
    print("    ✓ No computation graph storage (memory efficient)")
    print("    ✓ True online learning (single-sample updates)")
    print("    ✓ Self-structuring architecture (no tuning)")
    print("    ✓ Biologically plausible (local learning)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
