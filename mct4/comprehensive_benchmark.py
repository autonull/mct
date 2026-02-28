#!/usr/bin/env python3
"""
MCT4 Comprehensive Benchmark Suite

Benchmark MCT4 against standard baselines:
- Logistic Regression (linear baseline)
- Simple Neural Network (gradient-based baseline)
- Random Forest (non-linear baseline)

Tasks:
1. Classification (multiple datasets)
2. Regression (multiple functions)
3. Pattern/Sequence learning
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/me/mct')

from mct4 import MCT4, MCT4Config, Primitive, NodeType


# ============================================================================
# MCT4 IMPLEMENTATIONS
# ============================================================================

class MCT4Classifier:
    """MCT4 classifier optimized for benchmark performance."""
    
    def __init__(self, input_dim, n_classes, hidden_size=32, eta=0.5):
        self.D = max(input_dim, n_classes * 2, 48)
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        self.config = MCT4Config(D=self.D, t_budget=6, eta=eta, N=1)
        self.model = MCT4(self.config)
        self.momentum = {}
        self._init_graph(hidden_size)
    
    def _init_graph(self, n_hidden):
        self.model.state.nodes = {}
        self.model.state.next_node_id = 0
        
        inp = self.model.state.create_node(NodeType.INPUT, Primitive.FORK, rho_base=2.0)
        
        hiddens = []
        prims = [Primitive.GELU, Primitive.RELU, Primitive.TANH, Primitive.GELU]
        for i in range(n_hidden):
            h = self.model.state.create_node(NodeType.HIDDEN, prims[i % 4], rho_base=2.0)
            hiddens.append(h)
            self.model.state.add_edge(inp.id, h.id)
        
        c1 = self.model.state.create_node(NodeType.HIDDEN, Primitive.ADD, rho_base=2.0)
        for h in hiddens:
            self.model.state.add_edge(h.id, c1.id)
        
        out = self.model.state.create_node(NodeType.OUTPUT, Primitive.SOFTMAX, rho_base=2.0)
        self.model.state.add_edge(c1.id, out.id)
        self.model.state.add_edge(inp.id, out.id)
        
        for node in self.model.state.nodes.values():
            if node.node_type != NodeType.INPUT:
                node.W = np.eye(self.D) * 0.6 + np.random.randn(self.D, self.D) * 0.15
    
    def train_step(self, X, y):
        self.model.reset_sequence()
        X_emb = np.zeros(self.D); X_emb[:self.input_dim] = X
        Y_target = np.zeros(self.D); Y_target[y] = 1.0
        
        outputs = self.model.forward(X_emb)
        if not outputs: return
        
        grad = list(outputs.values())[0] - Y_target
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
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for i in np.random.permutation(len(X)):
                self.train_step(X[i], y[i])
    
    def predict(self, X):
        self.model.reset_sequence()
        X_emb = np.zeros(self.D); X_emb[:self.input_dim] = X
        outputs = self.model.forward(X_emb)
        if not outputs: return 0
        return np.argmax(list(outputs.values())[0][:self.n_classes])
    
    def score(self, X, y):
        return sum(1 for i in range(len(X)) if self.predict(X[i]) == y[i]) / len(X)


class MCT4Regressor:
    """MCT4 regressor."""
    
    def __init__(self, input_dim=1, hidden_size=24, eta=0.2):
        self.D = 48
        self.input_dim = input_dim
        
        self.config = MCT4Config(D=self.D, t_budget=6, eta=eta, N=1)
        self.model = MCT4(self.config)
        self.momentum = {}
        self.output_scale = 1.0
        self._init_graph(hidden_size)
    
    def _init_graph(self, n_hidden):
        self.model.state.nodes = {}
        self.model.state.next_node_id = 0
        
        inp = self.model.state.create_node(NodeType.INPUT, Primitive.FORK, rho_base=2.0)
        
        hiddens = []
        prims = [Primitive.GELU, Primitive.RELU, Primitive.TANH]
        for i in range(n_hidden):
            h = self.model.state.create_node(NodeType.HIDDEN, prims[i % 3], rho_base=2.0)
            hiddens.append(h)
            self.model.state.add_edge(inp.id, h.id)
        
        c1 = self.model.state.create_node(NodeType.HIDDEN, Primitive.ADD, rho_base=2.0)
        for h in hiddens:
            self.model.state.add_edge(h.id, c1.id)
        
        out = self.model.state.create_node(NodeType.OUTPUT, Primitive.FORK, rho_base=2.0)
        self.model.state.add_edge(c1.id, out.id)
        self.model.state.add_edge(inp.id, out.id)
        
        for node in self.model.state.nodes.values():
            if node.node_type != NodeType.INPUT:
                node.W = np.eye(self.D) * 0.5 + np.random.randn(self.D, self.D) * 0.2
    
    def train_step(self, X, y):
        self.model.reset_sequence()
        X_emb = np.zeros(self.D)
        if np.isscalar(X):
            X_emb[0] = X
        else:
            X_emb[:len(X)] = X
        
        outputs = self.model.forward(X_emb)
        if not outputs: return
        
        Y_pred = list(outputs.values())[0][0] * self.output_scale
        error = (Y_pred - y) * self.output_scale
        
        grad = np.zeros(self.D)
        grad[0] = np.clip(error, -5, 5)
        
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
    
    def train(self, X, y, epochs):
        self.output_scale = max(1.0, np.std(y) * 2)
        for epoch in range(epochs):
            for i in np.random.permutation(len(X)):
                self.train_step(X[i], y[i])
    
    def predict(self, X):
        self.model.reset_sequence()
        X_emb = np.zeros(self.D)
        if np.isscalar(X):
            X_emb[0] = X
        else:
            X_emb[:len(X)] = X
        outputs = self.model.forward(X_emb)
        if not outputs: return 0.0
        return list(outputs.values())[0][0] * self.output_scale
    
    def r2(self, X, y):
        preds = np.array([self.predict(x) for x in X])
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return max(-1, 1 - ss_res / ss_tot)


# ============================================================================
# BASELINE IMPLEMENTATIONS
# ============================================================================

class LogisticRegression:
    """Simple logistic regression baseline."""
    
    def __init__(self, input_dim, n_classes, lr=0.1):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.W = np.random.randn(input_dim, n_classes) * 0.01
        self.b = np.zeros(n_classes)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def train(self, X, y, epochs):
        n = len(X)
        for epoch in range(epochs):
            for i in np.random.permutation(n):
                logits = X[i] @ self.W + self.b
                probs = self.softmax(logits)
                grad = probs.copy()
                grad[y[i]] -= 1
                self.W -= self.lr * np.outer(X[i], grad)
                self.b -= self.lr * grad
    
    def predict(self, X):
        logits = X @ self.W + self.b
        return np.argmax(logits, axis=-1)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class SimpleNN:
    """Simple 2-layer neural network baseline."""
    
    def __init__(self, input_dim, n_classes, hidden=32, lr=0.01):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = np.random.randn(input_dim, hidden) * scale1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, n_classes) * scale2
        self.b2 = np.zeros(n_classes)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_grad(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def train(self, X, y, epochs):
        n = len(X)
        for epoch in range(epochs):
            for i in np.random.permutation(n):
                x = X[i]
                target = y[i]
                
                # Forward
                h = self.relu(x @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = self.softmax(logits)
                
                # Backward
                dlogits = probs.copy()
                dlogits[target] -= 1
                
                dW2 = np.outer(h, dlogits)
                db2 = dlogits
                
                dh = self.W2 @ dlogits
                dh = dh * self.relu_grad(x @ self.W1 + self.b1)
                
                dW1 = np.outer(x, dh)
                db1 = dh
                
                # Update
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
    
    def predict(self, X):
        h = self.relu(X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return np.argmax(logits, axis=-1)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ============================================================================
# BENCHMARK DATASETS
# ============================================================================

def get_xor_data(n=400):
    """XOR dataset."""
    np.random.seed(42)
    X = np.random.randn(n, 2) * 0.5
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    return X, y


def get_circles_data(n=400):
    """Concentric circles dataset."""
    np.random.seed(42)
    n_half = n // 2
    theta1 = np.random.uniform(0, 2*np.pi, n_half)
    r1 = np.random.uniform(0.5, 1.0, n_half)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    
    theta2 = np.random.uniform(0, 2*np.pi, n_half)
    r2 = np.random.uniform(1.5, 2.5, n_half)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_half), np.ones(n_half)]).astype(int)
    return X, y


def get_moons_data(n=400):
    """Two moons dataset."""
    np.random.seed(42)
    n_half = n // 2
    theta = np.linspace(0, np.pi, n_half)
    
    x1 = np.cos(theta) + np.random.randn(n_half) * 0.1
    y1 = np.sin(theta) + np.random.randn(n_half) * 0.1
    x2 = np.cos(theta) + 1 + np.random.randn(n_half) * 0.1
    y2 = np.sin(theta) - 0.5 + np.random.randn(n_half) * 0.1
    
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_half), np.ones(n_half)]).astype(int)
    return X, y


def get_digits_data(n=500):
    """Synthetic digits-like dataset."""
    np.random.seed(42)
    n_classes = 10
    prototypes = np.random.randn(n_classes, 32) * 0.5
    
    X = np.array([prototypes[i % n_classes] + np.random.randn(32) * 0.25 
                  for i in range(n)])
    y = np.array([i % n_classes for i in range(n)])
    return X, y


def get_regression_data(func='quadratic', n=300):
    """Regression datasets."""
    np.random.seed(42)
    if func == 'quadratic':
        X = np.random.uniform(-2, 2, n)
        y = X**2 + np.random.randn(n) * 0.1
    elif func == 'sine':
        X = np.random.uniform(-np.pi, np.pi, n)
        y = np.sin(X) + np.random.randn(n) * 0.1
    elif func == 'cubic':
        X = np.random.uniform(-1.5, 1.5, n)
        y = X**3 - X + np.random.randn(n) * 0.1
    return X, y


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_classification(name, X, y, epochs=40):
    """Benchmark classifiers on a dataset."""
    print(f"\n{name}")
    print("-" * 50)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    n_classes = len(np.unique(y))
    input_dim = X.shape[1] if len(X.shape) > 1 else 1
    
    results = {}
    
    # Logistic Regression
    start = time.time()
    lr = LogisticRegression(input_dim, n_classes, lr=0.5)
    lr.train(X_train, y_train, epochs=epochs)
    lr_acc = lr.score(X_test, y_test)
    results['Logistic'] = lr_acc
    print(f"  Logistic Regression: {lr_acc:.1%} ({time.time()-start:.2f}s)")
    
    # Simple NN
    start = time.time()
    nn = SimpleNN(input_dim, n_classes, hidden=32, lr=0.1)
    nn.train(X_train, y_train, epochs=epochs)
    nn_acc = nn.score(X_test, y_test)
    results['SimpleNN'] = nn_acc
    print(f"  Simple Neural Net:   {nn_acc:.1%} ({time.time()-start:.2f}s)")
    
    # MCT4
    start = time.time()
    mct4 = MCT4Classifier(input_dim, n_classes, hidden_size=4, eta=0.5)
    mct4.train(X_train, y_train, epochs=epochs)
    mct4_acc = mct4.score(X_test, y_test)
    results['MCT4'] = mct4_acc
    print(f"  MCT4:                {mct4_acc:.1%} ({time.time()-start:.2f}s)")
    
    return results


def benchmark_regression(name, X, y, epochs=60):
    """Benchmark regressors on a dataset."""
    print(f"\n{name}")
    print("-" * 50)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    results = {}
    
    # Simple NN regressor
    start = time.time()
    nn = SimpleNN(1, 1, hidden=24, lr=0.05)
    for epoch in range(epochs):
        for i in np.random.permutation(len(X_train)):
            x = X_train[i:i+1]
            h = nn.relu(x @ nn.W1 + nn.b1)
            pred = (h @ nn.W2 + nn.b2)[0, 0]
            error = pred - y_train[i]
            nn.W2 -= nn.lr * error * h.T
            nn.b2 -= nn.lr * error
    nn_r2 = nn.r2(X_test.reshape(-1, 1), y_test)
    results['SimpleNN'] = nn_r2
    print(f"  Simple Neural Net:   R²={nn_r2:.3f} ({time.time()-start:.2f}s)")
    
    # MCT4
    start = time.time()
    mct4 = MCT4Regressor(input_dim=1, hidden_size=24, eta=0.2)
    mct4.train(X_train, y_train, epochs=epochs)
    mct4_r2 = mct4.r2(X_test, y_test)
    results['MCT4'] = mct4_r2
    print(f"  MCT4:                R²={mct4_r2:.3f} ({time.time()-start:.2f}s)")
    
    return results


def main():
    print("=" * 60)
    print("MCT4 COMPREHENSIVE BENCHMARK")
    print("vs Logistic Regression & Neural Networks")
    print("=" * 60)
    
    all_results = {'classification': {}, 'regression': {}}
    
    # Classification benchmarks
    print("\n" + "=" * 60)
    print("CLASSIFICATION BENCHMARKS")
    print("=" * 60)
    
    all_results['classification']['XOR'] = benchmark_classification(
        "XOR (Non-linear)", *get_xor_data(400), epochs=50
    )
    all_results['classification']['Circles'] = benchmark_classification(
        "Concentric Circles", *get_circles_data(400), epochs=50
    )
    all_results['classification']['Moons'] = benchmark_classification(
        "Two Moons", *get_moons_data(400), epochs=50
    )
    all_results['classification']['Digits'] = benchmark_classification(
        "10-Class Digits", *get_digits_data(500), epochs=40
    )
    
    # Regression benchmarks
    print("\n" + "=" * 60)
    print("REGRESSION BENCHMARKS")
    print("=" * 60)
    
    all_results['regression']['Quadratic'] = benchmark_regression(
        "Quadratic (x²)", *get_regression_data('quadratic', 300), epochs=80
    )
    all_results['regression']['Sine'] = benchmark_regression(
        "Sine", *get_regression_data('sine', 300), epochs=80
    )
    all_results['regression']['Cubic'] = benchmark_regression(
        "Cubic (x³-x)", *get_regression_data('cubic', 300), epochs=80
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nClassification Accuracy:")
    print(f"  {'Task':<20} {'Logistic':<12} {'NN':<12} {'MCT4':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    
    for task, results in all_results['classification'].items():
        logistic = results.get('Logistic', 0)
        nn = results.get('SimpleNN', 0)
        mct4 = results.get('MCT4', 0)
        print(f"  {task:<20} {logistic:>8.1%}   {nn:>8.1%}   {mct4:>8.1%}")
    
    print("\nRegression R²:")
    print(f"  {'Task':<20} {'NN':<12} {'MCT4':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    
    for task, results in all_results['regression'].items():
        nn = results.get('SimpleNN', 0)
        mct4 = results.get('MCT4', 0)
        print(f"  {task:<20} {nn:>8.3f}   {mct4:>8.3f}")
    
    # Averages
    print("\n" + "=" * 60)
    print("AVERAGES")
    print("=" * 60)
    
    class_avg = {
        'Logistic': np.mean([r.get('Logistic', 0) for r in all_results['classification'].values()]),
        'SimpleNN': np.mean([r.get('SimpleNN', 0) for r in all_results['classification'].values()]),
        'MCT4': np.mean([r.get('MCT4', 0) for r in all_results['classification'].values()]),
    }
    
    reg_avg = {
        'SimpleNN': np.mean([r.get('SimpleNN', 0) for r in all_results['regression'].values()]),
        'MCT4': np.mean([r.get('MCT4', 0) for r in all_results['regression'].values()]),
    }
    
    print(f"\nClassification: Logistic={class_avg['Logistic']:.1%}, NN={class_avg['SimpleNN']:.1%}, MCT4={class_avg['MCT4']:.1%}")
    print(f"Regression R²:  NN={reg_avg['SimpleNN']:.3f}, MCT4={reg_avg['MCT4']:.3f}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    if class_avg['MCT4'] >= class_avg['SimpleNN'] * 0.9:
        print("\n✓ MCT4 achieves 90%+ of NN performance on classification")
    if class_avg['MCT4'] >= 0.8:
        print("✓ MCT4 achieves 80%+ average classification accuracy")
    
    print("\nMCT4 Advantages:")
    print("  • No backpropagation required")
    print("  • No computation graph storage")
    print("  • Online, incremental learning")
    print("  • Self-structuring architecture")
    print("  • Biologically plausible")
    
    print("\nAreas for Improvement:")
    print("  • Regression performance")
    print("  • Sequence modeling")
    print("  • Deep architecture scaling")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
