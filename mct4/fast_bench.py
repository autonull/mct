#!/usr/bin/env python3
"""
MCT4 FAST BENCHMARK - Results in <30 seconds
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/me/mct')

from mct4 import MCT4, MCT4Config, Primitive, NodeType


class FastMCT4:
    def __init__(self, input_dim, n_classes, eta=0.5):
        self.D = max(input_dim, n_classes * 2, 32)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.config = MCT4Config(D=self.D, t_budget=6, eta=eta, N=1)
        self.model = MCT4(self.config)
        self.momentum = {}
        self._init_graph()
    
    def _init_graph(self):
        self.model.state.nodes = {}
        self.model.state.next_node_id = 0
        inp = self.model.state.create_node(NodeType.INPUT, Primitive.FORK, rho_base=2.0)
        h1 = self.model.state.create_node(NodeType.HIDDEN, Primitive.GELU, rho_base=2.0)
        h2 = self.model.state.create_node(NodeType.HIDDEN, Primitive.RELU, rho_base=2.0)
        c1 = self.model.state.create_node(NodeType.HIDDEN, Primitive.ADD, rho_base=2.0)
        out = self.model.state.create_node(NodeType.OUTPUT, Primitive.SOFTMAX, rho_base=2.0)
        for h in [h1, h2]:
            self.model.state.add_edge(inp.id, h.id)
            self.model.state.add_edge(h.id, c1.id)
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
        for _ in range(epochs):
            for i in np.random.permutation(len(X)):
                self.train_step(X[i], y[i])
    
    def predict(self, X):
        self.model.reset_sequence()
        X_emb = np.zeros(self.D); X_emb[:self.input_dim] = X
        outputs = self.model.forward(X_emb)
        return np.argmax(list(outputs.values())[0][:self.n_classes]) if outputs else 0
    
    def score(self, X, y):
        return sum(1 for i in range(len(X)) if self.predict(X[i]) == y[i]) / len(X)


def run_bench():
    print("=" * 55)
    print("MCT4 FAST BENCHMARK (<30s)")
    print("=" * 55)
    
    results = {}
    start_all = time.time()
    
    # XOR
    np.random.seed(42)
    X = np.random.randn(200, 2) * 0.5
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    clf = FastMCT4(2, 2, eta=0.8)
    clf.train(X, y, epochs=40)
    results['XOR'] = clf.score(X, y)
    
    # 10-Class
    np.random.seed(42)
    prototypes = np.random.randn(10, 32) * 0.5
    X = np.array([prototypes[i % 10] + np.random.randn(32) * 0.25 for i in range(300)])
    y = np.array([i % 10 for i in range(300)])
    clf = FastMCT4(32, 10, eta=0.5)
    clf.train(X, y, epochs=30)
    results['10-Class'] = clf.score(X, y)
    
    # Moons
    np.random.seed(42)
    n_half = 100
    theta = np.linspace(0, np.pi, n_half)
    X = np.vstack([
        np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.randn(n_half, 2) * 0.1,
        np.column_stack([np.cos(theta)+1, np.sin(theta)-0.5]) + np.random.randn(n_half, 2) * 0.1
    ])
    y = np.hstack([np.zeros(n_half), np.ones(n_half)]).astype(int)
    clf = FastMCT4(2, 2, eta=0.5)
    clf.train(X, y, epochs=30)
    results['Moons'] = clf.score(X, y)
    
    # Linear
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = FastMCT4(2, 2, eta=0.5)
    clf.train(X, y, epochs=20)
    results['Linear'] = clf.score(X, y)
    
    elapsed = time.time() - start_all
    
    print("\nRESULTS:")
    print("-" * 55)
    for task, acc in sorted(results.items(), key=lambda x: -x[1]):
        status = "✓✓✓" if acc >= 0.95 else ("✓✓" if acc >= 0.85 else ("✓" if acc >= 0.70 else "•"))
        print(f"  {status} {task}: {acc:.1%}")
    
    avg = np.mean(list(results.values()))
    print(f"\n  Average: {avg:.1%} (time: {elapsed:.1f}s)")
    print("=" * 55)
    
    if avg >= 0.85:
        print("  MCT4 IS COMPETITIVE")
    elif avg >= 0.70:
        print("  MCT4 SHOWS PROMISE")
    else:
        print("  MCT4 NEEDS WORK")
    print("=" * 55)


if __name__ == "__main__":
    run_bench()
