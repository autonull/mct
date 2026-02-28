#!/usr/bin/env python3
"""
MCT4 Balanced Demonstration

Properly tuned for each task type.
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/me/mct')

from mct4 import MCT4, MCT4Config, Primitive, NodeType


class TaskClassifier:
    """MCT4 classifier tuned for specific tasks."""
    
    def __init__(self, input_dim, n_classes, eta=0.5, hidden=3):
        self.D = max(input_dim, n_classes * 2, 32)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.eta = eta
        
        self.config = MCT4Config(D=self.D, t_budget=6, eta=eta, N=1)
        self.model = MCT4(self.config)
        self.momentum = {}
        self._init_graph(hidden)
    
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
            node.W -= self.eta * self.momentum[record.node_id]
    
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


def demo_10class():
    """10-class classification."""
    print("\n" + "="*55)
    print("TASK 1: 10-Class Classification")
    print("="*55)
    
    np.random.seed(42)
    prototypes = np.random.randn(10, 32) * 0.5
    
    X_train = np.array([prototypes[i % 10] + np.random.randn(32) * 0.25 for i in range(400)])
    y_train = np.array([i % 10 for i in range(400)])
    X_test = np.array([prototypes[i % 10] + np.random.randn(32) * 0.25 for i in range(100)])
    y_test = np.array([i % 10 for i in range(100)])
    
    clf = TaskClassifier(input_dim=32, n_classes=10, eta=0.5, hidden=4)
    print(f"Graph: {len(clf.model.state.nodes)} nodes")
    
    start = time.time()
    clf.train(X_train, y_train, epochs=30)
    
    print(f"  Train: {clf.score(X_train, y_train):.1%}")
    print(f"  Test:  {clf.score(X_test, y_test):.1%} ({time.time()-start:.1f}s)")
    return clf.score(X_test, y_test)


def demo_xor():
    """XOR classification."""
    print("\n" + "="*55)
    print("TASK 2: XOR (Non-linear)")
    print("="*55)
    
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2) * 0.5
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    
    clf = TaskClassifier(input_dim=2, n_classes=2, eta=0.8, hidden=4)
    print(f"Graph: {len(clf.model.state.nodes)} nodes")
    
    start = time.time()
    clf.train(X, y, epochs=80)
    
    acc = clf.score(X, y)
    print(f"  Accuracy: {acc:.1%} ({time.time()-start:.1f}s)")
    return acc


def demo_sequence():
    """Sequence pattern."""
    print("\n" + "="*55)
    print("TASK 3: Sequence (abc pattern)")
    print("="*55)
    
    pattern = "abc" * 30
    chars = sorted(set(pattern))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    
    pairs = [(c2i[pattern[i]], c2i[pattern[i+1]]) for i in range(len(pattern)-1)]
    
    # Simple sequence model
    D = 48
    vocab = 3
    
    config = MCT4Config(D=D, t_budget=6, eta=0.3, N=1)
    model = MCT4(config)
    model.state.nodes = {}
    model.state.next_node_id = 0
    
    inp = model.state.create_node(NodeType.INPUT, Primitive.FORK, rho_base=2.0)
    h1 = model.state.create_node(NodeType.HIDDEN, Primitive.GELU, rho_base=2.0)
    h2 = model.state.create_node(NodeType.HIDDEN, Primitive.GATE, rho_base=2.0)
    out = model.state.create_node(NodeType.OUTPUT, Primitive.SOFTMAX, rho_base=2.0)
    
    model.state.add_edge(inp.id, h1.id)
    model.state.add_edge(inp.id, h2.id)
    model.state.add_edge(h1.id, out.id)
    model.state.add_edge(h2.id, out.id)
    model.state.add_edge(inp.id, out.id)
    
    for node in model.state.nodes.values():
        if node.node_type != NodeType.INPUT:
            node.W = np.eye(D) * 0.6 + np.random.randn(D, D) * 0.15
    
    momentum = {}
    context = []
    
    def train_step(ci, co):
        nonlocal context
        model.reset_sequence()
        
        X = np.zeros(D)
        X[ci] = 1.0
        if context:
            X[vocab:vocab*2] = np.mean(context[-3:], axis=0)[:vocab] * 0.5
        
        Y = np.zeros(D); Y[co] = 1.0
        
        outputs = model.forward(X)
        if not outputs: return
        
        grad = list(outputs.values())[0] - Y
        node_tensions = model.learning_engine.retrograde_flow(grad, 3)
        
        for record in model.state.active_path:
            if record.node_id not in node_tensions: continue
            node = model.state.nodes[record.node_id]
            if node.node_type == NodeType.INPUT: continue
            
            grad_W = np.outer(node_tensions[record.node_id], record.V_in)
            if record.node_id not in momentum:
                momentum[record.node_id] = np.zeros_like(node.W)
            momentum[record.node_id] = 0.9 * momentum[record.node_id] + 0.1 * grad_W
            node.W -= 0.3 * momentum[record.node_id]
        
        context.append(list(outputs.values())[0].copy())
        if len(context) > 5: context.pop(0)
    
    print(f"Graph: {len(model.state.nodes)} nodes")
    start = time.time()
    
    for epoch in range(40):
        context = []
        for i in np.random.permutation(len(pairs)):
            train_step(pairs[i][0], pairs[i][1])
    
    # Test
    correct = sum(1 for ci, co in pairs if (
        model.reset_sequence() or
        model.forward(np.array([1.0 if j == ci else 0.0 for j in range(D)] + [0.0]*(D-D))) or
        True  # Always true, just need to call
    ) and np.argmax(list(model.forward(np.zeros(D)) or [np.zeros(D)])[0][:vocab]) == co)
    
    # Simpler test
    correct = 0
    for ci, co in pairs:
        model.reset_sequence()
        X = np.zeros(D); X[ci] = 1.0
        outputs = model.forward(X)
        pred = np.argmax(list(outputs.values())[0][:vocab]) if outputs else 0
        if pred == co: correct += 1
    
    acc = correct / len(pairs)
    print(f"  Accuracy: {acc:.1%} ({time.time()-start:.1f}s)")
    
    # Generate
    model.reset_sequence()
    gen = ['a']
    idx = 0
    for _ in range(10):
        X = np.zeros(D); X[idx] = 1.0
        outputs = model.forward(X)
        idx = np.argmax(list(outputs.values())[0][:vocab]) if outputs else 0
        gen.append(i2c.get(idx, '?'))
    print(f"  Generated: '{''.join(gen)}'")
    
    return acc


def main():
    print("\n" + "="*55)
    print("MCT4 BALANCED DEMONSTRATION")
    print("Properly Tuned for Each Task")
    print("="*55)
    
    r1 = demo_10class()
    r2 = demo_xor()
    r3 = demo_sequence()
    
    print("\n" + "="*55)
    print("RESULTS")
    print("="*55)
    print(f"  10-Class: {r1:.1%}")
    print(f"  XOR:      {r2:.1%}")
    print(f"  Sequence: {r3:.1%}")
    print(f"  Average:  {np.mean([r1,r2,r3]):.1%}")
    
    print("\n" + "="*55)
    print("MCT4: Local Learning Without Backpropagation")
    print("="*55)


if __name__ == "__main__":
    main()
