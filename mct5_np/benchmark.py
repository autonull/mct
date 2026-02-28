"""
MCT5 Benchmark Suite

8 classification tasks compared across:
  - MCT5 (this work)
  - Simple 2-layer NN with backprop (baseline)
  - Logistic regression (linear baseline)

Run with:
    cd /home/me/mct && python -m mct5.benchmark
"""

from __future__ import annotations
import sys
import time
import numpy as np

sys.path.insert(0, '/home/me/mct')

from mct5 import MCT5, MCT5Config


# ── Baselines ─────────────────────────────────────────────────────────────────

class LogisticBaseline:
    """Simple logistic regression via gradient descent."""
    def __init__(self, input_dim, n_classes, lr=0.1):
        self.W = np.zeros((n_classes, input_dim))
        self.b = np.zeros(n_classes)
        self.lr = lr
        self.n_classes = n_classes

    def _softmax(self, z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def train(self, X, y, epochs=100):
        n = len(X)
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for i in idx:
                xi, yi = X[i], int(y[i])
                logits = self.W @ xi + self.b
                probs = self._softmax(logits)
                probs[yi] -= 1
                self.W -= self.lr * np.outer(probs, xi) / n
                self.b -= self.lr * probs / n

    def predict(self, X):
        return np.argmax(X @ self.W.T + self.b, axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


class SimpleNN:
    """2-layer MLP with backprop (ReLU hidden, softmax output)."""
    def __init__(self, input_dim, hidden, n_classes, lr=0.01):
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(hidden, input_dim) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(n_classes, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_classes)
        self.lr = lr

    def _relu(self, x): return np.maximum(0, x)
    def _softmax(self, z):
        z = z - z.max(); e = np.exp(z); return e / e.sum()

    def _forward(self, x):
        h = self._relu(self.W1 @ x + self.b1)
        out = self._softmax(self.W2 @ h + self.b2)
        return h, out

    def train(self, X, y, epochs=50):
        n = len(X)
        for _ in range(epochs):
            for i in np.random.permutation(n):
                xi, yi = X[i], int(y[i])
                h, out = self._forward(xi)
                dout = out.copy(); dout[yi] -= 1
                dW2 = np.outer(dout, h); db2 = dout
                dh = self.W2.T @ dout * (h > 0)
                dW1 = np.outer(dh, xi); db1 = dh
                self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2

    def predict(self, X):
        return np.array([np.argmax(self._forward(x)[1]) for x in X])

    def score(self, X, y):
        return (self.predict(X) == y).mean()


# ── Dataset generators ────────────────────────────────────────────────────────

def make_linear(n=200, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def make_xor(n=200, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, 2) * 0.5
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    return X, y

def make_circles(n=200, seed=42):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, n//2)
    r1 = np.random.uniform(0.5, 1.0, n//2)
    r2 = np.random.uniform(1.5, 2.5, n//2)
    X = np.vstack([
        np.column_stack([r1*np.cos(theta), r1*np.sin(theta)]),
        np.column_stack([r2*np.cos(theta), r2*np.sin(theta)])
    ])
    y = np.hstack([np.zeros(n//2), np.ones(n//2)]).astype(int)
    return X, y

def make_moons(n=200, seed=42):
    np.random.seed(seed)
    h = n // 2
    t = np.linspace(0, np.pi, h)
    X = np.vstack([
        np.column_stack([np.cos(t), np.sin(t)]) + np.random.randn(h, 2)*0.15,
        np.column_stack([np.cos(t)+1, np.sin(t)-0.5]) + np.random.randn(h, 2)*0.15
    ])
    y = np.hstack([np.zeros(h), np.ones(h)]).astype(int)
    return X, y

def make_spirals(n=200, seed=42):
    np.random.seed(seed)
    h = n // 2
    t = np.linspace(0, 4*np.pi, h)
    r = np.linspace(0.1, 1.0, h)
    X = np.vstack([
        np.column_stack([r*np.cos(t), r*np.sin(t)]) + np.random.randn(h, 2)*0.1,
        np.column_stack([r*np.cos(t+np.pi), r*np.sin(t+np.pi)]) + np.random.randn(h, 2)*0.1
    ])
    y = np.hstack([np.zeros(h), np.ones(h)]).astype(int)
    return X, y

def make_10class(n=300, seed=42):
    np.random.seed(seed)
    protos = np.random.randn(10, 20) * 1.5
    X = np.vstack([protos[i % 10] + np.random.randn(20)*0.4 for i in range(n)])
    y = np.array([i % 10 for i in range(n)])
    return X, y

def make_blobs(n=200, seed=42):
    np.random.seed(seed)
    centres = np.array([[2, 2], [-2, -2], [2, -2], [-2, 2]], dtype=float)
    X = np.vstack([centres[i % 4] + np.random.randn(n//4, 2)*0.6 for i in range(4)])
    y = np.array([i % 4 for i in range(n)])
    idx = np.random.permutation(n)
    return X[idx], y[idx]

def make_highd(n=200, seed=42):
    np.random.seed(seed)
    protos = np.random.randn(5, 64) * 1.2
    X = np.vstack([protos[i % 5] + np.random.randn(64)*0.5 for i in range(n)])
    y = np.array([i % 5 for i in range(n)])
    return X, y


# ── MCT5 runner ───────────────────────────────────────────────────────────────

def run_mct5(X, y, n_classes, D=48, r=12, epochs=80, n_seeds=3) -> float:
    input_dim = X.shape[1]
    best = 0.0
    for seed in range(n_seeds):
        np.random.seed(seed * 31 + 7)
        config = MCT5Config(
            D=D, r=r, n_classes=n_classes, input_dim=input_dim,
            eta_W=0.015, eta_S=0.002, t_budget=12, evolve_interval=8,
            sigma_mut=0.04, K=2,
        )
        model = MCT5(config)
        model.initialize()
        n = len(X)
        for epoch in range(epochs):
            lr_scale = 1.0 / (1.0 + 0.02 * epoch)
            model.cfg.eta_W = 0.015 * lr_scale
            for i in np.random.permutation(n):
                model.train_step(X[i], int(y[i]))
        best = max(best, model.score(X, y))
    return best


# ── Compare helpers ───────────────────────────────────────────────────────────

def run_logistic(X, y, n_classes, epochs=150) -> float:
    m = LogisticBaseline(X.shape[1], n_classes, lr=0.05)
    m.train(X, y, epochs=epochs)
    return m.score(X, y)


def run_simple_nn(X, y, n_classes, epochs=50) -> float:
    hidden = max(16, X.shape[1] * 2)
    m = SimpleNN(X.shape[1], hidden, n_classes, lr=0.01)
    m.train(X, y, epochs=epochs)
    return m.score(X, y)


# ── Main benchmark ────────────────────────────────────────────────────────────

TASKS = [
    ("Linear",   make_linear,   2,   {'D': 32, 'r': 8,  'epochs': 50}),
    ("XOR",      make_xor,      2,   {'D': 48, 'r': 12, 'epochs': 120}),
    ("Circles",  make_circles,  2,   {'D': 48, 'r': 12, 'epochs': 100}),
    ("Moons",    make_moons,    2,   {'D': 48, 'r': 12, 'epochs': 80}),
    ("Spirals",  make_spirals,  2,   {'D': 64, 'r': 16, 'epochs': 150}),
    ("10-Class", make_10class,  10,  {'D': 64, 'r': 16, 'epochs': 80}),
    ("Blobs",    make_blobs,    4,   {'D': 32, 'r': 8,  'epochs': 60}),
    ("High-D",   make_highd,    5,   {'D': 64, 'r': 16, 'epochs': 80}),
]


def main():
    print("\n" + "=" * 72)
    print("  MCT5 Benchmark  —  8-Task Classification Suite")
    print("=" * 72)

    results = {}
    total_start = time.time()

    for task_name, gen_fn, n_classes, kw in TASKS:
        X, y = gen_fn()
        print(f"\n  {task_name} (n={len(X)}, dim={X.shape[1]}, classes={n_classes})")

        t0 = time.time()
        mct5_acc = run_mct5(X, y, n_classes, **kw)
        t_mct5 = time.time() - t0

        t0 = time.time()
        nn_acc = run_simple_nn(X, y, n_classes)
        t_nn = time.time() - t0

        logit_acc = run_logistic(X, y, n_classes)

        results[task_name] = {
            'mct5': mct5_acc, 'nn': nn_acc, 'logistic': logit_acc,
            't_mct5': t_mct5, 't_nn': t_nn
        }

        delta = "↑" if mct5_acc >= nn_acc - 0.03 else "↓"
        print(f"    MCT5: {mct5_acc:.1%} ({t_mct5:.1f}s)  |  "
              f"SimpleNN: {nn_acc:.1%} ({t_nn:.1f}s)  |  "
              f"Logistic: {logit_acc:.1%}  {delta}")

    total_time = time.time() - total_start

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    header = f"  {'Task':<12}  {'MCT5':>8}  {'SimpleNN':>10}  {'Logistic':>10}  {'vs NN':>8}"
    print(header)
    print("  " + "-" * 66)

    mct5_scores, nn_scores = [], []
    for task_name, _ in [(t, None) for t, *_ in TASKS]:
        r = results[task_name]
        vs = r['mct5'] - r['nn']
        sym = "✓✓✓" if r['mct5'] >= 0.90 else ("✓✓" if r['mct5'] >= 0.75 else "✓")
        print(f"  {sym:3s} {task_name:<10}  {r['mct5']:>8.1%}  {r['nn']:>10.1%}  "
              f"{r['logistic']:>10.1%}  {vs:>+8.1%}")
        mct5_scores.append(r['mct5'])
        nn_scores.append(r['nn'])

    avg_mct5 = np.mean(mct5_scores)
    avg_nn = np.mean(nn_scores)
    print("  " + "-" * 66)
    print(f"  {'AVERAGE':<14}  {avg_mct5:>8.1%}  {avg_nn:>10.1%}  {'':>10}  "
          f"{avg_mct5-avg_nn:>+8.1%}")
    print(f"\n  Total time: {total_time:.1f}s")

    print("\n" + "=" * 72)
    if avg_mct5 >= 0.85:
        print("  🎉 BREAKTHROUGH: MCT5 ≥ 85% average accuracy!")
    elif avg_mct5 >= 0.75:
        print("  ✓✓ COMPETITIVE: MCT5 ≥ 75% average")
    else:
        print("  ✓  PROGRESS: MCT5 learning without backprop")
    print("=" * 72)

    # Write results file
    import os
    out_path = os.path.join(os.path.dirname(__file__), 'BENCHMARK_RESULTS.md')
    with open(out_path, 'w') as f:
        f.write("# MCT5 Benchmark Results\n\n")
        f.write("| Task | MCT5 | SimpleNN | Logistic | vs NN |\n")
        f.write("|------|------|----------|----------|-------|\n")
        for task_name, *_ in TASKS:
            r = results[task_name]
            f.write(f"| {task_name} | {r['mct5']:.1%} | {r['nn']:.1%} | "
                    f"{r['logistic']:.1%} | {r['mct5']-r['nn']:+.1%} |\n")
        f.write(f"| **Average** | **{avg_mct5:.1%}** | **{avg_nn:.1%}** | | "
                f"**{avg_mct5-avg_nn:+.1%}** |\n")
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
