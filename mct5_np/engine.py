"""
MCT5 Engine — Top-Level API

Integrates forward, learning, and structural evolution into a clean interface.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .config import MCT5Config
from .types import GraphState, NodeType, ActiveRecord
from .primitives import Primitive
from .residue import HolographicResidue
from .forward import ForwardExecutor
from .learning import LearningEngine
from .structural import StructuralEvolution


@dataclass
class MCT5Metrics:
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    node_count_history: List[int] = field(default_factory=list)
    pruning_events: int = 0
    total_steps: int = 0


class MCT5:
    """
    Morphogenic Compute Topology v5.

    Example:
        config = MCT5Config(D=64, r=16, n_classes=10, input_dim=32)
        model = MCT5(config)
        model.initialize()

        for epoch in range(100):
            for x, y in data:
                loss = model.train_step(x, y)

        preds = [model.predict(x) for x in X_test]
    """

    def __init__(self, config: Optional[MCT5Config] = None):
        self.cfg = config or MCT5Config()
        cfg = self.cfg

        # Graph state
        self.state = GraphState(
            D=cfg.D, r=cfg.r,
            t_budget=cfg.t_budget,
            lambda_tau=cfg.lambda_tau,
            kappa_thresh=cfg.kappa_thresh,
        )

        # Holographic Residue
        self.residue = HolographicResidue(
            D=cfg.D,
            max_nodes=cfg.D + 128,
            omega=cfg.omega,
            phi_max=cfg.phi_max,
            decay=cfg.residue_decay,
        )

        # Phase executors
        self.forward_exec = ForwardExecutor(
            self.state, self.residue, lambda_async=cfg.lambda_async
        )
        self.learner = LearningEngine(
            self.state, self.residue,
            eta_W=cfg.eta_W, eta_S=cfg.eta_S,
            alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
            W_max=cfg.W_max,
            adam_beta1=cfg.adam_beta1, adam_beta2=cfg.adam_beta2, adam_eps=cfg.adam_eps,
            lambda_contrastive=cfg.lambda_contrastive,
            lambda_retrograde=cfg.lambda_retrograde,
            goodness_threshold=cfg.goodness_threshold,
        )
        self.evolver = StructuralEvolution(
            self.state, self.residue,
            sigma_mut=cfg.sigma_mut, K=cfg.K,
            tau_lateral=cfg.tau_lateral,
            quadratic_spawn_bias=cfg.quadratic_spawn_bias,
        )

        self.metrics = MCT5Metrics()
        self._ema_loss_prev: float = 1.0
        self._steps_since_evolve: int = 0

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, primitive_hidden: Primitive = Primitive.GELU):
        """Build the minimal starter DAG."""
        self.evolver.initialize_graph(primitive_hidden)

    # ── Core training ─────────────────────────────────────────────────────────

    def train_step(self, X: np.ndarray, y: int, reset_context: bool = False) -> float:
        """
        One complete train step (forward → learn → optional evolve).

        Args:
            X: Input feature vector (will be projected/padded to D)
            y: Integer class label
            reset_context: Reset Holographic Residue before this sample

        Returns:
            Scalar loss
        """
        cfg = self.cfg
        if reset_context:
            self.forward_exec.reset_context()

        X_d = self._embed(X)
        Y_star = self._one_hot(y)

        # ── Positive forward pass ─────────────────────────────────────────────
        outputs = self.forward_exec.execute(X_d)
        if not outputs:
            return 1.0

        output_id, Y_hat = next(iter(outputs.items()))
        loss = self.learner.learn(Y_hat, Y_star, output_id, is_positive=True)

        # ── Negative forward pass (hard negative: wrong label) ────────────────
        neg_label = self._hard_negative(y, cfg.n_classes)
        Y_neg = self._one_hot(neg_label)

        neg_outputs = self.forward_exec.execute(X_d)
        if neg_outputs:
            neg_id, Y_hat_neg = next(iter(neg_outputs.items()))
            self.learner.learn(Y_hat_neg, Y_neg, neg_id, is_positive=False)

        # ── Structural evolution ──────────────────────────────────────────────
        self._steps_since_evolve += 1
        if self._steps_since_evolve >= cfg.evolve_interval:
            self._run_evolution()
            self._steps_since_evolve = 0

        # ── Metrics ───────────────────────────────────────────────────────────
        self.state.tick()
        self.metrics.total_steps += 1
        self.metrics.loss_history.append(loss)
        self.metrics.node_count_history.append(len(self.state.nodes))

        return loss

    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray,
                    reset_context: bool = True) -> float:
        """Train on a batch; returns mean loss."""
        if reset_context:
            self.forward_exec.reset_context()
        losses = []
        for x, y in zip(X_batch, y_batch):
            loss = self.train_step(x, int(y), reset_context=False)
            losses.append(loss)
        return float(np.mean(losses))

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> int:
        """Return integer class prediction."""
        probs = self.predict_proba(X)
        return int(np.argmax(probs[: self.cfg.n_classes]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probability vector (length D; first n_classes are class probs)."""
        X_d = self._embed(X)
        outputs = self.forward_exec.execute(X_d)
        if not outputs:
            return np.ones(self.cfg.D) / self.cfg.D
        return next(iter(outputs.values()))

    def score(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Accuracy on a batch."""
        correct = sum(
            1 for x, y in zip(X_batch, y_batch)
            if self.predict(x) == int(y)
        )
        return correct / len(y_batch)

    # ── Context control ───────────────────────────────────────────────────────

    def reset_sequence(self):
        """Reset Holographic Residue (call at sequence / episode boundaries)."""
        self.forward_exec.reset_context()

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        nodes = list(self.state.nodes.values())
        prim_counts: Dict[str, int] = {}
        for n in nodes:
            prim_counts[n.primitive.name] = prim_counts.get(n.primitive.name, 0) + 1
        return {
            'total_nodes': len(nodes),
            'input': sum(1 for n in nodes if n.node_type == NodeType.INPUT),
            'hidden': sum(1 for n in nodes if n.node_type == NodeType.HIDDEN),
            'output': sum(1 for n in nodes if n.node_type == NodeType.OUTPUT),
            'avg_health': float(np.mean([n.rho for n in nodes])),
            'avg_tension': float(np.mean([n.tension_trace for n in nodes])),
            'total_edges': sum(len(v) for v in self.state.edges_out.values()),
            'kappa': self.state.kappa,
            'converged': self.state.is_converged(),
            'primitives': prim_counts,
            'ema_loss': self.learner.ema_loss(),
            'steps': self.metrics.total_steps,
            'pruning_events': self.metrics.pruning_events,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'cfg': self.cfg,
                'nodes': self.state.nodes,
                'edges_out': self.state.edges_out,
                'edges_in': self.state.edges_in,
                'next_id': self.state.next_id,
                'kappa': self.state.kappa,
                'metrics': self.metrics,
            }, f)

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.state.nodes = d['nodes']
        self.state.edges_out = d['edges_out']
        self.state.edges_in = d['edges_in']
        self.state.next_id = d['next_id']
        self.state.kappa = d['kappa']
        self.metrics = d.get('metrics', MCT5Metrics())
        self.state._topo_valid = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed(self, X: np.ndarray) -> np.ndarray:
        """Pad or truncate X to dimension D."""
        D = self.cfg.D
        if len(X) >= D:
            return X[:D].copy()
        out = np.zeros(D)
        out[:len(X)] = X
        return out

    def _one_hot(self, label: int) -> np.ndarray:
        """One-hot target of length D (first n_classes positions used)."""
        t = np.zeros(self.cfg.D)
        t[label % self.cfg.n_classes] = 1.0
        return t

    @staticmethod
    def _hard_negative(y: int, n_classes: int) -> int:
        """Sample a wrong label uniformly (excluding y)."""
        if n_classes <= 1:
            return y
        choices = [c for c in range(n_classes) if c != y]
        return int(np.random.choice(choices))

    def _run_evolution(self):
        """Run the structural evolution phase."""
        pruned = self.evolver.prune()
        self.metrics.pruning_events += len(pruned)
        max_edge = self.learner.max_tension_edge()
        self.evolver.evolve(
            pruned=pruned,
            max_tension_edge=max_edge,
            ema_loss=self.learner.ema_loss(),
            ema_loss_prev=self._ema_loss_prev,
        )
        self._ema_loss_prev = self.learner.ema_loss()
        self.state.pruning_events_this_pass = len(pruned)
