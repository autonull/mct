"""
MCT4 Engine - OPTIMIZED VERSION

High-performance MCT4 with vectorized operations.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .core_optimized import GraphState, NodeType, Context
from .primitives import Primitive
from .forward_optimized import ForwardExecutor
from .learning_optimized import LearningEngine
from .structural import StructuralEvolution  # Use existing structural


@dataclass
class MCT4Config:
    """Configuration."""
    D: int = 512
    t_budget: int = 20
    lambda_tau: float = 0.1
    lambda_async: float = 0.2
    decay_c: float = 0.95
    eta: float = 0.001
    alpha: float = 0.01
    beta: float = 0.05
    gamma: float = 0.001
    W_max: float = None
    sigma_mut: float = 0.05
    K: int = 2
    F: int = 2
    tau_lateral: float = 0.3
    kappa_thresh: int = 100
    N: int = 32
    use_factored: bool = False
    rank: int = 64
    
    def __post_init__(self):
        if self.W_max is None:
            self.W_max = np.sqrt(self.D)


@dataclass
class TrainingMetrics:
    """Training metrics."""
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    node_count_history: List[int] = field(default_factory=list)
    kappa_history: List[int] = field(default_factory=list)
    pruning_events: int = 0
    total_forward_passes: int = 0


class MCT4:
    """Optimized MCT4 engine."""
    
    def __init__(self, config: Optional[MCT4Config] = None):
        self.config = config or MCT4Config()
        
        self.state = GraphState(
            D=self.config.D,
            t_budget=self.config.t_budget,
            lambda_tau=self.config.lambda_tau,
            kappa_thresh=self.config.kappa_thresh,
            N=self.config.N
        )
        self.state.context = Context(D=self.config.D, decay_c=self.config.decay_c)
        
        self.forward_executor = ForwardExecutor(
            self.state, 
            lambda_async=self.config.lambda_async
        )
        
        self.learning_engine = LearningEngine(
            self.state,
            eta=self.config.eta,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            W_max=self.config.W_max
        )
        
        self.structural_evolution = StructuralEvolution(
            self.state,
            sigma_mut=self.config.sigma_mut,
            K=self.config.K,
            F=self.config.F,
            tau_lateral=self.config.tau_lateral,
            rank=self.config.rank
        )
        
        self.metrics = TrainingMetrics()
        self.in_sequence = False
    
    def initialize(self, primitive_hidden: Primitive = Primitive.GELU):
        self.structural_evolution.initialize_minimal_graph(primitive_hidden)
    
    def reset_sequence(self):
        self.forward_executor.reset_context()
        self.in_sequence = False
    
    def forward(self, X: np.ndarray, batch_X: Optional[List[np.ndarray]] = None) -> Dict[int, np.ndarray]:
        return self.forward_executor.execute(X, batch_X)
    
    def learn(self, Y_star: np.ndarray) -> float:
        outputs = {r.node_id: r.V_out for r in self.state.active_path 
                   if self.state.nodes[r.node_id].node_type == NodeType.OUTPUT}
        
        if not outputs:
            return 1.0
        
        output_id = list(outputs.keys())[0]
        Y = outputs[output_id]
        loss = float(np.mean((Y - Y_star) ** 2))
        
        self.learning_engine.learn(Y, Y_star, output_id)
        return loss
    
    def evolve(self):
        pruned = self.structural_evolution.prune()
        self.metrics.pruning_events += len(pruned)
        max_edge = self.learning_engine.get_max_tension_edge()
        self.structural_evolution.evolve(pruned, max_edge)
        self.state.increment_kappa()
    
    def train_step(self, X: np.ndarray, Y_star: np.ndarray, 
                   evolve: bool = True, reset_context: bool = False) -> float:
        if reset_context:
            self.reset_sequence()
        
        self.in_sequence = True
        self.forward(X)
        loss = self.learn(Y_star)
        
        if evolve:
            self.evolve()
        
        self.metrics.total_forward_passes += 1
        self.metrics.loss_history.append(loss)
        self.metrics.node_count_history.append(len(self.state.nodes))
        self.metrics.kappa_history.append(self.state.kappa)
        
        return loss
    
    def train_batch(self, X_batch: np.ndarray, Y_batch: np.ndarray,
                    evolve: bool = True) -> float:
        N = X_batch.shape[0]
        losses = []
        
        self.reset_sequence()
        
        for i in range(N):
            X = X_batch[i]
            Y_star = Y_batch[i]
            self.forward(X)
            loss = self.learn(Y_star)
            losses.append(loss)
        
        if evolve:
            self.evolve()
        
        avg_loss = float(np.mean(losses))
        self.metrics.total_forward_passes += N
        self.metrics.loss_history.append(avg_loss)
        self.metrics.node_count_history.append(len(self.state.nodes))
        self.metrics.kappa_history.append(self.state.kappa)
        
        return avg_loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        outputs = self.forward(X)
        if not outputs:
            return np.zeros(self.config.D)
        return list(outputs.values())[0]
    
    def get_stats(self) -> Dict[str, Any]:
        nodes = list(self.state.nodes.values())
        
        active_nodes = sum(1 for n in nodes if n.steps_idle == 0)
        avg_health = np.mean([n.rho_base for n in nodes]) if nodes else 0
        avg_tension = np.mean([n.tension_trace for n in nodes]) if nodes else 0
        
        primitive_counts = {}
        for n in nodes:
            p_name = n.primitive.name
            primitive_counts[p_name] = primitive_counts.get(p_name, 0) + 1
        
        return {
            'total_nodes': len(nodes),
            'active_nodes': active_nodes,
            'input_nodes': sum(1 for n in nodes if n.node_type == NodeType.INPUT),
            'hidden_nodes': sum(1 for n in nodes if n.node_type == NodeType.HIDDEN),
            'output_nodes': sum(1 for n in nodes if n.node_type == NodeType.OUTPUT),
            'avg_health': avg_health,
            'avg_tension': avg_tension,
            'primitives': primitive_counts,
            'kappa': self.state.kappa,
            'is_converged': self.state.is_converged(),
            'total_edges': sum(len(n.edges_out) for n in nodes),
        }
    
    def save_state(self, path: str):
        import pickle
        state_dict = {
            'nodes': self.state.nodes,
            'context': self.state.context,
            'next_node_id': self.state.next_node_id,
            'kappa': self.state.kappa,
            'metrics': self.metrics,
            'config': self.config,
        }
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)
    
    def load_state(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        
        self.state.nodes = state_dict['nodes']
        self.state.context = state_dict['context']
        self.state.next_node_id = state_dict['next_node_id']
        self.state.kappa = state_dict['kappa']
        self.metrics = state_dict.get('metrics', TrainingMetrics())
