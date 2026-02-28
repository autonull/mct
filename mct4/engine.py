"""
MCT4 Engine

Main engine class integrating all phases of MCT4 execution.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .core import GraphState, Node, NodeType, Context
from .primitives import Primitive
from .forward import ForwardExecutor
from .learning import LearningEngine
from .structural import StructuralEvolution


@dataclass
class MCT4Config:
    """Configuration for MCT4 engine."""
    # Vector dimensionality
    D: int = 512
    
    # Execution
    t_budget: int = 20
    lambda_tau: float = 0.1
    lambda_async: float = 0.2
    decay_c: float = 0.95
    
    # Learning
    eta: float = 0.001  # Learning rate
    alpha: float = 0.01  # Health reward (catalysis)
    beta: float = 0.05  # Health penalty (solvent)
    gamma: float = 0.001  # Atrophy rate
    W_max: float = None  # Max weight norm (default: sqrt(D))
    
    # Structural evolution
    sigma_mut: float = 0.05  # Mutation noise
    K: int = 2  # Spawn count per pruning
    F: int = 2  # Fork fan-out
    tau_lateral: float = 0.3  # Lateral wiring threshold
    kappa_thresh: int = 100  # Convergence threshold
    
    # Batch
    N: int = 32  # Batch size
    
    # Low-rank factorization
    use_factored: bool = False
    rank: int = 64
    
    def __post_init__(self):
        if self.W_max is None:
            self.W_max = np.sqrt(self.D)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    node_count_history: List[int] = field(default_factory=list)
    kappa_history: List[int] = field(default_factory=list)
    pruning_events: int = 0
    total_forward_passes: int = 0


class MCT4:
    """
    Morphogenic Compute Topology v4.0 Engine.
    
    A self-structuring, continuously-learning compute graph.
    """
    
    def __init__(self, config: Optional[MCT4Config] = None):
        self.config = config or MCT4Config()
        
        # Initialize state
        self.state = GraphState(
            D=self.config.D,
            t_budget=self.config.t_budget,
            lambda_tau=self.config.lambda_tau,
            kappa_thresh=self.config.kappa_thresh,
            N=self.config.N
        )
        self.state.context = Context(D=self.config.D, decay_c=self.config.decay_c)
        
        # Initialize components
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
        
        # Metrics
        self.metrics = TrainingMetrics()
        
        # Sequence state
        self.in_sequence = False
    
    def initialize(self, primitive_hidden: Primitive = Primitive.GELU):
        """
        Initialize the minimal viable graph.
        
        Args:
            primitive_hidden: Primitive for hidden nodes
        """
        self.structural_evolution.initialize_minimal_graph(primitive_hidden)
    
    def reset_sequence(self):
        """Reset context at sequence boundary."""
        self.forward_executor.reset_context()
        self.in_sequence = False
    
    def forward(self, X: np.ndarray, batch_X: Optional[List[np.ndarray]] = None) -> Dict[int, np.ndarray]:
        """
        Execute forward pass.
        
        Args:
            X: Input vector
            batch_X: Additional batch samples
            
        Returns:
            Dictionary of output node IDs to outputs
        """
        return self.forward_executor.execute(X, batch_X)
    
    def learn(self, Y_star: np.ndarray) -> float:
        """
        Execute learning phase after forward pass.
        
        Args:
            Y_star: Target output
            
        Returns:
            Loss value
        """
        outputs = {r.node_id: r.V_out for r in self.state.active_path 
                   if self.state.nodes[r.node_id].node_type == NodeType.OUTPUT}
        
        if not outputs:
            # No output fired - return high loss but don't crash
            return 1.0
        
        # Get first output node's result
        output_id = list(outputs.keys())[0]
        Y = outputs[output_id]
        
        # Compute loss (MSE)
        loss = float(np.mean((Y - Y_star) ** 2))
        
        # Execute learning
        self.learning_engine.learn(Y, Y_star, output_id)
        
        return loss
    
    def evolve(self):
        """Execute structural evolution phase."""
        # Prune
        pruned = self.structural_evolution.prune()
        self.metrics.pruning_events += len(pruned)
        
        # Get max tension edge
        max_edge = self.learning_engine.get_max_tension_edge()
        
        # Insert capacity and lateral wiring
        self.structural_evolution.evolve(pruned, max_edge)
        
        # Update convergence counter
        self.state.increment_kappa()
    
    def train_step(self, X: np.ndarray, Y_star: np.ndarray, 
                   evolve: bool = True, reset_context: bool = False) -> float:
        """
        Execute a complete training step.
        
        Args:
            X: Input vector
            Y_star: Target output
            evolve: Whether to run structural evolution
            reset_context: Whether to reset context (sequence boundary)
            
        Returns:
            Loss value
        """
        if reset_context:
            self.reset_sequence()
        
        self.in_sequence = True
        
        # Forward pass
        self.forward(X)
        
        # Learning
        loss = self.learn(Y_star)
        
        # Structural evolution
        if evolve:
            self.evolve()
        
        # Update metrics
        self.metrics.total_forward_passes += 1
        self.metrics.loss_history.append(loss)
        self.metrics.node_count_history.append(len(self.state.nodes))
        self.metrics.kappa_history.append(self.state.kappa)
        
        return loss
    
    def train_batch(self, X_batch: np.ndarray, Y_batch: np.ndarray,
                    evolve: bool = True) -> float:
        """
        Train on a batch of samples.
        
        Args:
            X_batch: Input batch of shape (N, D)
            Y_batch: Target batch of shape (N, D) or (N, num_classes)
            evolve: Whether to run structural evolution after batch
            
        Returns:
            Average loss over batch
        """
        N = X_batch.shape[0]
        losses = []
        
        # Reset context for new batch
        self.reset_sequence()
        
        # Process each sample in batch
        for i in range(N):
            X = X_batch[i]
            Y_star = Y_batch[i]
            
            # Forward
            self.forward(X)
            
            # Learn
            loss = self.learn(Y_star)
            losses.append(loss)
        
        # Structural evolution after batch
        if evolve:
            self.evolve()
        
        # Update metrics
        avg_loss = float(np.mean(losses))
        self.metrics.total_forward_passes += N
        self.metrics.loss_history.append(avg_loss)
        self.metrics.node_count_history.append(len(self.state.nodes))
        self.metrics.kappa_history.append(self.state.kappa)
        
        return avg_loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make a prediction without learning.
        
        Args:
            X: Input vector
            
        Returns:
            Output vector
        """
        outputs = self.forward(X)
        
        if not outputs:
            return np.zeros(self.config.D)
        
        # Return first output node's result
        return list(outputs.values())[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current graph statistics."""
        nodes = list(self.state.nodes.values())
        
        active_nodes = sum(1 for n in nodes if n.steps_idle == 0)
        avg_health = np.mean([n.rho_base for n in nodes])
        avg_tension = np.mean([n.tension_trace for n in nodes])
        
        # Count primitives
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
        """Save model state to file."""
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
        """Load model state from file."""
        import pickle
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        
        self.state.nodes = state_dict['nodes']
        self.state.context = state_dict['context']
        self.state.next_node_id = state_dict['next_node_id']
        self.state.kappa = state_dict['kappa']
        self.metrics = state_dict.get('metrics', TrainingMetrics())
