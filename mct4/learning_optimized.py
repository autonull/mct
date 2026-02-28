"""
MCT4 Learning Engine - OPTIMIZED VERSION

Vectorized batch learning with cached computations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .core_optimized import GraphState, Node, ActivePathRecord


class LearningEngine:
    """Optimized learning engine with vectorized operations."""
    
    def __init__(self, state: GraphState, 
                 eta: float = 0.001,
                 alpha: float = 0.01,
                 beta: float = 0.05,
                 gamma: float = 0.001,
                 W_max: float = None):
        self.state = state
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.W_max = W_max if W_max is not None else np.sqrt(state.D)
        self.edge_tension_attribution: Dict[Tuple[int, int], float] = defaultdict(float)
    
    def compute_tension(self, Y: np.ndarray, Y_star: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute tension with proper softmax gradient."""
        D = len(Y)
        T_v = (Y - Y_star) / np.sqrt(D)
        mse = np.mean((Y_star - Y) ** 2)
        T_norm = min(1.0, np.sqrt(mse))
        return T_v, T_norm
    
    def retrograde_flow(self, T_v: np.ndarray, output_node_id: int) -> Dict[int, np.ndarray]:
        """Vectorized retrograde flow."""
        active_path = self.state.active_path
        if not active_path:
            return {}
        
        sorted_path = sorted(active_path, key=lambda r: -r.hop)
        path_map = {r.node_id: r for r in active_path}
        
        node_tensions: Dict[int, np.ndarray] = {}
        node_tensions[output_node_id] = T_v
        
        self.edge_tension_attribution = defaultdict(float)
        
        for record in sorted_path:
            node_id = record.node_id
            
            if node_id not in node_tensions:
                continue
            
            T_current = node_tensions[node_id]
            incoming_sources = list(record.inbox_sources.keys())
            
            if not incoming_sources:
                continue
            
            # Compute blame weights (vectorized)
            norms = np.array([np.linalg.norm(record.inbox_sources[src]) + 1e-9 
                             for src in incoming_sources])
            total_norm = norms.sum() + 1e-9
            blame_weights = norms / total_norm
            
            # Distribute tension
            for idx, src_id in enumerate(incoming_sources):
                w_blame = blame_weights[idx]
                
                if src_id in self.state.nodes:
                    attenuation = max(0.1, 1.0 - self.state.nodes[src_id].tension_trace * 0.5)
                else:
                    attenuation = 1.0
                
                T_local = T_current * w_blame * attenuation
                
                if src_id in node_tensions:
                    node_tensions[src_id] += T_local
                else:
                    node_tensions[src_id] = T_local
                
                self.edge_tension_attribution[(src_id, node_id)] = w_blame * np.linalg.norm(T_current)
        
        return node_tensions
    
    def update_weights(self, node_tensions: Dict[int, np.ndarray]):
        """Vectorized weight updates."""
        for record in self.state.active_path:
            node_id = record.node_id
            
            if node_id not in node_tensions:
                continue
            
            T_local = node_tensions[node_id]
            node = self.state.nodes[node_id]
            
            # Average V_in over batch
            V_in = record.V_in.mean(axis=0)
            
            if node.W_factored:
                node.update_weight_factored(T_local, V_in, self.eta)
            else:
                delta_W = self.eta * np.outer(T_local, V_in)
                node.update_weight_full(delta_W)
            
            node.clamp_weight_norm(self.W_max)
            node.last_attributed_tension = np.linalg.norm(T_local)
    
    def update_tension_traces(self, node_tensions: Dict[int, np.ndarray]):
        """Update tension traces."""
        for record in self.state.active_path:
            node_id = record.node_id
            node = self.state.nodes[node_id]
            
            T_norm = np.linalg.norm(node_tensions.get(node_id, 0))
            node.tension_trace = 0.9 * node.tension_trace + 0.1 * T_norm
            
            if node.tension_trace > 0.3:
                node.high_tension_count += 1
            else:
                node.high_tension_count = 0
    
    def update_health(self, T_norm: float, node_tensions: Dict[int, np.ndarray]):
        """Update health with clamping."""
        for record in self.state.active_path:
            node_id = record.node_id
            node = self.state.nodes[node_id]
            
            w_blame = 1.0
            if node_id in node_tensions and T_norm > 0:
                T_local_norm = np.linalg.norm(node_tensions[node_id])
                w_blame = min(1.0, T_local_norm / (T_norm + 1e-9))
            
            reward = self.alpha * (1.0 - T_norm)
            penalty = self.beta * (1.0 + T_norm ** 2) * T_norm * w_blame
            delta_rho = np.clip(reward - penalty, -0.1, 0.1)
            
            node.rho_base = np.clip(node.rho_base + delta_rho, -10, 10)
    
    def apply_atrophy(self):
        """Apply atrophy to idle nodes."""
        gamma = self.gamma * (0.5 if self.state.is_converged() else 1.0)
        
        for node in self.state.nodes.values():
            if node.steps_idle > 50:
                node.rho_base -= gamma * node.steps_idle
    
    def learn(self, Y: np.ndarray, Y_star: np.ndarray, output_node_id: int):
        """Execute full learning phase."""
        T_v, T_norm = self.compute_tension(Y, Y_star)
        node_tensions = self.retrograde_flow(T_v, output_node_id)
        self.update_weights(node_tensions)
        self.update_tension_traces(node_tensions)
        self.update_health(T_norm, node_tensions)
        
        for edge, tension in self.edge_tension_attribution.items():
            self.state.edge_tensions[edge] = tension
    
    def get_max_tension_edge(self) -> Optional[Tuple[int, int]]:
        if not self.edge_tension_attribution:
            return None
        return max(self.edge_tension_attribution.items(), key=lambda x: x[1])[0]
