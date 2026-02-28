"""
MCT4 Phase 2 - Learning

Local, online learning without storing a computation graph.
Retrograde error signal propagates in reverse topological order.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .core import GraphState, Node, ActivePathRecord


class LearningEngine:
    """
    Executes Phase 2 of MCT4: Learning via retrograde flow.
    
    No backward pass in the backpropagation sense - uses local learning rules
    with retrograde error signals.
    """
    
    def __init__(self, state: GraphState, 
                 eta: float = 0.001,  # Learning rate
                 alpha: float = 0.01,  # Health reward (catalysis)
                 beta: float = 0.05,   # Health penalty (solvent)
                 gamma: float = 0.001,  # Atrophy rate
                 W_max: float = None):  # Max weight norm
        self.state = state
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.W_max = W_max if W_max is not None else np.sqrt(state.D)
        
        # Track tension attribution per edge for structural evolution
        self.edge_tension_attribution: Dict[Tuple[int, int], float] = defaultdict(float)
    
    def compute_tension(self, Y: np.ndarray, Y_star: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute tension (error signal).
        
        For gradient descent, we want to move in the direction (Y* - Y), not (Y - Y*).
        T_v = (Y* - Y) / √D         ← negative gradient direction
        ||T|| = MSE(Y*, Y) ∈ [0,1]  ← scalar error magnitude
        
        Args:
            Y: Actual output (already passed through softmax for classification)
            Y_star: Target output (one-hot encoded)
            
        Returns:
            Tuple of (error direction vector, error magnitude scalar)
        """
        D = len(Y)
        
        # Negative gradient direction: we want to increase Y towards Y*
        T_v = (Y_star - Y) / np.sqrt(D)
        
        # MSE normalized to [0, 1] range
        mse = np.mean((Y_star - Y) ** 2)
        T_norm = min(1.0, np.sqrt(mse))
        
        return T_v, T_norm
    
    def retrograde_flow(self, T_v: np.ndarray, output_node_id: int) -> Dict[int, np.ndarray]:
        """
        Propagate tension signal in reverse topological order.
        
        Simplified version: each node receives error signal scaled by its weight contribution.
        """
        active_path = self.state.active_path
        if not active_path:
            return {}
        
        # Sort by hop (descending) for reverse order
        sorted_path = sorted(active_path, key=lambda r: -r.hop)
        path_map = {r.node_id: r for r in active_path}
        
        # Tension at each node
        node_tensions: Dict[int, np.ndarray] = {}
        node_tensions[output_node_id] = T_v
        
        self.edge_tension_attribution = defaultdict(float)
        
        # Propagate in reverse order
        for record in sorted_path:
            node_id = record.node_id
            
            if node_id not in node_tensions:
                continue
            
            T_current = node_tensions[node_id]
            node = self.state.nodes.get(node_id)
            
            incoming_sources = list(record.inbox_sources.keys())
            if not incoming_sources:
                continue
            
            # For each incoming source, propagate error
            for src_id in incoming_sources:
                V_from_src = record.inbox_sources[src_id]
                
                # Simple error propagation: pass T_current scaled by weight
                # This is like backprop but using local information only
                src_node = self.state.nodes.get(src_id)
                
                if src_node is not None:
                    # Scale by how much this source contributed (based on V_from_src magnitude)
                    src_norm = np.linalg.norm(V_from_src) + 1e-9
                    
                    # Pass full signal (don't divide by total)
                    # This ensures error signal doesn't vanish
                    T_local = T_current * 0.5  # Simple scaling
                    
                    if src_id in node_tensions:
                        node_tensions[src_id] += T_local
                    else:
                        node_tensions[src_id] = T_local
                    
                    self.edge_tension_attribution[(src_id, node_id)] = np.linalg.norm(T_local)
        
        return node_tensions
    
    def update_weights(self, node_tensions: Dict[int, np.ndarray]):
        """
        Update weight matrices for all active nodes.
        
        ΔW = η · T_local,i ⊗ V_in,i      ← rank-1 update
        W_i ← W_i + ΔW
        
        For low-rank factored mode:
        A ← A + η · T_local,i ⊗ (Bᵀ V_in,i)
        B ← B + η · V_in,i ⊗ (Aᵀ T_local,i)
        """
        for record in self.state.active_path:
            node_id = record.node_id
            
            if node_id not in node_tensions:
                continue
            
            T_local = node_tensions[node_id]
            node = self.state.nodes[node_id]
            V_in = record.V_in
            
            # Update weights
            if node.W_factored:
                node.update_weight_factored(T_local, V_in, self.eta)
            else:
                # Rank-1 outer product update
                delta_W = self.eta * np.outer(T_local, V_in)
                node.update_weight_full(delta_W)
            
            # Clamp weight norm
            node.clamp_weight_norm(self.W_max)
            
            # Store attributed tension for structural evolution
            node.last_attributed_tension = np.linalg.norm(T_local)
    
    def update_tension_traces(self, node_tensions: Dict[int, np.ndarray]):
        """
        Update tension trace EMA for all active nodes.
        
        tension_trace_i ← 0.9 · tension_trace_i + 0.1 · ||T_local,i||
        """
        for record in self.state.active_path:
            node_id = record.node_id
            node = self.state.nodes[node_id]
            
            if node_id in node_tensions:
                T_norm = np.linalg.norm(node_tensions[node_id])
            else:
                T_norm = 0.0
            
            node.tension_trace = 0.9 * node.tension_trace + 0.1 * T_norm
            
            # Track consecutive high tension for lateral wiring
            if node.tension_trace > 0.3:
                node.high_tension_count += 1
            else:
                node.high_tension_count = 0
    
    def update_health(self, T_norm: float, node_tensions: Dict[int, np.ndarray]):
        """
        Update health (rho_base) for all active nodes.
        
        Δρ = α · (1 − ||T||) − β · (1 + ||T||²) · ||T|| · w_blame,i
        ρ_base,i ← ρ_base,i + Δρ
        
        Health and weight updates are decoupled.
        """
        for record in self.state.active_path:
            node_id = record.node_id
            node = self.state.nodes[node_id]
            
            # Compute blame weight for this node
            w_blame = 1.0
            if node_id in node_tensions and T_norm > 0:
                T_local_norm = np.linalg.norm(node_tensions[node_id])
                w_blame = min(1.0, T_local_norm / (T_norm + 1e-9))
            
            # Health update with dampening
            reward = self.alpha * (1.0 - T_norm)
            penalty = self.beta * (1.0 + T_norm ** 2) * T_norm * w_blame
            delta_rho = reward - penalty
            
            # Clamp delta to prevent explosion
            delta_rho = np.clip(delta_rho, -0.1, 0.1)
            
            node.rho_base += delta_rho
            
            # Clamp health to reasonable range
            node.rho_base = np.clip(node.rho_base, -10, 10)
    
    def apply_atrophy(self):
        """
        Apply atrophy to all nodes.
        
        steps_idle ← steps_idle + 1
        if steps_idle > 50:
            ρ_base,i ← ρ_base,i − γ · steps_idle
        """
        # Adjust gamma based on convergence state
        gamma = self.gamma
        if self.state.is_converged():
            gamma *= 0.5
        
        for node in self.state.nodes.values():
            if node.steps_idle > 50:
                node.rho_base -= gamma * node.steps_idle
    
    def learn(self, Y: np.ndarray, Y_star: np.ndarray, output_node_id: int):
        """
        Execute full learning phase.
        
        Args:
            Y: Actual output from forward pass
            Y_star: Target output
            output_node_id: ID of output node that fired
        """
        # Step 1: Compute tension
        T_v, T_norm = self.compute_tension(Y, Y_star)
        
        # Step 2: Retrograde flow
        node_tensions = self.retrograde_flow(T_v, output_node_id)
        
        # Step 3: Update weights
        self.update_weights(node_tensions)
        
        # Step 4: Update tension traces
        self.update_tension_traces(node_tensions)
        
        # Step 5: Update health
        self.update_health(T_norm, node_tensions)
        
        # Store edge tensions in state for structural evolution
        for edge, tension in self.edge_tension_attribution.items():
            self.state.edge_tensions[edge] = tension
    
    def get_max_tension_edge(self) -> Optional[Tuple[int, int]]:
        """
        Get the edge with highest attributed tension from last pass.
        
        Used for capacity insertion.
        """
        if not self.edge_tension_attribution:
            return None
        
        return max(self.edge_tension_attribution.items(), key=lambda x: x[1])[0]
