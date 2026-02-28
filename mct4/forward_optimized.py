"""
MCT4 Forward Execution - OPTIMIZED VERSION

Vectorized batch processing with efficient data structures.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import heapq

from .core_optimized import GraphState, Node, ActivePathRecord, NodeType
from .primitives import apply_primitive


class ForwardExecutor:
    """Optimized forward executor with vectorized batch support."""
    
    def __init__(self, state: GraphState, lambda_async: float = 0.2):
        self.state = state
        self.lambda_async = lambda_async
        self.current_t = 0
        self.fired_this_pass: Dict[int, ActivePathRecord] = {}
        
    def reset_context(self):
        self.state.context.reset()
    
    def execute(self, X: np.ndarray, batch_X: Optional[List[np.ndarray]] = None) -> Dict[int, np.ndarray]:
        """
        Execute forward pass with vectorized batch processing.
        
        Args:
            X: Primary input vector (D,)
            batch_X: Additional batch samples [(D,), ...]
            
        Returns:
            Dictionary of output node IDs to output vectors (D,)
        """
        # Reset state
        self.current_t = 0
        self.fired_this_pass = {}
        self.state.active_path = []
        self.state.pruning_events_this_pass = 0
        
        # Prepare batch as (N, D) array
        if batch_X is not None:
            batch_inputs = np.vstack([X] + batch_X)
        else:
            batch_inputs = X.reshape(1, -1)
        
        N = len(batch_inputs)

        # Clear inboxes and reset state
        for node in self.state.nodes.values():
            node.clear_inbox()
            node.last_hop = -1

        # Get input nodes and fire them
        input_nodes = self.state.get_input_nodes()
        processed_at_hop = {}
        ready_queue = []
        
        for node in input_nodes:
            processed_at_hop[node.id] = 0
            node.last_hop = 0
            node.steps_idle = 0
            
            # Input nodes broadcast input to all downstream
            for dst_id in node.edges_out:
                if dst_id in self.state.nodes:
                    self.state.nodes[dst_id].add_to_inbox(node.id, batch_inputs.copy(), 0)
                    if dst_id not in processed_at_hop:
                        heapq.heappush(ready_queue, (1, dst_id))
        
        outputs = {}
        output_fired = False
        self.current_t = 1
        
        while ready_queue and not output_fired and self.current_t <= self.state.t_budget:
            # Get nodes at current hop
            current_hop_nodes = []
            while ready_queue and ready_queue[0][0] == self.current_t:
                hop, node_id = heapq.heappop(ready_queue)
                if node_id in self.state.nodes and node_id not in processed_at_hop:
                    current_hop_nodes.append(node_id)
                    processed_at_hop[node_id] = self.current_t
            
            if not current_hop_nodes:
                if ready_queue and ready_queue[0][0] > self.current_t:
                    self.current_t = ready_queue[0][0]
                    continue
                self.current_t += 1
                continue
            
            tau = self.state.get_tau(self.current_t)
            
            for node_id in current_hop_nodes:
                node = self.state.nodes[node_id]
                
                # Aggregate inbox (vectorized)
                V_batch = node.aggregate_inbox(self.current_t, self.lambda_async, N)
                
                if len(V_batch) == 0 or np.allclose(V_batch, 0):
                    node.steps_idle += 1
                    # Still propagate to downstream
                    for dst_id in node.edges_out:
                        if dst_id in self.state.nodes and dst_id not in processed_at_hop:
                            next_hop = self.current_t + 1
                            if next_hop <= self.state.t_budget:
                                heapq.heappush(ready_queue, (next_hop, dst_id))
                    continue
                
                # Compute activation potential (vectorized)
                # ρ = ρ_base + S·X + S·C
                dot_SX = np.dot(node.S, V_batch[0])  # Use first sample
                dot_SC = np.dot(node.S, self.state.context.C)
                rho = node.rho_base + dot_SX + dot_SC
                
                if rho < tau:
                    # Ghost signal
                    self.state.context.add_ghost_batch(np.array([rho]), node.S)
                    node.steps_idle += 1
                    continue
                
                # Node fires!
                node.last_hop = self.current_t
                
                # Apply weight and primitive (vectorized)
                V_weighted = node.apply_weight_batch(V_batch)
                V_out = np.array([apply_primitive(node.primitive, v) for v in V_weighted])
                
                # Route to downstream
                for dst_id in node.edges_out:
                    if dst_id in self.state.nodes:
                        self.state.nodes[dst_id].add_to_inbox(node.id, V_out, self.current_t)
                        if dst_id not in processed_at_hop:
                            next_hop = self.current_t + 1
                            if next_hop <= self.state.t_budget:
                                heapq.heappush(ready_queue, (next_hop, dst_id))
                
                # Record for learning
                inbox_sources = {}
                for sender in node.inbox_senders:
                    mask = np.array(node.inbox_senders) == sender
                    if np.any(mask):
                        inbox_sources[sender] = node.inbox_vectors[mask].mean(axis=0)
                
                record = ActivePathRecord(
                    node_id=node_id,
                    V_in=V_batch,
                    V_out=V_out,
                    V_weighted=V_weighted,
                    hop=self.current_t,
                    inbox_sources=inbox_sources
                )
                self.fired_this_pass[node_id] = record
                self.state.active_path.append(record)
                
                node.steps_idle = 0
                
                # Check if output
                if node.node_type == NodeType.OUTPUT:
                    # Return mean output for loss computation
                    outputs[node_id] = V_out.mean(axis=0)
                    output_fired = True
                    break
            
            if not output_fired:
                self.current_t += 1
        
        # Update idle counters
        for node in self.state.nodes.values():
            if node.id not in self.fired_this_pass:
                node.steps_idle += 1
        
        return outputs
    
    def get_active_path(self) -> List[ActivePathRecord]:
        return self.state.active_path
