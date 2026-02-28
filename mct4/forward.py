"""
MCT4 Phase 1 - Forward Execution

Asynchronous execution via priority queue ordered by hop count.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import heapq

from .core import GraphState, Node, ActivePathRecord, NodeType
from .primitives import apply_primitive


class ForwardExecutor:
    """
    Executes Phase 1 of MCT4: Forward pass through the graph.
    
    Nodes fire asynchronously based on activation potential and dynamic threshold.
    """
    
    def __init__(self, state: GraphState, lambda_async: float = 0.2):
        self.state = state
        self.lambda_async = lambda_async  # Inbox signal decay rate
        
        # Execution state
        self.current_t = 0  # Current hop count
        self.fired_this_pass: Dict[int, ActivePathRecord] = {}
        
    def reset_context(self):
        """Reset context vector at sequence boundary."""
        self.state.context.reset()
    
    def execute(self, X: np.ndarray, batch_X: Optional[List[np.ndarray]] = None) -> Dict[int, np.ndarray]:
        """
        Execute forward pass through the graph.
        
        Args:
            X: Primary input vector (or first batch sample)
            batch_X: Additional batch samples for parallel execution
            
        Returns:
            Dictionary of output node IDs to their output vectors
        """
        # Reset execution state
        self.current_t = 0
        self.fired_this_pass = {}
        self.state.active_path = []
        self.state.pruning_events_this_pass = 0
        
        # Prepare batch inputs
        if batch_X is not None:
            batch_inputs = [X] + batch_X
        else:
            batch_inputs = [X]
        
        N = len(batch_inputs)
        
        # Clear inboxes and reset idle counters
        for node in self.state.nodes.values():
            node.inbox = defaultdict(list)
            node.batch_outputs = []
            node.last_hop = -1
        
        # Inject input into input nodes - input nodes always fire with external input
        input_nodes = self.state.get_input_nodes()
        for node in input_nodes:
            for batch_idx, inp in enumerate(batch_inputs):
                # Input nodes receive external input directly
                node.inbox[node.id].append((inp.copy(), 0))
        
        # Priority queue: (hop_count, node_id)
        # Nodes are processed in hop order
        ready_queue = []
        processed_at_hop: Dict[int, int] = {}  # node_id -> hop when processed
        
        # Initialize with input nodes at hop 0 - input nodes always fire
        for node in input_nodes:
            # Don't add to queue - process immediately
            processed_at_hop[node.id] = 0
            
            # Process input nodes immediately
            node.last_hop = 0
            node.steps_idle = 0
            
            for batch_idx, inp in enumerate(batch_inputs):
                # Input nodes pass through their input
                V_out = inp.copy()
                node.batch_outputs.append(V_out)
                
                # Route to outgoing edges
                for dst_id in node.edges_out:
                    if dst_id in self.state.nodes:
                        self.state.nodes[dst_id].inbox[node.id].append((V_out.copy(), 0))
                        # Add downstream node to queue at hop 1
                        if dst_id not in processed_at_hop:
                            heapq.heappush(ready_queue, (1, dst_id))
        
        outputs = {}
        output_fired = False
        
        self.current_t = 1  # Start from hop 1 for hidden nodes
        
        while ready_queue and not output_fired and self.current_t <= self.state.t_budget:
            # Get all nodes at current hop level
            current_hop_nodes = []
            while ready_queue and ready_queue[0][0] == self.current_t:
                hop, node_id = heapq.heappop(ready_queue)
                if node_id in self.state.nodes and node_id not in processed_at_hop:
                    current_hop_nodes.append(node_id)
                    processed_at_hop[node_id] = self.current_t  # Mark as being processed
            
            if not current_hop_nodes:
                # Check if we have nodes waiting at higher hops
                if ready_queue and ready_queue[0][0] > self.current_t:
                    self.current_t = ready_queue[0][0]
                    continue
                self.current_t += 1
                continue
            
            # Process nodes at this hop
            tau = self.state.get_tau(self.current_t)
            
            for node_id in current_hop_nodes:
                node = self.state.nodes[node_id]
                
                # Compute activation potential
                V_batch = self._aggregate_inbox(node, N)
                
                if len(V_batch) == 0:
                    node.steps_idle += 1
                    # Still add downstream nodes if this one has edges
                    for dst_id in node.edges_out:
                        if dst_id not in processed_at_hop and dst_id in self.state.nodes:
                            next_hop = self.current_t + 1
                            if next_hop <= self.state.t_budget:
                                heapq.heappush(ready_queue, (next_hop, dst_id))
                    continue
                
                # Use first batch sample for activation decision
                V_in = V_batch[0]
                
                # ρᵢ = ρ_base,i + dot(Sᵢ, X) + dot(Sᵢ, C)
                rho = node.rho_base + np.dot(node.S, V_in) + np.dot(node.S, self.state.context.C)
                
                # Check activation threshold
                if rho < tau:
                    # Node fails to fire - contribute ghost signal
                    self.state.context.add_ghost(rho, node.S)
                    node.steps_idle += 1
                    continue
                
                # Node fires!
                processed_at_hop[node_id] = self.current_t
                node.last_hop = self.current_t
                
                # Process all batch samples
                V_out_batch = []
                V_in_batch = []
                
                for batch_idx in range(N):
                    V_in_b = V_batch[batch_idx]
                    V_in_batch.append(V_in_b)
                    
                    # Apply weight matrix
                    V_weighted = node.apply_weight(V_in_b)
                    
                    # Apply primitive
                    V_out = apply_primitive(node.primitive, V_weighted)
                    V_out_batch.append(V_out)
                    
                    # Route to outgoing edges
                    for dst_id in node.edges_out:
                        if dst_id in self.state.nodes:
                            self.state.nodes[dst_id].inbox[node_id].append((V_out.copy(), self.current_t))
                
                # Record for learning phase
                avg_V_in = np.mean(V_in_batch, axis=0)
                avg_V_out = np.mean(V_out_batch, axis=0)
                V_weighted_avg = node.apply_weight(avg_V_in)
                
                inbox_sources = {}
                for src_id, messages in node.inbox.items():
                    if messages:
                        inbox_sources[src_id] = messages[0][0]
                
                record = ActivePathRecord(
                    node_id=node_id,
                    V_in=avg_V_in,
                    V_out=avg_V_out,
                    V_weighted=V_weighted_avg,
                    hop=self.current_t,
                    inbox_sources=inbox_sources
                )
                self.fired_this_pass[node_id] = record
                self.state.active_path.append(record)
                
                node.batch_outputs = V_out_batch
                node.steps_idle = 0
                
                # Check if output node fired
                if node.node_type == NodeType.OUTPUT:
                    outputs[node_id] = avg_V_out
                    output_fired = True
                    break  # Halt when output fires
            
            # Add downstream nodes to queue
            for node_id in current_hop_nodes:
                if node_id in self.state.nodes:
                    node = self.state.nodes[node_id]
                    for dst_id in node.edges_out:
                        if dst_id not in processed_at_hop and dst_id in self.state.nodes:
                            next_hop = self.current_t + 1
                            if next_hop <= self.state.t_budget:
                                heapq.heappush(ready_queue, (next_hop, dst_id))
            
            if not output_fired:
                self.current_t += 1
        
        # Update idle counters for nodes that didn't fire
        for node in self.state.nodes.values():
            if node.id not in self.fired_this_pass:
                node.steps_idle += 1
        
        return outputs
    
    def _aggregate_inbox(self, node: Node, N: int) -> List[np.ndarray]:
        """
        Aggregate inbox messages for a node.
        
        Returns list of aggregated vectors, one per batch sample.
        """
        if not node.inbox:
            return []
        
        # Check arity: if >50% of required input ports are filled, proceed
        # For simplicity, treat each sender as a "port"
        required_ports = max(1, len(node.edges_in))
        filled_ports = len([k for k, v in node.inbox.items() if v])
        
        if filled_ports < required_ports * 0.5:
            # Zero-fill missing ports - still proceed
            pass
        
        # Aggregate messages per batch slot
        # Each sender may have sent different batch samples
        batch_aggregated = []
        
        for batch_idx in range(N):
            vectors = []
            
            for sender_id, messages in node.inbox.items():
                for vec, time_arrived in messages:
                    # Decay based on arrival time
                    decay = np.exp(-self.lambda_async * (self.current_t - time_arrived))
                    vectors.append(vec * decay)
            
            if vectors:
                # Mean aggregate
                aggregated = np.mean(vectors, axis=0)
            else:
                aggregated = np.zeros(node.D)
            
            batch_aggregated.append(aggregated)
        
        return batch_aggregated
    
    def get_active_path(self) -> List[ActivePathRecord]:
        """Get records of nodes that fired during this forward pass."""
        return self.state.active_path
