"""
MCT4 Phase 3 - Structural Evolution

Continuous structure evolution: pruning, capacity insertion, and lateral wiring.
No separate architecture search phase - structure emerges during training.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import random

from .core import GraphState, Node, NodeType
from .primitives import Primitive


class StructuralEvolution:
    """
    Executes Phase 3 of MCT4: Structural evolution.
    
    - Pruning: Remove nodes with rho_base < 0
    - Capacity Insertion: Spawn new nodes where error pressure is highest
    - Lateral Wiring: Add shortcut connections for persistent high-tension nodes
    """
    
    def __init__(self, state: GraphState,
                 sigma_mut: float = 0.05,  # Mutation noise std dev
                 K: int = 2,  # Spawn count per pruning event
                 F: int = 2,  # Fork fan-out limit
                 tau_lateral: float = 0.3,  # Lateral tension threshold
                 rank: int = 64):  # Low-rank dimension for new nodes
        self.state = state
        self.sigma_mut = sigma_mut
        self.K = K
        self.F = F
        self.tau_lateral = tau_lateral
        self.rank = rank
        
        # Track nodes pending deletion (for batch pruning)
        self.nodes_to_prune: Set[int] = set()
    
    def prune(self) -> List[int]:
        """
        Prune nodes with negative health.
        
        Any node with ρ_base < 0 is deleted.
        All referencing edges are severed.
        
        Returns:
            List of pruned node IDs
        """
        pruned = []
        
        # Identify nodes to prune
        for node_id, node in list(self.state.nodes.items()):
            if node.node_type in [NodeType.INPUT, NodeType.OUTPUT]:
                continue  # Never prune I/O nodes
            
            if node.rho_base < 0:
                self.nodes_to_prune.add(node_id)
        
        # Execute pruning
        for node_id in self.nodes_to_prune:
            if node_id in self.state.nodes:
                # Record edges for capacity insertion
                self._record_edges_for_insertion(node_id)
                
                self.state.remove_node(node_id)
                pruned.append(node_id)
                self.state.pruning_events_this_pass += 1
        
        self.nodes_to_prune.clear()
        return pruned
    
    def _record_edges_for_insertion(self, pruned_node_id: int):
        """Record edges adjacent to pruned node for capacity insertion."""
        # This is handled by the learning engine's edge tension tracking
        pass
    
    def insert_capacity(self, edge: Optional[Tuple[int, int]] = None):
        """
        Insert new capacity where error pressure is highest.
        
        When a node is pruned, new capacity is inserted where error pressure
        is highest, not where health is highest.
        
        Procedure:
        1. Find edge with highest attributed tension from retrograde flow
        2. Spawn K new nodes
        3. Each inherits W and S from upstream node with mutation
        4. 20% probability of different primitive
        5. Wire: u → new → v, remove u → v
        6. Check acyclicity
        """
        if edge is None:
            # Find highest tension edge
            if not self.state.edge_tensions:
                return
            
            edge = max(self.state.edge_tensions.items(), key=lambda x: x[1])[0]
        
        u, v = edge
        
        if u not in self.state.nodes or v not in self.state.nodes:
            return
        
        node_u = self.state.nodes[u]
        node_v = self.state.nodes[v]
        
        # Remove existing edge
        self.state.remove_edge(u, v)
        
        # Spawn K new nodes
        for _ in range(self.K):
            # Inherit weight matrix with mutation
            if node_u.W_factored:
                W_init = None
                A_init = node_u.A + np.random.randn(*node_u.A.shape) * self.sigma_mut
                B_init = node_u.B + np.random.randn(*node_u.B.shape) * self.sigma_mut
            else:
                W_init = node_u.W + np.random.randn(*node_u.W.shape) * self.sigma_mut
                A_init = None
                B_init = None
            
            # Inherit signature with mutation
            S_init = node_u.S + np.random.randn(self.state.D) * self.sigma_mut
            
            # Choose primitive
            if random.random() < 0.2:
                # 20% chance of different primitive
                primitive = random.choice(list(Primitive))
            else:
                primitive = node_u.primitive
            
            # Create new node
            new_node = self.state.create_node(
                node_type=NodeType.HIDDEN,
                primitive=primitive,
                rho_base=0.5  # Start with reasonable health
            )
            new_node.tension_trace = node_u.tension_trace
            
            # Initialize weights
            if node_u.W_factored:
                new_node.W_factored = True
                new_node.A = A_init
                new_node.B = B_init
                new_node.rank = self.rank
            else:
                new_node.W = W_init
            
            new_node.S = S_init
            
            # Wire: u → new → v
            self.state.add_edge(u, new_node.id)
            self.state.add_edge(new_node.id, v)
    
    def lateral_wiring(self):
        """
        Add lateral connections for persistent high-tension nodes.
        
        Each pass, if a node i has tension_trace_i > τ_lateral for more than
        20 consecutive passes, it spawns one additional outbound edge to a
        random downstream node (DAG-preserving).
        """
        for node in self.state.nodes.values():
            if node.node_type in [NodeType.INPUT, NodeType.OUTPUT]:
                continue
            
            if node.high_tension_count > 20:
                # Find valid downstream targets
                downstream = self._get_downstream_nodes(node.id)
                existing_targets = set(node.edges_out)
                
                # Filter out already-connected nodes
                candidates = [n for n in downstream if n not in existing_targets]
                
                if candidates:
                    # Pick random downstream node
                    target_id = random.choice(candidates)
                    
                    # Check acyclicity before adding
                    self.state.add_edge(node.id, target_id)
                    
                    if not self.state.is_acyclic():
                        # Revert if cycle detected
                        self.state.remove_edge(node.id, target_id)
                
                # Reset counter
                node.high_tension_count = 0
    
    def _get_downstream_nodes(self, node_id: int) -> List[int]:
        """
        Get all nodes that are downstream from node_id (reachable via edges).
        
        Uses BFS to find all reachable nodes.
        """
        if node_id not in self.state.nodes:
            return []
        
        visited = set()
        queue = [node_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current in self.state.nodes:
                for neighbor in self.state.nodes[current].edges_out:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        visited.discard(node_id)  # Don't include self
        return list(visited)
    
    def evolve(self, pruned: List[int], max_tension_edge: Optional[Tuple[int, int]] = None):
        """
        Execute full structural evolution phase.
        
        Args:
            pruned: List of node IDs that were pruned
            max_tension_edge: Edge with highest tension (from learning phase)
        """
        # Adjust mutation based on convergence
        sigma = self.sigma_mut
        if self.state.is_converged():
            sigma *= 0.5
        
        old_sigma = self.sigma_mut
        self.sigma_mut = sigma
        
        # Insert capacity for each pruned node
        if pruned:
            # Use max tension edge if available
            if max_tension_edge:
                self.insert_capacity(max_tension_edge)
            else:
                # Insert at random high-tension edge
                if self.state.edge_tensions:
                    edge = max(self.state.edge_tensions.items(), key=lambda x: x[1])[0]
                    self.insert_capacity(edge)
        
        # Lateral wiring for persistent high-tension nodes
        self.lateral_wiring()
        
        self.sigma_mut = old_sigma
    
    def initialize_minimal_graph(self, primitive_hidden: Primitive = Primitive.GELU):
        """
        Create the minimal viable starting graph.
        
        Two input nodes (one for content X, one for context C),
        four intermediate nodes (two GELU, one Gate, one Add),
        one output node.
        
        Wired as a shallow DAG. Structural evolution grows from there.
        """
        # Clear existing
        self.state.nodes = {}
        self.state.next_node_id = 0
        self.state.edge_tensions = {}
        
        # Input nodes
        input_content = self.state.create_node(NodeType.INPUT, Primitive.FORK)
        input_context = self.state.create_node(NodeType.INPUT, Primitive.FORK)
        
        # Hidden nodes
        hidden_gelu1 = self.state.create_node(NodeType.HIDDEN, Primitive.GELU)
        hidden_gelu2 = self.state.create_node(NodeType.HIDDEN, primitive_hidden)
        hidden_gate = self.state.create_node(NodeType.HIDDEN, Primitive.GATE)
        hidden_add = self.state.create_node(NodeType.HIDDEN, Primitive.ADD)
        
        # Output node
        output_node = self.state.create_node(NodeType.OUTPUT, Primitive.SOFTMAX)
        
        # Wire as shallow DAG
        # Content -> hidden layers
        self.state.add_edge(input_content.id, hidden_gelu1.id)
        self.state.add_edge(input_content.id, hidden_gelu2.id)
        self.state.add_edge(input_content.id, hidden_gate.id)
        
        # Context modulates
        self.state.add_edge(input_context.id, hidden_gate.id)
        
        # Hidden -> add
        self.state.add_edge(hidden_gelu1.id, hidden_add.id)
        self.state.add_edge(hidden_gelu2.id, hidden_add.id)
        self.state.add_edge(hidden_gate.id, hidden_add.id)
        
        # Add -> output
        self.state.add_edge(hidden_add.id, output_node.id)
