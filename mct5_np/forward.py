"""
MCT5 Phase 1 — Forward Execution

Executes the DAG in topological (depth-layer) order.  All nodes at the same
depth are processed together to enable vectorised NumPy operations.

Activation potential for node i:
    ρᵢ(active) = ρᵢ + dot(Sᵢ, V_in) + Re(⟨Sᵢ, R⟩)

Nodes with ρᵢ(active) < τ(t) inject ghost signals into R and go idle.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .types import GraphState, Node, NodeType, ActiveRecord
from .primitives import Primitive, apply_primitive
from .residue import HolographicResidue


class ForwardExecutor:
    """
    Executes Phase 1 of MCT5 forward pass.

    Strategy:
    1. Compute cached topo order and group into depth layers.
    2. Propagate activations layer by layer.
    3. At each node: compute ρ, check τ, fire or ghost-inject.
    4. Record ActiveRecord for every firing node.
    """

    def __init__(self,
                 state: GraphState,
                 residue: HolographicResidue,
                 lambda_async: float = 0.1):
        self.state = state
        self.residue = residue
        self.lambda_async = lambda_async

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Run a single forward pass with input vector X ∈ ℝᴰ.

        Returns:
            {output_node_id: output_vector} for each output node that fired.
        """
        state = self.state
        state.active_path.clear()

        # Node-id → output vector for fired nodes this pass
        node_outputs: Dict[int, np.ndarray] = {}
        # Node-id → {sender_id: vec} used during this pass
        inbox: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        # Arrival hop per sender (for async decay)
        inbox_time: Dict[int, Dict[int, int]] = defaultdict(dict)

        topo = state.topo_order()
        outputs: Dict[int, np.ndarray] = {}

        for hop, node_id in enumerate(topo):
            node = state.nodes.get(node_id)
            if node is None:
                continue

            tau = state.get_tau(node.topo_depth)

            if node.node_type == NodeType.INPUT:
                # Input nodes always fire with the external input
                V_out = X.copy()
                node_outputs[node_id] = V_out
                node.steps_idle = 0
                node.last_hop = 0  # type: ignore[attr-defined]
                self._route(node_id, V_out, node.topo_depth, state, inbox, inbox_time)
                record = ActiveRecord(
                    node_id=node_id, V_in=X, V_weighted=X, V_out=V_out,
                    hop=0, inbox_contributions={}, goodness=float(np.dot(V_out, V_out))
                )
                state.active_path.append(record)
                continue

            # Build aggregated input from inbox
            V_in, contribs = self._aggregate(node_id, node, inbox, inbox_time, node.topo_depth)

            if V_in is None:
                # No messages arrived at all (node is unreachable this pass)
                node.steps_idle += 1
                continue

            # Activation potential
            residue_boost = float(np.real(np.dot(node.S, self.residue.R)))
            rho_active = node.rho + float(np.dot(node.S, V_in)) + residue_boost

            if rho_active < tau:
                # Ghost injection
                self.residue.inject_ghost(node_id, rho_active, node.topo_depth)
                node.steps_idle += 1
                continue

            # ── Node fires ────────────────────────────────────────────────────
            node.steps_idle = 0

            # Linear transform: W @ V_in
            V_weighted = node.apply_W(V_in) + node.bias

            # Primitive nonlinearity
            V_out = apply_primitive(node.primitive, V_weighted)

            # Guard against NaN/Inf
            if not np.isfinite(V_out).all():
                V_out = np.zeros_like(V_out)

            node_outputs[node_id] = V_out
            self._route(node_id, V_out, node.topo_depth, state, inbox, inbox_time)

            goodness = float(np.dot(V_out, V_out))
            record = ActiveRecord(
                node_id=node_id, V_in=V_in, V_weighted=V_weighted, V_out=V_out,
                hop=node.topo_depth, inbox_contributions=contribs, goodness=goodness
            )
            state.active_path.append(record)

            if node.node_type == NodeType.OUTPUT:
                outputs[node_id] = V_out

        # End of pass: update residue
        self.residue.end_of_pass()

        # Idle counters for non-firing hidden nodes
        fired_ids = {r.node_id for r in state.active_path}
        for node in state.nodes.values():
            if node.node_type == NodeType.HIDDEN and node.id not in fired_ids:
                node.steps_idle += 1

        return outputs

    def execute_batch(self, X_batch: np.ndarray) -> Tuple[List[Dict[int, np.ndarray]], float]:
        """
        Run forward pass for each sample in X_batch (N, D).
        Returns list of output dicts and mean loss (0 for forward-only).
        """
        results = []
        for x in X_batch:
            outputs = self.execute(x)
            results.append(outputs)
        return results, 0.0

    def reset_context(self):
        """Clear holographic residue at sequence boundary."""
        self.residue.reset()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _route(self, src_id: int, V_out: np.ndarray, t: int,
               state: GraphState,
               inbox: Dict[int, Dict[int, np.ndarray]],
               inbox_time: Dict[int, Dict[int, int]]):
        """Send V_out to all downstream neighbours."""
        for dst_id in state.edges_out.get(src_id, []):
            if dst_id in state.nodes:
                inbox[dst_id][src_id] = V_out.copy()
                inbox_time[dst_id][src_id] = t

    def _aggregate(self, node_id: int, node: Node,
                   inbox: Dict[int, Dict[int, np.ndarray]],
                   inbox_time: Dict[int, Dict[int, int]],
                   t_now: int
                   ) -> Tuple[Optional[np.ndarray], Dict[int, np.ndarray]]:
        """
        Aggregate inbox messages into a single V_in vector.
        Returns (V_in, contributions_dict) or (None, {}) if no messages.
        """
        messages = inbox.get(node_id, {})
        if not messages:
            return None, {}

        D = node.D
        contributions: Dict[int, np.ndarray] = {}
        vectors: List[np.ndarray] = []

        for sender_id, vec in messages.items():
            t_arrived = inbox_time[node_id].get(sender_id, t_now)
            decay = np.exp(-self.lambda_async * (t_now - t_arrived))
            v = vec * decay
            contributions[sender_id] = v
            vectors.append(v)

        if not vectors:
            return None, {}

        # Aggregate: primitive-aware
        if node.primitive in (Primitive.GATE, Primitive.BILINEAR, Primitive.CONCAT_PROJECT):
            # Multi-input primitives want the list
            agg = apply_primitive(node.primitive, [v for v in vectors])
        else:
            agg = np.mean(vectors, axis=0)

        return agg, contributions
