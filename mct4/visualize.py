#!/usr/bin/env python3
"""
MCT4 Visualization Tools

Graph visualization and training metrics display.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mct4 import MCT4, MCT4Config, Primitive, NodeType


def visualize_graph(model: MCT4, save_path: Optional[str] = None):
    """
    Visualize the MCT4 graph structure.
    
    Uses networkx and matplotlib for graph visualization.
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install networkx and matplotlib for visualization:")
        print("  pip install networkx matplotlib")
        return
    
    # Build networkx graph
    G = nx.DiGraph()
    
    node_colors = {
        NodeType.INPUT: '#4CAF50',    # Green
        NodeType.HIDDEN: '#2196F3',   # Blue
        NodeType.OUTPUT: '#F44336',   # Red
    }
    
    primitive_shapes = {
        Primitive.GELU: 'o',
        Primitive.RELU: 's',
        Primitive.TANH: 'D',
        Primitive.ADD: '^',
        Primitive.GATE: 'v',
        Primitive.SOFTMAX: 'p',
        Primitive.ATTENTION: '*',
        Primitive.FORK: 'h',
        Primitive.L2NORM: 'X',
        Primitive.CONCAT: 'P',
    }
    
    for node_id, node in model.state.nodes.items():
        G.add_node(node_id)
        
        # Node attributes for visualization
        G.nodes[node_id]['pos'] = (node.last_hop if node.last_hop >= 0 else 0, node_id)
        G.nodes[node_id]['color'] = node_colors.get(node.node_type, 'gray')
        G.nodes[node_id]['primitive'] = node.primitive.name
        G.nodes[node_id]['health'] = node.rho_base
        G.nodes[node_id]['tension'] = node.tension_trace
    
    for node in model.state.nodes.values():
        for dst_id in node.edges_out:
            G.add_edge(node.id, dst_id)
    
    # Compute layout based on hop depth
    pos = {}
    hop_groups = {}
    
    for node_id, node in model.state.nodes.items():
        hop = node.last_hop if node.last_hop >= 0 else 0
        if hop not in hop_groups:
            hop_groups[hop] = []
        hop_groups[hop].append(node_id)
    
    # Position nodes by hop (x-axis) and index within hop (y-axis)
    for hop, nodes in sorted(hop_groups.items()):
        for i, node_id in enumerate(nodes):
            pos[node_id] = (hop, i - len(nodes) / 2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Draw nodes
    node_colors_list = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [500 + 200 * G.nodes[n]['health'] for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, 
                          node_size=node_sizes, ax=ax, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, ax=ax, alpha=0.5)
    
    # Draw labels
    labels = {}
    for node_id in G.nodes():
        prim = G.nodes[node_id]['primitive']
        labels[node_id] = f"{node_id}\n{prim[:4]}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title('MCT4 Graph Structure', fontsize=14)
    ax.set_xlabel('Hop Depth')
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graph visualization saved to {save_path}")
    else:
        plt.show()


def plot_training_metrics(model: MCT4, save_path: Optional[str] = None):
    """Plot training metrics from model history."""
    import matplotlib.pyplot as plt
    
    metrics = model.metrics
    
    if not metrics.loss_history:
        print("No training history available.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Loss history
    ax = axes[0, 0]
    ax.plot(metrics.loss_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.grid(True, alpha=0.3)
    
    # Node count history
    ax = axes[0, 1]
    ax.plot(metrics.node_count_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Node Count')
    ax.set_title('Graph Size Evolution')
    ax.grid(True, alpha=0.3)
    
    # Kappa (convergence) history
    ax = axes[1, 0]
    ax.plot(metrics.kappa_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('κ (Convergence Counter)')
    ax.set_title('Convergence Monitor')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=model.config.kappa_thresh, color='r', linestyle='--', 
               label='Threshold', alpha=0.5)
    ax.legend()
    
    # Health distribution
    ax = axes[1, 1]
    health_values = [n.rho_base for n in model.state.nodes.values()]
    if health_values:
        ax.hist(health_values, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Health (ρ_base)')
        ax.set_ylabel('Count')
        ax.set_title('Node Health Distribution')
        ax.axvline(x=0, color='r', linestyle='--', label='Pruning threshold')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    else:
        plt.show()


def print_graph_summary(model: MCT4):
    """Print detailed summary of graph state."""
    stats = model.get_stats()
    
    print("\n" + "=" * 60)
    print("MCT4 Graph Summary")
    print("=" * 60)
    
    print(f"\n📊 Structure:")
    print(f"   Total nodes:     {stats['total_nodes']}")
    print(f"   Input nodes:     {stats['input_nodes']}")
    print(f"   Hidden nodes:    {stats['hidden_nodes']}")
    print(f"   Output nodes:    {stats['output_nodes']}")
    print(f"   Total edges:     {stats['total_edges']}")
    print(f"   Active nodes:    {stats['active_nodes']} ({stats['active_nodes']/max(1,stats['total_nodes'])*100:.1f}%)")
    
    print(f"\n📈 State:")
    print(f"   Avg health:      {stats['avg_health']:.4f}")
    print(f"   Avg tension:     {stats['avg_tension']:.4f}")
    print(f"   Convergence κ:   {stats['kappa']} / {model.config.kappa_thresh}")
    print(f"   Is converged:    {stats['is_converged']}")
    
    print(f"\n🔧 Primitives:")
    for prim, count in sorted(stats['primitives'].items()):
        bar = '█' * min(count, 30)
        print(f"   {prim:12} {count:3} {bar}")
    
    print(f"\n📉 Training:")
    print(f"   Total passes:    {model.metrics.total_forward_passes}")
    print(f"   Pruning events:  {model.metrics.pruning_events}")
    
    if model.metrics.loss_history:
        recent_losses = model.metrics.loss_history[-10:]
        print(f"   Recent avg loss: {np.mean(recent_losses):.4f}")
    
    # Node details
    print(f"\n📋 Node Details:")
    print(f"   {'ID':>4} {'Type':>8} {'Prim':>10} {'Health':>8} {'Tension':>8} {'Idle':>6} {'Out-edges':>10}")
    print(f"   {'-'*4} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")
    
    for node_id, node in sorted(model.state.nodes.items()):
        type_str = node.node_type.value[:8]
        prim_str = node.primitive.name[:10]
        out_edges = len(node.edges_out)
        print(f"   {node_id:>4} {type_str:>8} {prim_str:>10} {node.rho_base:>8.3f} {node.tension_trace:>8.3f} {node.steps_idle:>6} {out_edges:>10}")
    
    print("=" * 60)


def compare_architectures(models: Dict[str, MCT4], save_path: Optional[str] = None):
    """Compare multiple model architectures side by side."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    names = list(models.keys())
    n_models = len(models)
    
    # Compare loss histories
    ax = axes[0]
    for name, model in models.items():
        if model.metrics.loss_history:
            ax.plot(model.metrics.loss_history, label=name, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Compare node counts
    ax = axes[1]
    for name, model in models.items():
        if model.metrics.node_count_history:
            ax.plot(model.metrics.node_count_history, label=name, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Node Count')
    ax.set_title('Graph Size Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Compare final stats
    ax = axes[2]
    final_node_counts = [len(m.state.nodes) for m in models.values()]
    final_losses = [np.mean(m.metrics.loss_history[-10:]) if m.metrics.loss_history else 0 
                    for m in models.values()]
    
    x = np.arange(n_models)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_node_counts, width, label='Nodes', color='steelblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, final_losses, width, label='Loss', color='coral')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Node Count', color='steelblue')
    ax2.set_ylabel('Final Loss', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title('Final Architecture Comparison')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def main():
    """Demo visualization tools."""
    print("MCT4 Visualization Demo")
    print("=" * 60)
    
    # Create and train a small model for visualization
    config = MCT4Config(D=32, t_budget=10, eta=0.01, kappa_thresh=20, N=8)
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    # Quick training
    print("\nTraining small model for visualization...")
    for i in range(50):
        X = np.random.randn(config.D) * 0.5
        Y = np.zeros(config.D)
        Y[:2] = [1, 0] if X[0] > 0 else [0, 1]
        model.train_step(X, Y, evolve=True)
    
    # Print summary
    print_graph_summary(model)
    
    # Try visualizations
    try:
        visualize_graph(model)
    except Exception as e:
        print(f"Graph visualization skipped: {e}")
    
    try:
        plot_training_metrics(model)
    except Exception as e:
        print(f"Metrics plot skipped: {e}")
    
    print("\nVisualization demo complete!")


if __name__ == "__main__":
    main()
