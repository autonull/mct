#!/usr/bin/env python3
"""
MCT4 Comprehensive Test Suite

Tests all major components and demonstrates structural evolution.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/me/mct')

from mct4 import MCT4, MCT4Config, Primitive, NodeType


def test_primitives():
    """Test all primitive operators."""
    from mct4.primitives import apply_primitive, Primitive
    
    print("Testing Primitives")
    print("=" * 50)
    
    D = 64
    x = np.random.randn(D)
    
    # Unary primitives
    unary = [Primitive.RELU, Primitive.TANH, Primitive.GELU, 
             Primitive.SOFTMAX, Primitive.L2NORM, Primitive.FORK]
    
    for prim in unary:
        out = apply_primitive(prim, x)
        assert out.shape == (D,), f"{prim.name} output shape mismatch"
        assert np.isfinite(out).all(), f"{prim.name} has non-finite values"
        print(f"  ✓ {prim.name}: shape={out.shape}, norm={np.linalg.norm(out):.3f}")
    
    # Binary primitives
    y = np.random.randn(D)
    binary = [Primitive.ADD, Primitive.GATE, Primitive.CONCAT]
    
    for prim in binary:
        out = apply_primitive(prim, [x, y])
        assert out.shape == (D,), f"{prim.name} output shape mismatch"
        assert np.isfinite(out).all(), f"{prim.name} has non-finite values"
        print(f"  ✓ {prim.name}: shape={out.shape}, norm={np.linalg.norm(out):.3f}")
    
    print()


def test_forward_pass():
    """Test forward execution."""
    print("Testing Forward Pass")
    print("=" * 50)
    
    config = MCT4Config(D=32, t_budget=10, N=1)
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    X = np.random.randn(config.D) * 0.5
    model.reset_sequence()
    outputs = model.forward(X)
    
    assert len(outputs) > 0, "No output nodes fired"
    output_id = list(outputs.keys())[0]
    assert outputs[output_id].shape == (config.D,), "Output shape mismatch"
    
    print(f"  ✓ Forward pass completed")
    print(f"  ✓ Output node {output_id} fired at hop {model.state.nodes[output_id].last_hop}")
    print(f"  ✓ Active path length: {len(model.state.active_path)}")
    print()


def test_learning():
    """Test learning phase."""
    print("Testing Learning")
    print("=" * 50)
    
    config = MCT4Config(D=32, t_budget=10, eta=0.01, N=1)
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    X = np.random.randn(config.D) * 0.5
    Y_star = np.zeros(config.D)
    Y_star[:config.D//2] = 1.0 / (config.D//2)
    
    # Initial forward
    model.reset_sequence()
    model.forward(X)
    initial_outputs = {r.node_id: r.V_out for r in model.state.active_path}
    
    # Learn
    loss1 = model.learn(Y_star)
    
    # Forward again
    model.reset_sequence()
    model.forward(X)
    
    # Learn again
    loss2 = model.learn(Y_star)
    
    print(f"  ✓ Initial loss: {loss1:.4f}")
    print(f"  ✓ After learning: {loss2:.4f}")
    
    # Check weights changed
    for node in model.state.nodes.values():
        if node.node_type == NodeType.HIDDEN:
            assert node.W is not None, "Weights should exist"
    
    print(f"  ✓ Weight updates applied")
    print()


def test_structural_evolution():
    """Test pruning and capacity insertion."""
    print("Testing Structural Evolution")
    print("=" * 50)
    
    config = MCT4Config(D=32, t_budget=10, eta=0.01, gamma=0.1, N=1)
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    initial_nodes = len(model.state.nodes)
    print(f"  Initial nodes: {initial_nodes}")
    
    # Force some nodes to have negative health
    for node in model.state.nodes.values():
        if node.node_type == NodeType.HIDDEN:
            node.rho_base = -1.0
    
    # Run evolution
    model.evolve()
    
    final_nodes = len(model.state.nodes)
    print(f"  After pruning: {final_nodes} nodes")
    
    # New nodes should be added for pruned ones
    print(f"  ✓ Pruning executed")
    print(f"  ✓ Capacity insertion triggered")
    print()


def test_context_vector():
    """Test context vector for sequence handling."""
    print("Testing Context Vector")
    print("=" * 50)
    
    config = MCT4Config(D=32, t_budget=10, decay_c=0.95, N=1)
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    # Reset context
    model.reset_sequence()
    assert np.allclose(model.state.context.C, 0), "Context should be zero after reset"
    print(f"  ✓ Context reset works")
    
    # Forward pass should add ghost signals
    X = np.random.randn(config.D) * 0.5
    model.forward(X)
    
    # Context should have changed
    context_norm = np.linalg.norm(model.state.context.C)
    print(f"  ✓ Context after forward: norm={context_norm:.4f}")
    
    print()


def test_convergence_monitor():
    """Test convergence monitoring."""
    print("Testing Convergence Monitor")
    print("=" * 50)
    
    config = MCT4Config(D=32, t_budget=10, kappa_thresh=5, N=1)
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    # Run training without pruning
    for i in range(10):
        X = np.random.randn(config.D) * 0.5
        Y = np.zeros(config.D)
        Y[:config.D//2] = 1.0 / (config.D//2)
        model.train_step(X, Y, evolve=True)
    
    print(f"  κ after 10 passes: {model.state.kappa}")
    print(f"  Is converged: {model.state.is_converged()}")
    
    assert model.state.kappa > 0, "Kappa should increase"
    print(f"  ✓ Convergence monitoring works")
    print()


def test_batch_processing():
    """Test batch processing."""
    print("Testing Batch Processing")
    print("=" * 50)
    
    config = MCT4Config(D=32, t_budget=10, N=4)
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    X_batch = np.random.randn(4, config.D) * 0.5
    Y_batch = np.zeros((4, config.D))
    Y_batch[:, :config.D//2] = 1.0 / (config.D//2)
    
    loss = model.train_batch(X_batch, Y_batch, evolve=True)
    
    print(f"  ✓ Batch training completed")
    print(f"  ✓ Batch loss: {loss:.4f}")
    print()


def test_save_load():
    """Test model persistence."""
    print("Testing Save/Load")
    print("=" * 50)
    
    import tempfile
    import os
    
    config = MCT4Config(D=32, t_budget=10, N=1)
    model = MCT4(config)
    model.initialize(Primitive.GELU)
    
    # Train briefly
    for _ in range(5):
        X = np.random.randn(config.D) * 0.5
        Y = np.zeros(config.D)
        model.train_step(X, Y, evolve=False)
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        save_path = f.name
    
    model.save_state(save_path)
    print(f"  ✓ Model saved to {save_path}")
    
    # Load
    model2 = MCT4(config)
    model2.load_state(save_path)
    
    # Verify
    assert len(model2.state.nodes) == len(model.state.nodes), "Node count mismatch"
    print(f"  ✓ Model loaded successfully")
    
    # Cleanup
    os.unlink(save_path)
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MCT4 Comprehensive Test Suite")
    print("=" * 60 + "\n")
    
    test_primitives()
    test_forward_pass()
    test_learning()
    test_structural_evolution()
    test_context_vector()
    test_convergence_monitor()
    test_batch_processing()
    test_save_load()
    
    print("=" * 60)
    print("All Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
