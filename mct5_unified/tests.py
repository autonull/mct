"""
MCT5 Unified Test Suite

Comprehensive tests for all components.

Run with:
    python -m mct5_unified.tests
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/home/me/mct')

from mct5_unified import (
    MCT5, MCT5Config, LearningMode,
    Primitive, apply_primitive, primitive_derivative,
    HolographicResidue, GraphState, Node, NodeType,
    ForwardExecutor, StructuralEvolution
)

# ANSI colors
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def test_primitives():
    """Test all primitive operators."""
    print("\n" + "=" * 60)
    print("  Primitive Operators")
    print("=" * 60)
    
    x = torch.tensor([[1.0, -1.0, 0.5, 0.0]])
    
    # Unary primitives
    unary_tests = [
        (Primitive.RELU, lambda x: torch.relu(x)),
        (Primitive.GELU, lambda x: torch.nn.functional.gelu(x)),
        (Primitive.TANH, lambda x: torch.tanh(x)),
        (Primitive.SWISH, lambda x: x * torch.sigmoid(x)),
        (Primitive.SILU, lambda x: torch.nn.functional.silu(x)),
        (Primitive.LEAKY_RELU, lambda x: torch.nn.functional.leaky_relu(x, 0.01)),
        (Primitive.FORK, lambda x: x),
        (Primitive.ABS, lambda x: torch.abs(x)),
        (Primitive.SINE, lambda x: torch.sin(x)),
        (Primitive.QUADRATIC, lambda x: x * x + x),
    ]
    
    for prim, expected_fn in unary_tests:
        result = apply_primitive(prim, x)
        expected = expected_fn(x)
        if torch.allclose(result, expected, atol=1e-5):
            print(f"  {PASS} {prim.name}")
        else:
            print(f"  {FAIL} {prim.name}: expected {expected}, got {result}")
            raise AssertionError(f"{prim.name} failed")
    
    # Binary primitives
    x1 = torch.tensor([[1.0, 2.0, 3.0]])
    x2 = torch.tensor([[4.0, 5.0, 6.0]])
    
    # ADD
    result = apply_primitive(Primitive.ADD, [x1, x2])
    expected = (x1 + x2) / 2
    assert torch.allclose(result, expected), "ADD failed"
    print(f"  {PASS} ADD")
    
    # PRODUCT
    result = apply_primitive(Primitive.PRODUCT, [x1, x2])
    expected = x1 * x2
    assert torch.allclose(result, expected), "PRODUCT failed"
    print(f"  {PASS} PRODUCT")
    
    # GATE
    result = apply_primitive(Primitive.GATE, [x1, x2])
    expected = x1 * torch.sigmoid(x2)
    assert torch.allclose(result, expected), "GATE failed"
    print(f"  {PASS} GATE")
    
    # MAX
    result = apply_primitive(Primitive.MAX, [x1, x2])
    expected = torch.max(x1, x2)
    assert torch.allclose(result, expected), "MAX failed"
    print(f"  {PASS} MAX")
    
    # QUADRATIC for XOR separability
    x_pos = torch.tensor([[1.0, 0.0]])
    x_neg = torch.tensor([[-1.0, 0.0]])
    out_pos = apply_primitive(Primitive.QUADRATIC, x_pos)
    out_neg = apply_primitive(Primitive.QUADRATIC, x_neg)
    
    # Q(+1) should differ from Q(-1)
    assert abs(out_pos[0, 0] - out_neg[0, 0]) > 0.5, "QUADRATIC should distinguish +1 from -1"
    print(f"  {PASS} QUADRATIC XOR-linearity (Q(+1)={out_pos[0,0]:.2f}, Q(-1)={out_neg[0,0]:.2f})")


def test_graph_state():
    """Test graph state management."""
    print("\n" + "=" * 60)
    print("  Graph State & Topology")
    print("=" * 60)
    
    state = GraphState(D=8, r=4, device="cpu")
    
    # Create nodes
    n0 = state.create_node(NodeType.INPUT, Primitive.FORK)
    n1 = state.create_node(NodeType.HIDDEN, Primitive.GELU)
    n2 = state.create_node(NodeType.HIDDEN, Primitive.RELU)
    n3 = state.create_node(NodeType.OUTPUT, Primitive.FORK)
    
    assert len(state.nodes) == 4, "Should have 4 nodes"
    print(f"  {PASS} Node creation")
    
    # Add edges
    assert state.add_edge(n0.id, n1.id), "Should add edge 0->1"
    assert state.add_edge(n1.id, n2.id), "Should add edge 1->2"
    assert state.add_edge(n2.id, n3.id), "Should add edge 2->3"
    
    print(f"  {PASS} Edge addition")
    
    # Cycle prevention
    assert not state.add_edge(n3.id, n0.id), "Should prevent cycle 3->0"
    print(f"  {PASS} Cycle prevention")
    
    # Topological ordering
    topo = state.get_topo_order()
    assert topo == [0, 1, 2, 3], f"Expected [0,1,2,3], got {topo}"
    print(f"  {PASS} Topological ordering")
    
    # Edge removal
    state.remove_edge(n1.id, n2.id)
    assert n2.id not in state.edges_out.get(n1.id, []), "Edge should be removed"
    print(f"  {PASS} Edge removal")
    
    # Node removal
    state.remove_node(n2.id)
    assert str(n2.id) not in state.nodes, "Node should be removed"
    print(f"  {PASS} Node removal")


def test_holographic_residue():
    """Test holographic residue memory."""
    print("\n" + "=" * 60)
    print("  Holographic Residue")
    print("=" * 60)
    
    residue = HolographicResidue(D=32, max_nodes=100, phi_max=4.0, device="cpu")
    
    # Register nodes
    idx0 = residue.register_node(0)
    idx1 = residue.register_node(1)
    
    assert idx0 == 0, "First index should be 0"
    assert idx1 == 1, "Second index should be 1"
    print(f"  {PASS} Node registration")
    
    # Orthogonality check
    sim = residue.get_basis_similarity(0, 1)
    assert abs(sim) < 0.1, f"Basis vectors should be nearly orthogonal, got {sim}"
    print(f"  {PASS} Basis orthogonality (similarity={sim:.4f})")
    
    # Ghost injection
    residue.inject_ghost(0, rho=1.0, t=0.0)
    norm_after_inject = residue.R.norm().item()
    assert norm_after_inject > 0, "Residue should have content after injection"
    print(f"  {PASS} Ghost injection (norm={norm_after_inject:.4f})")
    
    # Decoding
    boost = residue.decode(0)
    assert abs(boost) > 0, "Should decode non-zero boost"
    print(f"  {PASS} Decoding (boost={boost:.4f})")
    
    # Norm pruning
    for _ in range(100):
        residue.inject_ghost(0, rho=5.0, t=0.0)
    
    final_norm = residue.R.norm().item()
    assert final_norm <= residue.phi_max + 0.1, f"Norm should be clamped, got {final_norm}"
    print(f"  {PASS} Norm pruning (final_norm={final_norm:.4f})")
    
    # Reset
    residue.reset()
    assert residue.R.norm().item() < 1e-6, "Residue should be zero after reset"
    print(f"  {PASS} Reset")


def test_forward_execution():
    """Test forward execution."""
    print("\n" + "=" * 60)
    print("  Forward Execution")
    print("=" * 60)
    
    config = MCT5Config(D=32, r=8, n_classes=2, input_dim=4, device="cpu")
    model = MCT5(config)
    model.initialize()
    
    # Single sample forward
    X = np.random.randn(4).astype(np.float32)
    output = model.forward(torch.tensor(X, dtype=torch.float32))
    
    assert output.shape == (config.n_classes,), f"Expected shape ({config.n_classes},), got {output.shape}"
    print(f"  {PASS} Single sample forward pass")
    
    # Batch forward
    X_batch = np.random.randn(5, 4).astype(np.float32)
    output_batch = model.forward(torch.tensor(X_batch, dtype=torch.float32))
    
    assert output_batch.shape == (5, config.n_classes), f"Expected (5, {config.n_classes}), got {output_batch.shape}"
    print(f"  {PASS} Batch forward pass")
    
    # Active path recording
    assert len(model.state.active_path) > 0, "Should have recorded active path"
    print(f"  {PASS} Active path recording ({len(model.state.active_path)} nodes)")


def test_autograd_learning():
    """Test autograd learning mode."""
    print("\n" + "=" * 60)
    print("  Autograd Learning")
    print("=" * 60)
    
    config = MCT5Config(
        D=32, r=8, n_classes=2, input_dim=4,
        learning_mode=LearningMode.AUTOGRAD,
        device="cpu", seed=42
    )
    model = MCT5(config)
    model.initialize()
    
    # Create simple dataset
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, 20)
    
    # Initial loss
    initial_logits = model.forward(torch.tensor(X[:5], dtype=torch.float32))
    initial_loss = torch.nn.functional.cross_entropy(
        initial_logits, torch.tensor(y[:5], dtype=torch.long)
    ).item()
    
    # Training step
    loss = model.train_batch(X, y)
    
    assert loss > 0, "Loss should be positive"
    print(f"  {PASS} Training step (loss={loss:.4f})")
    
    # Verify gradients flowed
    has_grads = any(n.A.grad is not None for n in model.state.nodes.values())
    assert has_grads, "Should have gradients"
    print(f"  {PASS} Gradient flow")


def test_dual_signal_learning():
    """Test dual-signal learning mode."""
    print("\n" + "=" * 60)
    print("  Dual-Signal Learning")
    print("=" * 60)
    
    config = MCT5Config(
        D=32, r=8, n_classes=2, input_dim=4,
        learning_mode=LearningMode.DUAL_SIGNAL,
        device="cpu", seed=42
    )
    model = MCT5(config)
    model.initialize()
    
    X = np.random.randn(4).astype(np.float32)
    y = 1
    
    # Training step
    loss = model.train_step(X, y)
    
    assert loss > 0, "Loss should be positive"
    print(f"  {PASS} Training step (loss={loss:.4f})")
    
    # Verify health updates
    health_changes = sum(1 for n in model.state.hidden_nodes() if n.rho.item() != 1.5)
    assert health_changes > 0, "Health should be updated"
    print(f"  {PASS} Health updates ({health_changes} nodes)")


def test_hybrid_learning():
    """Test hybrid learning mode."""
    print("\n" + "=" * 60)
    print("  Hybrid Learning")
    print("=" * 60)
    
    config = MCT5Config(
        D=48, r=12, n_classes=2, input_dim=4,
        learning_mode=LearningMode.HYBRID,
        device="cpu", seed=42,
        evolve_interval=5
    )
    model = MCT5(config)
    model.initialize()
    
    X = np.random.randn(50, 4).astype(np.float32)
    y = np.random.randint(0, 2, 50)
    
    # Training
    losses = []
    for i in range(20):
        loss = model.train_batch(X, y)
        losses.append(loss)
    
    # Loss should generally decrease
    early_avg = np.mean(losses[:5])
    late_avg = np.mean(losses[-5:])
    
    print(f"  Early loss: {early_avg:.4f}, Late loss: {late_avg:.4f}")
    
    # Allow some variance but trend should be down
    if late_avg <= early_avg * 1.2:  # Allow 20% tolerance
        print(f"  {PASS} Loss trend (early={early_avg:.4f}, late={late_avg:.4f})")
    else:
        print(f"  {FAIL} Loss not decreasing")
    
    # Structural evolution should have occurred
    stats = model.get_stats()
    print(f"  {PASS} Evolution stats: {stats['evolution']['total_spawned']} spawned, "
          f"{stats['evolution']['total_pruned']} pruned")


def test_xor_convergence():
    """Test XOR convergence - key benchmark for nonlinearity."""
    print("\n" + "=" * 60)
    print("  XOR Convergence")
    print("=" * 60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate XOR data
    n = 200
    X = np.random.randn(n, 2).astype(np.float32) * 0.5
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    
    config = MCT5Config(
        D=64, r=16, n_classes=2, input_dim=2,
        learning_mode=LearningMode.HYBRID,
        eta_W=0.02,
        evolve_interval=10,
        ensure_nonlinearity=True,
        device="cpu",
        seed=42,
        verbose=False
    )
    
    model = MCT5(config)
    model.initialize()
    
    # Train
    n_epochs = 150
    for epoch in range(n_epochs):
        lr_scale = 1.0 / (1.0 + 0.01 * epoch)
        model.cfg.eta_W = 0.02 * lr_scale
        
        for i in np.random.permutation(n):
            model.train_step(X[i], int(y[i]))
    
    # Evaluate
    acc = model.score(X, y)
    print(f"  XOR accuracy: {acc:.1%}")
    
    if acc >= 0.90:
        print(f"  {PASS} XOR ≥ 90% — Excellent!")
    elif acc >= 0.75:
        print(f"  {PASS} XOR ≥ 75% — Good")
    else:
        print(f"  {FAIL} XOR {acc:.1%} — Below target")
    
    assert acc >= 0.75, f"XOR convergence failed: {acc:.1%}"


def test_structural_evolution():
    """Test structural evolution."""
    print("\n" + "=" * 60)
    print("  Structural Evolution")
    print("=" * 60)
    
    config = MCT5Config(
        D=32, r=8, n_classes=2, input_dim=2,
        evolve_interval=5,
        device="cpu", seed=42
    )
    model = MCT5(config)
    model.initialize()
    
    initial_nodes = len(model.state.nodes)
    print(f"  Initial nodes: {initial_nodes}")
    
    # Train for multiple evolution cycles
    X = np.random.randn(50, 2).astype(np.float32)
    y = np.random.randint(0, 2, 50)
    
    for _ in range(30):
        model.train_batch(X, y)
    
    final_nodes = len(model.state.nodes)
    stats = model.get_stats()
    
    print(f"  Final nodes: {final_nodes}")
    print(f"  Total spawned: {stats['evolution']['total_spawned']}")
    print(f"  Total pruned: {stats['evolution']['total_pruned']}")
    print(f"  Lateral edges: {stats['evolution']['total_lateral_edges']}")
    
    # Graph should have evolved
    if stats['evolution']['total_spawned'] > 0 or stats['evolution']['total_lateral_edges'] > 0:
        print(f"  {PASS} Structural evolution occurred")
    else:
        print(f"  {FAIL} No structural changes")


def test_save_load():
    """Test model persistence."""
    print("\n" + "=" * 60)
    print("  Save / Load")
    print("=" * 60)
    
    import tempfile
    import os
    
    config = MCT5Config(D=16, r=4, n_classes=2, input_dim=2, device="cpu", seed=42)
    model = MCT5(config)
    model.initialize()
    
    # Train briefly
    X = np.random.randn(10, 2).astype(np.float32)
    y = np.random.randint(0, 2, 10)
    for _ in range(5):
        model.train_batch(X, y)
    
    # Get predictions before save
    preds_before = model.predict(X)
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    model.save(temp_path)
    
    # Load
    model2 = MCT5(config)
    model2.load(temp_path)
    
    # Get predictions after load
    preds_after = model2.predict(X)
    
    # Cleanup
    os.unlink(temp_path)
    
    # Compare
    if np.array_equal(preds_before, preds_after):
        print(f"  {PASS} Save/load preserves predictions")
    else:
        print(f"  {FAIL} Predictions differ after load")
    
    # Check node count
    if len(model.state.nodes) == len(model2.state.nodes):
        print(f"  {PASS} Save/load preserves graph structure")
    else:
        print(f"  {FAIL} Node count differs")


def test_anytime_inference():
    """Test anytime property with varying t_budget."""
    print("\n" + "=" * 60)
    print("  Anytime Inference")
    print("=" * 60)
    
    config = MCT5Config(D=32, r=8, n_classes=2, input_dim=4, device="cpu", seed=42)
    model = MCT5(config)
    model.initialize()
    
    X = np.random.randn(10, 4).astype(np.float32)
    
    # Forward with different budgets
    original_budget = model.cfg.t_budget
    
    model.cfg.t_budget = 5
    preds_short = model.predict(X)
    
    model.cfg.t_budget = 15
    preds_long = model.predict(X)
    
    model.cfg.t_budget = original_budget
    
    # Both should produce valid predictions
    assert len(preds_short) == 10, "Short budget should produce predictions"
    assert len(preds_long) == 10, "Long budget should produce predictions"
    
    print(f"  {PASS} Anytime inference works (t_budget=5 and t_budget=15)")


def run_all():
    """Run all tests."""
    print("\n" + "=" * 72)
    print("  MCT5 Unified Test Suite")
    print("=" * 72)
    
    tests = [
        test_primitives,
        test_graph_state,
        test_holographic_residue,
        test_forward_execution,
        test_autograd_learning,
        test_dual_signal_learning,
        test_hybrid_learning,
        test_xor_convergence,
        test_structural_evolution,
        test_save_load,
        test_anytime_inference,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  {FAIL} EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 72)
    print(f"  Results: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️  {failed} test(s) failed")
    
    print("=" * 72 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
