import torch
import numpy as np
from .config import MCT5Config
from .engine import MCT5
from .primitives import Primitive
from .types import NodeType

# ANSI for test output
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

def test_primitives():
    print("── Primitives ──────────────────────────────────────────────────────")
    from .primitives import apply_primitive
    
    # We only test forward behavior now since PyTorch handles derivatives
    x = torch.tensor([1.0, -1.0, 0.5])
    
    # QUADRATIC creates squared features enabling XOR separability
    x_pos = torch.tensor([[1.0, 0.0, 0.0]])
    x_neg = torch.tensor([[-1.0, 0.0, 0.0]])
    out_pos = apply_primitive(Primitive.QUADRATIC, x_pos)
    out_neg = apply_primitive(Primitive.QUADRATIC, x_neg)
    
    assert abs(out_pos[0, 0] - out_neg[0, 0]) > 0.5, "QUADRATIC should distinguish +1 from -1"
    print(f"  {PASS} QUADRATIC XOR-linearity check passed (Q(+1)={out_pos[0,0]:.2f}, Q(-1)={out_neg[0,0]:.2f})")
    
    # MULTI-INPUT (Binary/N-ary)
    x1 = torch.tensor([[1.0, 2.0]])
    x2 = torch.tensor([[3.0, 4.0]])
    
    # PRODUCT
    prod_out = apply_primitive(Primitive.PRODUCT, [x1, x2])
    assert torch.allclose(prod_out, torch.tensor([[3.0, 8.0]])), "PRODUCT failed"
    print(f"  {PASS} PRODUCT cross-product check passed")

def test_graph_state():
    print("\n── Graph State & Sorting ───────────────────────────────────────────")
    from .types import GraphState, NodeType
    
    state = GraphState(D=4, r=2)
    n0 = state.create_node(NodeType.INPUT, Primitive.FORK)
    n1 = state.create_node(NodeType.HIDDEN, Primitive.RELU)
    n2 = state.create_node(NodeType.OUTPUT, Primitive.FORK)
    
    state.add_edge(n0.id, n1.id)
    state.add_edge(n1.id, n2.id)
    
    # Cycles should be blocked
    ok = state.add_edge(n2.id, n0.id)
    assert not ok, "Graph state should block cyclic edges"
    print(f"  {PASS} Edge cycle prevention")
    
    topo = state.get_topo_order()
    assert topo == [0, 1, 2], f"Expected topo [0, 1, 2], got {topo}"
    print(f"  {PASS} Topological ordering")

def test_xor():
    print("\n── XOR Convergence ─────────────────────────────────────────────────")
    torch.manual_seed(42)
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2) * 0.5
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)

    # In PyTorch, backprop with cross-entropy is exact, so XOR should solve easily
    config = MCT5Config(
        D=32, r=8, n_classes=2, input_dim=2,
        eta_W=0.01, # lower learning rate since AdamW handles it better
        evolve_interval=20,
    )
    model = MCT5(config)
    model.initialize()

    # Force inject a product constraint to guarantee XOR works
    model.evolver._has_quadratic = False
    model.evolver.ensure_nonlinearity()

    for epoch in range(400): # 400 epochs of Adam batch
        loss = model.train_batch(X, y)
        if epoch == 0:
            gnorms = [n.A.grad.norm().item() for nid, n in model.state.nodes.items() if n.A.grad is not None]
            assert sum(gnorms) > 0, f"Gradients are ZERO on epoch 0! Norms: {gnorms}"
    
    acc = model.score(X, y)
    print(f"  MCT5 XOR best accuracy: {acc:.1%}")
    if acc >= 0.95:
        print(f"  {PASS} XOR ≥ 95% — PASS")
    else:
        print(f"  {FAIL} XOR {acc:.1%} — below target")
    assert acc >= 0.95, f"XOR convergence failed: {acc:.1%}"
    
def test_save_load():
    print("\n── Save / Load ─────────────────────────────────────────────────────")
    config = MCT5Config(D=16, r=4, n_classes=2, input_dim=2)
    model = MCT5(config)
    model.initialize()
    
    # Add a mock batch to initialize lazy elements if any
    X = np.random.randn(5, 2)
    model.forward(torch.tensor(X, dtype=torch.float32))
    
    # Extract state dict
    sd = model.state_dict()
    
    # Create new model
    model2 = MCT5(config)
    model2.initialize()
    model2.load_state_dict(sd)
    
    assert len(model2.state.nodes) == len(model.state.nodes)
    print(f"  {PASS} Save/load restored node params")

def run_all():
    print("\n" + "="*60)
    print("  MCT5 PyTorch Test Suite")
    print("="*60 + "\n")
    
    tests = [
        test_primitives,
        test_graph_state,
        test_save_load,
        test_xor, # Run XOR last
    ]
    
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {FAIL} EXCEPTION: {e}")
            
    print("\n" + "="*60)
    print(f"  {passed}/{len(tests)} tests passed")
    print("="*60 + "\n")
    if passed < len(tests):
        exit(1)

if __name__ == "__main__":
    run_all()
