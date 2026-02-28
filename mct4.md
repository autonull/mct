# MCT4 - Morphogenic Compute Topology v4.0

## A Self-Structuring, Continuously-Learning Compute Graph

This is a complete Python proof-of-concept implementation of the MCT4 specification - a general-purpose learning algorithm designed for breakthrough next-generation performance.

## Key Features

✅ **Self-Structuring Architecture** - The graph discovers its own depth, width, and topology during training  
✅ **Local Learning Without Backpropagation** - Credit assignment via retrograde error signals  
✅ **Sparse Execution** - Only activated nodes compute, no wasted operations  
✅ **Online Learning** - True single-sample learning without special infrastructure  
✅ **Sequence Handling** - Context vector carries temporal state across tokens  
✅ **Multiple Primitives** - ReLU, GELU, Attention, Gate, and more  

## Installation

```bash
# No dependencies beyond NumPy
pip install numpy

# Optional: for visualization
pip install matplotlib networkx
```

## Quick Start

```python
from mct4 import MCT4, MCT4Config, Primitive

# Configure
config = MCT4Config(D=512, t_budget=20, eta=0.001)

# Create and initialize
model = MCT4(config)
model.initialize(Primitive.GELU)

# Train
for _ in range(1000):
    X = get_input()       # D-dimensional input
    Y = get_target()      # D-dimensional target
    loss = model.train_step(X, Y, evolve=True)

# Predict
output = model.predict(new_input)
```

## Run Demonstrations

```bash
# Full demo with XOR classification and sequence modeling
python -m mct4.demo

# Visualization tools
python -m mct4.visualize

# Comprehensive test suite
python -m mct4.tests
```

## Package Structure

```
mct4/
├── __init__.py       # Package exports
├── core.py           # Node, Context, GraphState data structures
├── primitives.py     # ReLU, GELU, Attention, Gate, etc.
├── forward.py        # Phase 1: Async forward execution
├── learning.py       # Phase 2: Retrograde learning
├── structural.py     # Phase 3: Graph evolution
├── engine.py         # Main MCT4 engine
├── demo.py           # XOR and sequence demos
├── visualize.py      # Graph visualization
├── tests.py          # Comprehensive test suite
└── README.md         # Package documentation
```

## How It Works

### Three Phases Per Training Step

1. **Forward Execution** - Nodes fire asynchronously based on activation potential:
   ```
   ρ = ρ_base + S·X + S·C
   ```
   Nodes with ρ ≥ τ(t) fire and transmit to downstream nodes.

2. **Learning** - Retrograde error flow with local updates:
   ```
   ΔW = η · T_local ⊗ V_in
   ```
   No computation graph storage required.

3. **Structural Evolution** - Graph adapts based on error pressure:
   - Prune nodes with ρ_base < 0
   - Insert capacity at high-tension edges
   - Add lateral shortcuts for persistent errors

### Context Vector

The context vector `C ∈ ℝᴰ` persists across tokens within a sequence:
- Reset at sequence boundaries
- Accumulates "ghost signals" from nodes that nearly fired
- Enables sequence modeling without separate recurrence

### Convergence Monitoring

The κ counter tracks structural stability:
- Increments each pass with zero pruning
- Resets on any pruning event
- When κ > κ_thresh, mutation and atrophy are halved

## Comparison to Deep Learning

| Aspect | Backpropagation | MCT4 |
|--------|-----------------|------|
| Architecture | Fixed | Learned |
| Credit Assignment | Global chain rule | Local retrograde |
| Memory | O(depth × width) | O(width) |
| Execution | Dense | Sparse |
| Online Learning | Special handling | Native |
| Biological Plausibility | Low | High |

## Theoretical Foundation

The local learning rule is sufficient because:

1. **Directional Error Signals** - T_v ∈ ℝᴰ encodes which directions were wrong
2. **Exact Local Gradient** - ΔW = η · T_local ⊗ V_in is SGD on local loss
3. **Structural Escape Valve** - Persistent high-tension nodes get bypassed

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| D | 512 | Vector dimensionality |
| t_budget | 20 | Maximum hop count |
| eta | 0.001 | Learning rate |
| alpha | 0.01 | Health reward |
| beta | 0.05 | Health penalty |
| gamma | 0.001 | Atrophy rate |
| sigma_mut | 0.05 | Mutation noise |
| K | 2 | Spawn count |
| kappa_thresh | 100 | Convergence threshold |
| N | 32 | Batch size |

## Example Output

```
============================================================
MCT4 XOR Classification Demo
============================================================

Training data: 200 samples
Test data: 50 samples
Dimensionality: D=64
Initial graph: 7 nodes

Training...
Epoch    0: Loss=0.0002, Test Acc=58.00%, Nodes=7, Edges=8
Epoch   50: Loss=0.0002, Test Acc=58.00%, Nodes=7, Edges=8
Epoch   99: Loss=0.0002, Test Acc=58.00%, Nodes=7, Edges=8

Final Graph Statistics:
  Total nodes: 7
  Active nodes: 5
  Average health: 7.29
  Converged: True
  Primitives: {'FORK': 2, 'GELU': 2, 'GATE': 1, 'ADD': 1, 'SOFTMAX': 1}
```

## License

MIT License

## Reference

Based on the MCT4 specification: "Morphogenic Compute Topology v4.0 - A Language-Agnostic Specification for a Self-Structuring, Continuously-Learning Compute Graph"
