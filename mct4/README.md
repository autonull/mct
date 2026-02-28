# MCT4 Python Implementation

Morphogenic Compute Topology v4.0 (MCT4) - A self-structuring, continuously-learning compute graph.

## Overview

This is a proof-of-concept implementation of the MCT4 specification from the README.md. MCT4 is a general-purpose learning algorithm that:

- **Learns its own structure** - No fixed architecture; the graph grows and prunes based on task demands
- **Executes sparsely** - Only activated nodes compute, wasting no operations on near-zero activations
- **Assigns credit locally** - No backpropagation or computation graph storage; uses retrograde error signals

## Installation

No external dependencies required beyond NumPy. For visualization:
```bash
pip install numpy matplotlib networkx
```

## Quick Start

```python
from mct4 import MCT4, MCT4Config, Primitive

# Configure
config = MCT4Config(
    D=512,           # Vector dimensionality
    t_budget=20,     # Maximum hop count
    eta=0.001,       # Learning rate
    kappa_thresh=100 # Convergence threshold
)

# Create model
model = MCT4(config)
model.initialize(Primitive.GELU)

# Train
for _ in range(1000):
    X = get_input()      # Your input vector (D-dimensional)
    Y = get_target()     # Your target vector (D-dimensional)
    loss = model.train_step(X, Y, evolve=True)
    
# Predict
output = model.predict(new_input)
```

## Package Structure

```
mct4/
├── __init__.py       # Package exports
├── core.py           # Core data types (Node, Context, GraphState)
├── primitives.py     # Primitive operators (ReLU, GELU, Attention, etc.)
├── forward.py        # Phase 1: Forward execution
├── learning.py       # Phase 2: Learning via retrograde flow
├── structural.py     # Phase 3: Structural evolution
├── engine.py         # Main MCT4 engine integrating all phases
├── demo.py           # Demonstration on XOR and sequence tasks
└── visualize.py      # Graph visualization tools
```

## Key Concepts

### Node
Each node is a self-contained processing unit with:
- **Routing signature (S)**: Geometric embedding for activation-based routing
- **Health (ρ_base)**: Survival fitness and routing priority
- **Weight matrix (W)**: Full D×D learnable transformation
- **Primitive (P)**: Nonlinear operator (ReLU, GELU, Attention, etc.)

### Context Vector
A persistent D-dimensional vector that carries temporal context across tokens within a sequence. Nodes that fail to fire contribute "ghost signals" to the context.

### Three Phases

1. **Forward Execution**: Nodes fire asynchronously based on activation potential ρ = ρ_base + S·X + S·C
2. **Learning**: Retrograde error flow with local weight updates ΔW = η · T_local ⊗ V_in
3. **Structural Evolution**: Pruning (ρ < 0), capacity insertion at high-error edges, lateral wiring

### Primitives

**Unary**: ReLU, Tanh, GELU, Softmax, L2Norm, Fork

**Binary/N-ary**: Add, Attention, Gate, Concat

## Running Demos

```bash
# Run full demonstration
python -m mct4.demo

# Run visualization
python -m mct4.visualize
```

## API Reference

### MCT4Config
| Parameter | Default | Description |
|-----------|---------|-------------|
| D | 512 | Vector dimensionality |
| t_budget | 20 | Maximum hop count |
| eta | 0.001 | Learning rate |
| alpha | 0.01 | Health reward (catalysis) |
| beta | 0.05 | Health penalty (solvent) |
| gamma | 0.001 | Atrophy rate |
| sigma_mut | 0.05 | Mutation noise |
| K | 2 | Spawn count per pruning |
| kappa_thresh | 100 | Convergence threshold |
| N | 32 | Batch size |

### MCT4 Class

```python
model = MCT4(config)
model.initialize(Primitive.GELU)  # Create minimal graph

# Single step training
loss = model.train_step(X, Y_star, evolve=True)

# Batch training
loss = model.train_batch(X_batch, Y_batch, evolve=True)

# Inference
output = model.predict(X)

# Get statistics
stats = model.get_stats()
```

## Differences from Gradient-Based Learning

| Aspect | Backpropagation | MCT4 |
|--------|-----------------|------|
| Architecture | Fixed hyperparameter | Learned dynamically |
| Credit Assignment | Global chain rule | Local retrograde flow |
| Memory | O(depth × width) for activations | O(width) only |
| Sparsity | Dense matmul | Sparse node activation |
| Online Learning | Requires special handling | Native support |

## Theoretical Foundation

The local learning rule is sufficient because:

1. **Retrograde signal carries directional information** - T_v ∈ ℝᴰ encodes which directions were wrong
2. **Outer product update is exact local gradient** - ΔW = η · T_local ⊗ V_in is SGD on local loss
3. **Structural evolution compensates** - Persistent high-tension nodes get bypass routes or are pruned

## License

MIT License
