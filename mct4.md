Morphogenic Compute Topology v4.0 (MCT4)
A Language-Agnostic Specification for a Self-Structuring, Continuously-Learning Compute Graph
1. The Central Argument
Gradient-based deep learning dominates not because backpropagation is the best learning algorithm, but because it was the first to combine sufficient expressiveness (deep nonlinear composition), efficient hardware mapping (batched matrix multiplication on GPUs), and a reliable enough credit assignment signal (chain rule) to make large-scale training tractable. Each of these three pillars has exploitable weaknesses:
Expressiveness: Fixed architectures are chosen before seeing data. The structure is a hyperparameter, not a learned variable.
Hardware mapping: Dense matrix operations waste compute on near-zero activations. Sparse biological neural systems perform orders of magnitude more operations per watt.
Credit assignment: Backpropagation requires locking all layer activations in memory until the backward pass completes. This is biologically implausible, prevents true online learning, and creates a hard memory scaling wall.
MCT4 attacks all three. It learns its own structure, executes sparsely, and assigns credit locally without storing a computation graph. The tradeoff is not capability — it is familiarity. MCT4 requires different initialization intuitions and different infrastructure than PyTorch. The spec below is complete and self-contained.
2. Global State
2.1 Vector Dimensionality (D)
All node inputs, outputs, weight matrices, and the context vector operate at dimensionality D (e.g., 512 for medium tasks, 2048 for large). This is the single most important architectural constant.
2.2 Context Vector (C)
A real-valued vector C ∈ ℝᴰ, initialized to zero at the start of each sequence (not each token). It persists across forward passes within a sequence, functioning as a learned routing memory.
Each node that fails to fire contributes its near-miss potential as a ghost signal:

```
C ← C · decay_c + (ρᵢ / D) · Sᵢ
```

decay_c ∈ (0,1) (e.g., 0.95). Ghost signals decay naturally. C is not reset between tokens in a sequence — it carries forward temporal context. It resets only at sequence boundaries. This is the mechanism by which MCT4 handles sequential data without a separate recurrence architecture.
2.3 Latency Clock (t)
A monotonic hop counter reset at the start of every forward pass. Budget: t_budget (e.g., 20 hops).
Dynamic threshold:

```
τ(t) = exp(λ_τ · (t − t_budget))
```

As t → t_budget, τ → ∞, collapsing all routing to the shortest available output path. MCT4 is an anytime algorithm: interrupted at any point, it emits the best current result.
2.4 Convergence Monitor (κ)
A global counter incremented each pass with zero pruning events, reset on any pruning. When κ > κ_thresh, mutation noise and atrophy are halved. The graph shifts from structural exploration to parameter refinement. This is the equivalent of learning rate scheduling in gradient descent, but driven by structural stability rather than epoch count.
2.5 Batch State
MCT4 supports parallel execution of N samples through the same graph. Each node maintains N inbox slots and accumulates N output vectors. Health and weight updates aggregate over the batch after all N forward passes complete. This provides the same variance reduction benefits as mini-batch gradient descent. Batch size N=1 is valid and corresponds to true online learning — a regime inaccessible to most gradient-based systems without significant engineering overhead.
3. The Node
Each node i is an asynchronous, self-contained processing unit with full parametric capacity.
Field Type Description id int Unique identifier S ℝᴰ Routing signature: geometric embedding used for activation potential ρ_base ℝ Health scalar: survival fitness and routing priority W ℝᴰˣᴰ Learnable weight matrix, initialized to identity × ε P enum Primitive nonlinearity (Section 7) inbox map sender_id → [(vector, time_arrived)] — N slots for batched execution edges_out list Target node IDs steps_idle int Passes since last firing tension_trace ℝ Exponential moving average of ‖T‖ observed at this node
The weight matrix W ∈ ℝᴰˣᴰ gives each node the representational power of a full linear layer. This is the critical upgrade from a bias vector. A node with W initialized to identity and a nonlinear primitive P is equivalent to one layer of a residual network. The full MCT4 graph is therefore at least as expressive as a deep residual network of comparable depth, while being able to grow beyond it.
Memory note: For large D, W can be factored as W = A Bᵀ where A, B ∈ ℝᴰˣʳ for rank r << D (e.g., r = 64). This reduces per-node memory from D² to 2Dr and learning updates to rank-1 outer products — equivalent to LoRA-style factored adaptation. Full-rank and low-rank modes are both valid; choose based on resource constraints.
4. Phase 1 — Forward Execution
For input X ∈ ℝᴰ, execute asynchronously via a priority queue ordered by hop count. Halt when any output node fires or t > t_budget.
Step 1: Activation Potential

```
ρᵢ = ρ_base,i + dot(Sᵢ, X) + dot(Sᵢ, C)
```

The input X drives content-based routing. The context C biases routing based on accumulated near-miss history — nodes that were close to firing on recent tokens are easier to activate now. Together they implement a form of dynamic attention over the graph without an explicit attention mechanism.
Step 2: Ghost Contribution (Fail to Fire)
If ρᵢ < τ(t):

```
C ← C · decay_c + (ρᵢ / D) · Sᵢ
```

Step 3: Inbox Processing (Fire)
If ρᵢ ≥ τ(t), process inbox:

1. Decay each waiting vector: V_in ← V_in · exp(−λ_async · (t − time_arrived))

2. Arity check: if >50% of required input ports are filled, proceed. Zero-fill missing ports.

3. Aggregate N batch slots independently.

Step 4: Execution
Apply weight matrix, then primitive, then route:

```
V_out = P(W · V_in)
```

The weight matrix transforms before the nonlinearity. This ordering (linear → nonlinear) matches standard neural layer convention and ensures W operates on the full input signal before compression. Transmit V_out to all edges_out. Reset steps_idle to 0. Clear inbox.
5. Phase 2 — Learning
MCT4 learning is local, online, and does not require storing a computation graph. There is no backward pass in the backpropagation sense — there is a retrograde error signal that does the same work without the memory overhead.
Step 1: Tension

```
T_v = (Y* − Y) / √D         ← normalized error direction
‖T‖ = MSE(Y*, Y) ∈ [0,1]   ← scalar error magnitude
```

For batch execution, average T_v and ‖T‖ over the N samples.
Step 2: Retrograde Flow
T_v propagates in reverse topological order up the active path. At each multi-input node, blame is partitioned by incoming signal magnitude:

```
w_blame,j = ‖V_in,j‖₂ / (Σ‖V_in‖₂ + ε)
```

The attenuated tension signal reaching node i is:

```
T_local,i = T_v · w_blame,i · (1 − tension_trace_i · 0.5)
```

The tension_trace term implements a local learning rate adaptation: nodes that have been consistently wrong attenuate their update step, equivalent to RMSProp-style variance normalization without global state.
Step 3: Weight Matrix Update
For each active node i, update W via the outer product of the error signal and the pre-activation input:

```
ΔW = η · T_local,i ⊗ V_in,i      ← rank-1 update
W_i ← W_i + ΔW
```

This is not a heuristic. It is the exact gradient of the local squared error ‖T_local,i − W_i V_in,i‖² with respect to W_i, equivalent to one step of stochastic gradient descent on the local loss. The retrograde flow approximates the chain rule by passing the error signal upstream; each node performs its exact local gradient step given what it receives. The approximation relative to full backpropagation is in the fidelity of the upstream signal, not in the local update rule itself.
For low-rank factored mode: A ← A + η · T_local,i ⊗ (Bᵀ V_in,i) and B ← B + η · V_in,i ⊗ (Aᵀ T_local,i).
After update, clamp ‖W‖_F ≤ W_max (e.g., W_max = √D) via spectral rescaling if exceeded. This prevents weight explosion without element-wise clamping, which would distort learned directions.
Step 4: Tension Trace Update

```
tension_trace_i ← 0.9 · tension_trace_i + 0.1 · ‖T_local,i‖
```

Step 5: Health Update

```
Δρ = α · (1 − ‖T‖) − β · (1 + ‖T‖²) · ‖T‖ · w_blame,i
ρ_base,i ← ρ_base,i + Δρ
```

Health and weight updates are decoupled. Health controls structural participation; W controls what is computed. A node can have learned excellent weights but low health (dormant but recoverable) or high health but poor weights (fires often but is still learning). These two axes of node state are independent.
Step 6: Atrophy
For all nodes:

```
steps_idle ← steps_idle + 1
if steps_idle > 50:
    ρ_base,i ← ρ_base,i − γ · steps_idle
```

6. Phase 3 — Structural Evolution
Structure evolves continuously. There is no separate architecture search phase — MCT4 discovers its own depth, width, and topology during training.
6.1 Pruning
Any node with ρ_base < 0 is deleted. All referencing edges are severed.
6.2 Capacity Insertion
When a node is pruned, new capacity is inserted where error pressure is highest, not where health is highest. This is the key structural principle: growth is driven by failure, not success.
Procedure:

1. Among all active edges (u → v) in the last pass, find the one with the highest attributed tension magnitude from retrograde flow.

2. Spawn K new nodes (default K=2).

3. Each new node inherits W_new = W_u + 𝒩(0, σ_mut · I) and S_new = S_u + 𝒩(0, σ_mut). Inheriting the upstream weight matrix gives new nodes a working starting point rather than random initialization, drastically shortening their learning warmup.

4. With 20% probability, assign a randomly different primitive to promote functional diversity.

5. Wire: u → new → v. Remove u → v.

6. Incremental DFS acyclicity check. On cycle detection, skip this spawn (rare with DAG insertion).

7. Initialize ρ_base,new = ρ_base,u · 0.5, tension_trace_new = tension_trace_u.

6.3 Lateral Wiring
Beyond depth insertion, MCT4 can grow laterally. Each pass, if a node i has tension_trace_i > τ_lateral (e.g., 0.3) for more than 20 consecutive passes, it spawns one additional outbound edge to a random downstream node that it is not already connected to (DAG-preserving). This creates shortcut paths for persistent error signals, equivalent to the skip connections that made ResNet trainable — but discovered dynamically rather than designed in.
6.4 Convergence Dampening
When κ > κ_thresh: σ_mut ← σ_mut · 0.5, γ ← γ · 0.5. The graph crystallizes. Dampening lifts on the next pruning event.
7. Primitive Operator Set
All operators output ℝᴰ. Post-W application, the primitive is the nonlinearity. Choosing a primitive is choosing the activation function — but nodes can mutate primitives, so the graph searches over activation functions as part of training.
Unary (1 input):
Op Definition Role ReLU(X) max(0, X) elementwise Sparse activation, fast Tanh(X) tanh(X) elementwise Bounded nonlinearity GELU(X) X · Φ(X) elementwise Smooth gating, transformer-grade Softmax(X) standard softmax Probability / attention normalization L2Norm(X) X / ‖X‖₂ Directional normalization Fork(X) Pass-through with temporary fan-out to ≤F downstream nodes Exploratory branching
Binary/N-ary (2+ inputs; aggregate >2 via mean before op):
Op Definition Role Add(X,Y) X + Y Residual connection Attention(X,Y) softmax(XYᵀ/√D) · Y, pooled to D Full attention operation Gate(X,Y) X ⊙ σ(Y) Multiplicative gating Concat(X,Y) concat then mean-pool to D Feature fusion
The Attention primitive makes a single node capable of implementing a full attention head. A cluster of nodes with Attention primitives wired in parallel is equivalent to multi-head attention — discovered structurally if the task requires it.
8. Handling Standard Deep Learning Tasks
8.1 Supervised Classification
Input node receives X. Output node emits logit vector Y ∈ ℝᶜ (or ℝᴰ projected to C classes by a fixed linear readout). Target Y* is the one-hot label. Standard cross-entropy tension: T_v = softmax(Y) − Y*. The graph learns depth and width appropriate to the task.
8.2 Sequence Modeling (Language, Audio, Time Series)
Process token xₜ as one forward pass. Context C carries state between tokens within a sequence — it is the recurrent memory. The graph does not need a separate architecture for sequences; the context vector is the recurrence. For autoregressive generation, the output node's emission at step t becomes the input at step t+1.
8.3 Vision
Patch-embed images (e.g., 16×16 patches → D-dimensional vectors) and process each patch as a sequence token. The graph develops spatial routing structure through lateral wiring and capacity insertion driven by spatial error patterns.
8.4 Batching and Scale
Run N samples simultaneously through the graph (Section 2.5). Weight updates aggregate over the batch. There is no theoretical limit on N other than memory. The graph itself can grow unboundedly via capacity insertion — there is no D + M cap. Graph size is regulated entirely by atrophy: nodes that don't contribute get pruned. Scale is earned, not allocated.
9. Why the Local Learning Rule Is Sufficient
The central objection to any non-backprop learning rule is credit assignment: without the chain rule, how does an early node know its contribution to the final error? MCT4's answer has three components:
The retrograde signal carries directional information. T_v is a vector in ℝᴰ, not a scalar. It encodes which directions in output space were wrong, not just how wrong. Each node that receives this signal updates its weights to reduce its contribution in those directions. This is richer than scalar error signals used in many proposed backprop alternatives.
The outer product update is the exact local gradient. For a node computing V_out = P(W V_in), the gradient of ‖T_local − V_out‖² with respect to W is T_local ⊗ V_in (ignoring the nonlinearity's Jacobian, which is element-wise bounded). The update rule is not a heuristic approximation — it is stochastic gradient descent on the local objective. The approximation is in treating T_local as the target rather than the exact upstream gradient, which is the same approximation made by target propagation (Lee et al., 2015) — a theoretically grounded alternative to backpropagation.
Structural evolution compensates for misalignment. If a node's weights cannot reduce its tension trace below threshold via learning alone, the graph grows a bypass route around it. Persistent high-tension nodes eventually become structurally isolated and pruned, replaced by new capacity with fresh initialization near the problem locus. The graph has a structural escape valve for learning stalls that gradient descent lacks entirely.
10. Hyperparameters
Parameter Symbol Default Description Vector dim D 512 Uniform dimensionality Context decay decay_c 0.95 Ghost signal half-life per step Catalysis α 0.01 Health reward rate Solvent β 0.05 Health penalty rate Atrophy γ 0.001 Idle decay per step Learning rate η 0.001 Weight matrix update step size Weight max norm W_max √D Spectral norm ceiling for W Rank (factored) r 64 Low-rank factor dimension Mutation noise σ_mut 0.05 Spawn perturbation std dev Async decay λ_async 0.2 Inbox signal decay rate Tau steepness λ_τ 0.1 Latency kill-switch slope Spawn count K 2 New nodes per pruning event Fork fan-out F 2 Max temporary Fork edges Lateral tension threshold τ_lateral 0.3 Tension EMA to trigger wiring growth Convergence threshold κ_thresh 100 Passes before dampening Batch size N 32 Samples per weight update
11. Complexity and Hardware Mapping
Per forward pass: O(k · h · N_active) where k = mean out-degree, h = mean path depth ≤ t_budget, N_active = nodes that fire.
Per weight update: O(D² · N_active) for full-rank, O(Dr · N_active) for low-rank factored.
GPU mapping: Nodes that fire in the same hop are independent and can be batched into a single matmul. Represent each hop as a sparse matrix-vector product over the active frontier. This maps to cusparse or torch.sparse operations. The priority queue determines hop assignment; nodes at the same hop depth execute in parallel. With h=20 hops and k=4, a graph of 512 nodes executes in 20 sparse matmul rounds — comparable to a 20-layer transformer block in operation count, with substantially lower memory due to sparsity and absence of KV cache.
Memory: No activation storage for the backward pass. MCT4 requires only current inbox values and the context vector C during forward execution. Total inference memory is O(D · N_active) plus O(D²) per node for weights. A graph of 1000 nodes at D=512 requires ~1GB for full-rank weights — manageable on a single GPU.
12. Implementation Bootstrap
Start with the minimal viable graph: two input nodes (one for content X, one for context C), four intermediate nodes (two GELU, one Gate, one Add), one output node. Wire them as a shallow DAG. Let structural evolution grow from there. Do not pre-design depth or width. The task will impose structure through the tension signal.
Core data structures:

* Node registry: Hash map id → Node.

* Adjacency: CSR sparse list per node.

* Execution queue: Min-heap on hop count, with per-hop batch aggregation.

* Context: Single shared ℝᴰ vector with atomic updates under multi-threading.

* Batch buffer: Per-node N × D matrix of inbox slots.

Implementation languages in order of recommendation: Rust (memory safety + concurrency), C++ with BLAS, Python with torch.sparse (prototyping only — Python's GIL makes true asynchrony require multiprocessing).

