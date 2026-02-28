# SPECIFICATION: Morphogenic Compute Topology (MCT) v2.0
**Powered by Tension-Dissipative Morphogenesis (TDM)**
## 1. Abstract
This document defines the complete, language-agnostic specification for a continuous-learning, dynamically morphing computational topology. MCT replaces fixed-architecture Neural Networks and Backpropagation with a fluid, latency-bounded Directed Acyclic Graph (DAG). Learning is achieved via strictly local, physics-inspired "Tension" shockwaves and counterfactual structural evolution (Morphogenesis). It is designed to scale dynamically, process high-dimensional vector spaces (vision, NLP, control), and execute with $O(k)$ sparse complexity.
---
## 2. Core Global State
The system operates on a uniform vector dimensionality, $D$ (e.g., $D = 512$).
### 2.1 The Holographic Residue ($R_t$)
The global context memory. It anticipates routing needs by maintaining a compressed, phase-shifted history of ghost signals (nodes that almost fired).
* **Vector State:** $R_t \in \mathbb{C}^D$ (A vector of complex numbers).
* **Orthogonal Bases ($B$):** A static matrix where each row $B_i \in \mathbb{R}^D$ is a Gram-Schmidt orthogonalized vector assigned uniquely to Node $i$ upon creation.
* **Capacity Limit:** If $||R_t||_2 > \Phi_{max}$, the lowest magnitude components are aggressively pruned to prevent noise washout.
### 2.2 The Latency Clock ($t$)
A monotonic counter (or hardware timer) resetting at the start of every forward pass.
* **Budget ($t_{budget}$):** The hard limit for execution (e.g., $80ms$ or a max hop-count of $15$).
* **Dynamic Decay Curve ($\tau$):** The activation threshold for any node.
    $$ \tau(t) = \exp(\lambda_{tau} \cdot (t - t_{budget})) $$
    *(As time runs out, $\tau$ approaches infinity, forcing the network to halt deep branching and output immediately).*
---
## 3. The Node: Primitive Compute Unit
Each Node $i$ in the graph acts as an asynchronous, self-contained processing unit.
**Data Structure:**
* `ID`: Unique integer.
* `State_Vector` ($S_i \in \mathbb{R}^D$): The geometric routing signature.
* `Rho_Base` ($\rho_{base, i} \in \mathbb{R}$): The historical health/catalysis of the node.
* `Primitive_Op` ($P_i$): A specific mathematical operation (see Section 7).
* `Inbox`: A mapping of `Sender_ID` $\rightarrow$ `(Signal_Vector, Time_Arrived)`.
* `Outbound_Edges`: List of target Node IDs.
* `Steps_Idle`: Integer tracking iterations since last firing.
---
## 4. Phase 1: Forward Morphogenesis (Execution)
For a given input vector $X \in \mathbb{R}^D$, the system executes asynchronously until an output node fires.
### Step 1: Potential Calculation
Every node evaluates its Activation Potential ($\rho_i$):
$$ \rho_i = \rho_{base, i} + \text{Real}(\text{Dot}(S_i, R_t)) $$
### Step 2: Ghost Binding (Failure to Fire)
If $\rho_i < \tau(t)$, the node remains dormant. Its potential is bound to the Holographic Residue as a "Ghost" signal:
$$ \theta_{phase} = t_{elapsed} \cdot \omega $$
$$ R_{t} \leftarrow R_{t} + \rho_i \cdot (B_i \cdot e^{i \theta_{phase}}) $$
*(Older signals rotate in the complex plane, allowing high-density memory superposition).*
### Step 3: Asynchronous Reception & Decay
If $\rho_i \ge \tau(t)$, the node is active. It processes its `Inbox`:
1. **Exponential Decay:** Every vector in the inbox decays based on time waited:
    $$ V_{in} \leftarrow V_{in} \cdot \exp(-\lambda_{async} \cdot (t - Time\_Arrived)) $$
2. **Arity Check:** If $>50\%$ of the Primitive $P_i$'s required ports are filled, it executes. Missing ports are zero-filled.
### Step 4: Execution & Routing
The node applies $P_i$ to its inbox vectors, producing Output Vector $V_{out}$.
* $V_{out}$ is transmitted to all `Outbound_Edges`.
* The node's `Steps_Idle` is reset to 0.
* `Inbox` is cleared.
---
## 5. Phase 2: Tension-Dissipative Morphogenesis (Learning)
When the graph produces a final Prediction Vector ($Y$) against a Target Vector ($Y^*$), learning occurs exclusively along the **Active Path** (the exact sub-graph of nodes that fired).
### Step 1: Tension Generation
Calculate the normalized Tension Vector ($T_v$) and its scalar magnitude ($||T||$):
$$ T_v = \frac{Y^* - Y}{\sqrt{D}} $$
$$ ||T|| = \text{MSE}(Y^*, Y) \in [0, 1] $$
### Step 2: Retrograde Flow & Proportional Attribution
The Shockwave $T_v$ travels in reverse topological order up the Active Path.
For multi-port (binary/N-ary) nodes, blame is split proportionally based on the L2-norm of the incoming signals:
$$ W_{blame, j} = \frac{||V_{in, j}||_2}{\sum ||V_{in}||_2 + \epsilon} $$
### Step 3: The Update Rule (Dynamic Beta Scaling)
For each active node $i$, update its health ($\rho_{base, i}$).
*Let $\alpha$ be the Catalysis rate, $\beta_{base}$ be the Solvent rate.*
$$ \text{Solvent\_Multiplier} = 1.0 + ||T||^2 $$
$$ \Delta \rho = \underbrace{\alpha (1 - ||T||)}_{\text{Reward}} - \underbrace{\left( \beta_{base} \cdot \text{Solvent\_Multiplier} \cdot ||T|| \cdot W_{blame, i} \right)}_{\text{Punishment}} $$
$$ \rho_{base, i} \leftarrow \rho_{base, i} + \Delta \rho $$
### Step 4: Atrophy (Dormancy Penalty)
For all nodes in the global graph (active or inactive):
`Steps_Idle` increments by 1.
If `Steps_Idle` > 50:
$$ \rho_{base, i} \leftarrow \rho_{base, i} - (\gamma_{atrophy} \cdot \text{Steps\_Idle}) $$
---
## 6. Phase 3: Structural Evolution (Mutation)
This phase executes instantly for any node whose $\rho_{base}$ drops below $0.0$.
### Step 1: Lysis (Death)
The node is deleted from memory. All inbound and outbound edges referencing it are severed.
### Step 2: Viral Resonance (Spawning)
The energy of the dissolved node spawns $K$ (e.g., $K=3$) new nodes in the void to explore alternatives.
1. **Target Selection:** Find the Node $M$ currently possessing the highest $\rho_{base}$ in the global graph.
2. **Inheritance:** The new nodes copy $M$'s State Vector ($S_M$) and Primitive ($P_M$).
3. **Gaussian Perturbation:** Add noise to the inherited state to promote diversity:
    $$ S_{new} = S_M + \mathcal{N}(0, \sigma_{mut}) $$
4. **Primitive Mutation:** With a $20\%$ probability, randomly swap the Primitive $P_{new}$ to a different operation.
5. **Wiring:** Randomly connect the new nodes to the inputs and outputs of the dissolved node's local neighborhood.
---
## 7. Primitive Operator Set ($P$)
To ensure general-purpose, non-linear compute capabilities, the system draws from this foundational vectorized instruction set:
**Unary Operators (1 Input):**
* `Tanh(X)`: Non-linear squash.
* `Softmax(X)`: Probability distribution.
* `L2Norm(X)`: Vector stabilization.
* `Fork(X)`: Duplicates $X$ to $N$ asynchronous downstream paths.
**Binary/N-ary Operators (2+ Inputs):**
* `Add(X, Y)`: Residual combination.
* `MatMul(X, Y)`: Cross-dimensional projection (enables attention-like feature crossing).
* `Concat(X, Y)`: Dimensional stacking (requires dimensionality reduction step post-op).
* `Gate(X, Y)`: Output is $X \odot \sigma(Y)$ (element-wise gating).
---
## 8. System Hyperparameters (Standard Init)
For implementation, the following constants yield stable crystallization on complex datasets (e.g., Vectorized Interleaved Chaos/XOR):
| Parameter | Symbol | Suggested Value | Description |
| :--- | :---: | :---: | :--- |
| **Vector Dim** | $D$ | `512` | Dimensionality of all tensors. |
| **Catalysis Rate** | $\alpha$ | `0.01` | Reward scaling for perfect predictions. |
| **Base Solvent Rate** | $\beta_{base}$ | `0.05` | Base punishment scaling for errors. |
| **Atrophy Rate** | $\gamma_{atrophy}$ | `0.001` | Decay per step for dormant nodes. |
| **Mutation Noise**| $\sigma_{mut}$ | `0.1` | Standard deviation for Gaussian Resonance. |
| **Async Decay** | $\lambda_{async}$ | `0.2` | Decay rate for waiting port signals. |
| **Tau Multiplier**| $\lambda_{tau}$ | `0.1` | Steepness of the latency kill-switch curve. |
| **Phase Velocity**| $\omega$ | `0.05` | Radian rotation per step for ghost residues. |
| **Resonance Count**| $K$ | `3` | Number of nodes spawned during Lysis. |
---
**End of Specification.**
*Implementation Note: To achieve maximum FLOP efficiency, the graph adjacency lists should be implemented using sparse matrix representations (CSR/CSC) or safe concurrent pointers (e.g., Rust `Arc<RwLock<Node>>`), while primitive operations should be delegated to highly optimized BLAS/SIMD hardware intrinsics.*
 
## Analysis of Morphogenic Compute Topology (MCT) v2.0
This is a creative and ambitious design that blends ideas from several fields — spiking neural networks, evolutionary computing, physics-inspired learning, and dynamic graph topology. Here's a critical breakdown:
---
### What's Genuinely Interesting
**The Holographic Residue** is the most novel concept here. Using a complex-valued superposition vector where dormant ("ghost") nodes encode their potential with a phase rotation is elegant. It borrows from holographic reduced representations and allows the system to maintain a kind of soft memory of near-misses without storing them explicitly. The orthogonal basis assignment per node is a smart choice to prevent destructive interference. This is theoretically sound and underexplored in mainstream architectures.
**The Latency Kill-Switch** via $\tau(t)$ is pragmatically valuable. Making the activation threshold grow exponentially as the time budget expires forces the network toward fast, confident paths under pressure. This is reminiscent of anytime algorithms and gives the system graceful degradation properties that fixed architectures completely lack.
**Tension-Dissipative Learning** is a biologically plausible alternative to backpropagation in that it's strictly local to the active path and avoids the global gradient calculation. The proportional blame attribution via L2-norm of inputs is a reasonable heuristic that avoids vanishing/exploding gradient dynamics by design.
**Structural Evolution via Lysis/Viral Resonance** is essentially a continuous, online version of NEAT (NeuroEvolution of Augmenting Topologies), but triggered by health collapse rather than generational selection. Copying the fittest node and perturbing it is a sensible warm-start strategy rather than random initialization.
---
### Significant Theoretical Concerns
**The learning rule is not guaranteed to converge.** The $\Delta\rho$ update only modifies a scalar health value, not the actual computational parameters of the node. The State Vector $S_i$ (which governs routing) is never updated during learning — only during spawning via Gaussian perturbation. This means the system learns *which nodes survive*, not *what those nodes do*. Learning is entirely structural/evolutionary, not gradient-based. This is a valid paradigm, but the spec presents it as if it competes with backprop on equal footing, which it doesn't — it's closer to an evolutionary strategy with fitness-based selection, which is known to be sample-inefficient on high-dimensional problems.
**The Primitive operators have a dimensionality problem.** `MatMul(X, Y)` takes two $D$-dimensional vectors and produces... what? A scalar? A $D \times D$ matrix? The spec says it enables "attention-like feature crossing" but never specifies the output dimensionality or how it reconnects to the rest of the $D$-dimensional graph. `Concat(X, Y)` produces a $2D$-dimensional vector and the spec defers to a vague "dimensionality reduction step post-op." These are not implementation details — they're fundamental to whether the graph is well-defined at all.
**Ghost Binding has a capacity problem.** The spec acknowledges that $||R_t||$ can exceed $\Phi_{max}$ and resolves it by pruning the lowest-magnitude components. But it never specifies *when* this check happens (every step? every pass?), and aggressive pruning of $R_t$ mid-pass could cause routing instability. More importantly, orthogonality of the basis $B$ only guarantees non-interference at write time — repeated additions with phase rotations can still produce magnitude growth that's hard to predict without analysis of the specific $\omega$ and graph density.
**The DAG constraint is asserted but not enforced.** The spec calls the topology a DAG but during Viral Resonance, new nodes are "randomly connected to the inputs and outputs of the dissolved node's local neighborhood." Random wiring has no mechanism to prevent cycles, which would break the topological ordering required for the retrograde tension flow in Phase 2. This is a correctness bug, not a design choice.
---
### Practical Implementation Concerns
**Asynchronous execution with a shared mutable Holographic Residue is a race condition.** If nodes fire concurrently and all write to $R_t$ simultaneously (Step 2 of Phase 1), the result is non-deterministic without explicit locking. The spec recommends `Arc<RwLock<Node>>` for node safety but says nothing about protecting $R_t$, which is the most contended shared resource in the entire system.
**The $O(k)$ sparse complexity claim is underspecified.** $k$ is never defined. If $k$ refers to average node out-degree, then yes, routing is $O(k)$ per hop — but the number of hops, the cost of $\rho_i$ evaluation for every node every step, and the $O(N)$ atrophy update in Phase 2 Step 4 all contribute to actual complexity that could easily exceed a comparable dense network.
**The system has no stable fixed point by design.** Nodes are continuously dying and spawning, $R_t$ is continuously mutating, and there's no mechanism to detect that the graph has "converged." In practice this could mean the system perpetually churns rather than crystallizing a stable solution, especially under the atrophy pressure that penalizes any node that isn't constantly firing.
---
### Comparison to Existing Work
| MCT Concept | Closest Prior Art | Key Difference |
|---|---|---|
| Ghost Binding / Holographic Residue | Holographic Reduced Representations (Plate, 1995) | MCT applies it dynamically during inference |
| Lysis/Resonance | NEAT (Stanley & Miikkulainen, 2002) | Online, triggered by health rather than generational |
| Tension Shockwave | Contrastive Hebbian Learning | No global error signal; path-local only |
| Latency $\tau(t)$ | Anytime neural networks | Threshold-based rather than early-exit |
| Async spiking | Spiking Neural Networks (Maass, 1997) | Uses real-valued vectors, not spike timing |
MCT is best understood as a **continuous online neuroevolution system** with a novel memory mechanism, not a replacement for backprop-trained networks. The framing in the abstract oversells it.
---
### Summary Verdict
The spec is **creative and internally consistent in spirit**, but has **several correctness gaps** (DAG enforcement, MatMul output shape, concurrency on $R_t$) that would block a faithful implementation. The learning theory is the weakest part — structural evolution via health scoring is not demonstrably more capable than gradient descent, and likely far less sample-efficient. The Holographic Residue is the most original contribution and deserves more rigorous treatment.
If this is heading toward implementation, I'd prioritize resolving the dimensionality contracts for all primitives and the DAG enforcement during mutation before anything else.

