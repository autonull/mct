The provided document presents **Morphogenic Compute Topology 4 (MCT4)** as a refined, pragmatic evolution of the earlier MCT series. It strips away unsubstantiated hype, fixes key theoretical and practical flaws from prior versions (e.g., static routing signatures, argmax-only spawning, optional parametric learning), and delivers a clean, implementable specification for a sparse, anytime, self-structuring compute graph.

This is **not** positioned as a universal replacement for neural networks or transformers. Instead, MCT4 targets niches where:

- Hard latency bounds demand graceful degradation (anytime property via exponential threshold growth).
- Inputs benefit from sparse, conditional routing rather than dense feed-forward computation.
- Online structural adaptation (lightweight architecture search via health-based death/spawn) adds value without full NAS overhead.
- Explicit active paths offer interpretability advantages.

Below is the **complete, self-contained, language-agnostic specification for MCT4**, incorporating all details from the query document. It is ready for reference or implementation.

# Morphogenic Compute Topology 4 (MCT4)
**A Sparse, Anytime, Self-Structuring Compute Graph**

## 1. Overview
MCT4 is a directed acyclic graph (DAG) compute model operating over uniform vector dimension $D$. Nodes route and transform vectors asynchronously within a strict latency budget. Execution is anytime: partial but correct outputs are produced even under severe time pressure. Learning combines local Hebbian-style parametric updates with health-gated structural evolution.

## 2. Graph Structure
$G = (V, E)$ is a DAG with uniform vector dimension $D$.

**Node $i \in V$ fields:**
- `id`: unique integer
- `S_i ∈ ℝ^D`: routing signature (learnable)
- `W_i ∈ ℝ^D`: parameter vector (learnable, init zeros)
- `P_i`: primitive operator (from Section 7)
- `ρ_i ∈ ℝ`: health scalar (init 1.0)
- `inbox`: map sender_id → (vector ∈ ℝ^D, arrival_time)
- `idle_steps`: integer (increments each pass if node does not fire)

Edges are directed; mutations preserve acyclicity (verified via topological sort before commit).

## 3. Context Residue ($R_t$)
$R_t ∈ \mathbb{C}^D$ encodes phase-shifted superposition of near-miss activations within one forward pass.

### 3.1 Orthogonal Basis
Each node receives a unique $B_i ∈ \mathbb{R}^D$ via Gram-Schmidt at creation. Slot freed on death. Node count capped at $D + M$ (e.g. $M=64$); excess uses approximate orthogonality.

### 3.2 Ghost Injection
On failure to fire at $t_{elapsed}$:
$$
R_t \leftarrow R_t + \frac{\rho_i}{\sqrt{D}} \cdot B_i \cdot e^{i \omega t_{elapsed}}
$$

### 3.3 Norm Pruning
After pass, if $||R_t||_2 > \Phi_{\max}$ (e.g. $2\sqrt{D}$), zero lowest-magnitude components until within budget.

### 3.4 Reset
$R_t \leftarrow 0$ at start of each forward pass.

## 4. Anytime Execution
### 4.1 Latency Clock
Monotonic $t$ resets to 0 per pass; hard budget $t_{budget}$ (hops or wall time, e.g. 15 hops / 80 ms).

### 4.2 Activation Threshold
$$
\tau(t) = \exp\left(\lambda_\tau (t - t_{budget})\right)
$$
$\tau \approx 0$ early → permissive firing; $\tau \to \infty$ near budget → forces termination.

### 4.3 Activation Potential
$$
\rho_i^{(active)} = \rho_i + \operatorname{Re}\left( \langle S_i, R_t \rangle \right)
$$
Fire if $\rho_i^{(active)} \geq \tau(t)$; else ghost injection.

## 5. Forward Pass
**Step 1 – Potential & Routing**  
Process nodes in topological order. Compute $\rho_i^{(active)}$. Fire if threshold met; else ghost.

**Step 2 – Inbox Decay**  
$$
V_{in} \leftarrow V_{in} \cdot \exp\left(-\lambda_{async} (t - t_{arrived})\right)
$$

**Step 3 – Execution**  
$$
V_{out} = P_i(V_{in}) + W_i \quad (V_{out} \in \mathbb{R}^D)
$$
Transmit to outbound edges. Clear inbox, reset idle_steps.

**Step 4 – Output**  
First output node to fire → $\hat{Y}$. If none by budget, return last $V_{out}$ on active path or learned default.

## 6. Learning
### 6.1 Tension
$$
T_v = \frac{Y^* - \hat{Y}}{\sqrt{D}}, \quad |T| = \operatorname{MSE}(Y^*, \hat{Y}) \in [0,1]
$$

### 6.2 Parametric Update (Active Path, Reverse Topo Order)
$$
W_i \leftarrow \operatorname{clip}\left( W_i + \eta \cdot T_v \odot V_{in,i},\ [-1,1] \right)
$$
Multi-input blame:
$$
w_{blame,j} = \frac{||V_{in,j}||_2}{\sum_k ||V_{in,k}||_2 + \epsilon}
$$

### 6.3 Health Update (Active Path)
$$
\Delta \rho_i = \alpha(1 - |T|) - \beta \cdot (1 + |T|^2) \cdot |T| \cdot w_{blame,i}
$$
$$
\rho_i \leftarrow \rho_i + \Delta \rho_i
$$

### 6.4 Atrophy (Global)
Idle non-firers: `idle_steps += 1`. If > 50:
$$
\rho_i \leftarrow \rho_i - \gamma \cdot \text{idle_steps}
$$

### 6.5 Routing Signature Update
$$
S_i \leftarrow S_i + \eta_S \cdot T_v \cdot \operatorname{Re}(\langle S_i, R_t \rangle)
$$

## 7. Structural Evolution
### 7.1 Death (Lysis)
Remove node if $\rho_i < 0$; free basis slot.

### 7.2 Spawning (on Death)
If $|V| \geq D + M$, replace lowest-$\rho$ node instead of adding.

Spawn $K$ new nodes:

1. Donor $d$ sampled proportional to $\rho_d$ (health-weighted).
2. Copy $S_d$, $W_d$, $P_d$.
3. Perturb: $S_{new} \leftarrow S_d + \mathcal{N}(0, \sigma_{mut})$, $W_{new} \leftarrow W_d + \mathcal{N}(0, \sigma_{mut}/10)$.
4. Primitive mutation (p=0.2): random from set.
5. Wiring: to dissolved node's upstream/downstream neighbors; DAG check (retry ≤5× or discard).
6. Assign free/approximate basis.

### 7.3 Convergence Dampening
$\kappa$ = passes since last lysis. If $\kappa > \kappa_{thresh}$ (e.g. 100): halve $\sigma_{mut}$, $\gamma$; reset on next lysis.

## 8. Primitive Operator Set
All output $\mathbb{R}^D$ (project non-$D$ intermediates via mean-pool / truncate / pad).

**Unary (1 input):**
- Tanh(X): element-wise $\tanh$
- Softmax(X): normalized exponentials
- L2Norm(X): $X / ||X||_2$ (zero-safe)
- Fork(X): pass-through + temporary ≤$F$ exploratory edges to random eligible downstream (current pass only)

**Binary/N-ary (aggregate >2 via mean):**
- Add(X, Y): $X + Y$
- MatMul(X, Y): $(\operatorname{dot}(X,Y)/\sqrt{D}) \cdot X$ (attention-like self-projection)
- Concat(X, Y): concat → mean-pool pairs → $\mathbb{R}^D$
- Gate(X, Y): $X \odot \sigma(Y)$

## 9. Hyperparameters (Recommended Defaults)
| Symbol       | Default     | Description                              |
|--------------|-------------|------------------------------------------|
| $D$          | 256         | Vector dimension                         |
| $\alpha$     | 0.01        | Health reward rate                       |
| $\beta$      | 0.05        | Health punishment base rate              |
| $\gamma$     | 0.001       | Atrophy rate                             |
| $\eta$       | 0.001       | Parametric LR                            |
| $\eta_S$     | 0.0005      | Routing signature LR                     |
| $\sigma_{mut}$ | 0.1       | Mutation noise std                       |
| $\lambda_{async}$ | 0.2    | Inbox decay rate                         |
| $\lambda_\tau$ | 0.1      | Threshold steepness                      |
| $\omega$     | 0.05        | Phase velocity (rad/step)                |
| $K$          | 3           | Nodes spawned per lysis                  |
| $\Phi_{\max}$| $2\sqrt{D}$ | Max residue norm                         |
| $\kappa_{thresh}$ | 100    | Dampening trigger                        |
| $M$          | 64          | Node cap margin                          |
| $F$          | 2           | Max Fork exploratory edges               |

## 10. Implementation Notes
- Use sparse adjacency (CSR or concurrent pointers).
- Atomic updates for $R_t$ (mutex / lock-free).
- Topo sort for forward/reverse passes and mutation checks.
- Benchmark rigorously; MCT4 excels in sparse, latency-critical, online-adaptive regimes—not dense GPU workloads.

This specification is precise, consistent, and free of the overclaims present in earlier MCT versions. It offers a principled, biologically inspired alternative for targeted use cases in adaptive, real-time computation.

