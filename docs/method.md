# Method: Info-Gain Sampler

## Motivation

Masked Diffusion Models (MDMs) have emerged as a powerful alternative to autoregressive models for discrete sequence generation. By leveraging bidirectional attention, MDMs break free from strict left-to-right generation. However, this potential remains largely untapped due to a **training-inference mismatch**: while MDMs are trained under random masking patterns, inference entails an order-sensitive decoding process.

Existing samplers rely on **local certainty heuristics** (confidence, entropy, margin) to greedily select the next decoding target. These methods are non-robust due to the **myopia of local heuristics**: they ignore the long-term impact of current decisions on future uncertainty.

**Key observations:**

1. An optimal decoding action should be evaluated not only by its own prediction certainty but also by the *information gain* it provides for the remainder of generation.
2. MDMs' bidirectional architecture enables efficient information gain estimation in **one forward pass**, bypassing expensive iterative computations.

---

## Objective

We first define **state uncertainty** as the average marginal entropy over masked positions in state $z_t$:

$$\mathcal{H}(z_t) = \frac{1}{|\mathcal{M}_t|} \sum_{\ell \in \mathcal{M}_t} H^{(\ell)}(z_t)$$

The **information gain** of action $a_t$ is the reduction in state uncertainty it induces:

$$\text{IG}(a_t; z_t) := \mathcal{H}(z_t) - \mathcal{H}(z_{t-1})$$

where $z_{t-1} = \text{Apply}(z_t, a_t)$.

The **immediate cost** is the marginal entropy of the tokens being decoded at this step:

$$C(a_t \mid z_t) = \sum_{\ell \in A_t} H^{(\ell)}(z_t)$$

The Info-Gain Sampler selects the action that maximises:

$$J_{\text{IG}}(a_t \mid z_t) = \underbrace{\text{IG}(a_t; z_t)}_{\text{Information Gain}} - \underbrace{C(a_t \mid z_t)}_{\text{Immediate Cost}}$$

---

## Three-Step Cycle

At each decoding step:

1. **Sample** — generate $N$ diverse (token, position) candidates $\mathcal{C} = \{a_t^{(1)}, \dots, a_t^{(N)}\}$ via Gumbel sampling.
2. **Evaluate** — score every candidate in one batched forward pass: $J_{\text{IG}}(a_t^{(i)} \mid z_t)$ for all $i$.
3. **Transition** — commit the highest-scoring candidate $a_t^* = \arg\max_{a \in \mathcal{C}} J_{\text{IG}}(a \mid z_t)$ and repeat until all masked positions are filled.

---

## Implementation Details

- **Parallel candidate evaluation**: all $N$ candidates are scored in a single batched forward pass, fully exploiting MDMs' bidirectional architecture.
- **KV cache support**: optional prefix-cache and dual-cache modes accelerate inference (disabled by default for multimodal tasks).
- **Dynamic threshold**: a high-confidence bypass ($\gamma$) automatically skips the candidate-evaluation loop when uncertainty is already low, significantly reducing latency.
- **No external dependencies**: the core Info-Gain function is self-contained — no dllm required for the standalone API.
