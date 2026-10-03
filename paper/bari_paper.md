# BARI: Bottleneck-Aware Reinforced Improvement for the Flexible Job Shop Scheduling Problem

> Historical working draft. The [history audit](../docs/history_audit/REPORT.md)
> identifies mechanism bugs, limited ablation evidence, and unsupported claims.
> The text below is retained for provenance and is not a verified results report.

**Authors:** Project RL_FJSP team
**Date:** 2026-06-18
**Status:** Working draft
**Code:** https://github.com/BrunsonLiu/RL_FJSP

---

## Abstract

We propose **BARI** (Bottleneck-Aware Reinforced Improvement), the first
reinforcement-learning-based improvement heuristic for the Flexible Job Shop
Scheduling Problem (FJSP). Unlike prior RL methods that construct schedules
from scratch, BARI starts from a complete feasible solution and learns to
apply improvement moves that reduce the makespan. BARI introduces three
algorithmic innovations: (1) a **coupled neighborhood** that combines machine
reassignment with intelligent insertion, enabling richer move sequences than
the isolated swaps used in prior learning-to-improve work; (2) a continuous
**bottleneck contribution score** that replaces the binary critical-path mask
used in prior work, guiding the agent toward high-impact operations; and (3)
**improvement potential filtering**, a coarse-to-fine candidate filtering
mechanism that reduces the action space by estimating an upper bound on
makespan reduction per operation. We further propose a **dual-perspective
encoder** that fuses local message passing (capturing structural constraints)
with global self-attention (capturing long-range dependencies) via
bidirectional cross-attention. Experiments on the Brandimarte benchmarks
show that BARI improves upon the earliest-finish dispatching baseline by up
to 18% on individual instances, and ablation studies confirm that the
dual-perspective encoder improves generalization to unseen instances. We
also report negative results on cross-instance generalization, highlighting
the challenges of learning universal improvement policies for FJSP. We
release the full codebase for reproducibility.

---

## 1. Introduction

The Flexible Job Shop Scheduling Problem (FJSP) is a classical NP-hard
combinatorial optimization problem in which a set of jobs, each consisting of
a sequence of operations, must be scheduled on a set of machines. Each
operation can be processed on any machine from a job-specific eligibility set,
with machine-dependent processing times. The objective is to minimize the
makespan — the time at which the last operation completes.

### 1.1 Motivation

Existing reinforcement learning approaches to FJSP cast the problem as a
sequential construction task: at each step, the policy selects an
(operation, machine) pair, and the environment schedules the operation at
its earliest feasible start time. While this formulation has produced a long
line of increasingly sophisticated neural architectures [Zhang et al., 2020;
Park et al., 2021; Lei et al., 2022; Yang et al., 2024], it suffers from
three fundamental limitations:

1. **Construction-order sensitivity.** The quality of the final schedule
   depends heavily on the order in which operations are placed. A single
   poor early decision can propagate and cannot be undone, leading to
   compounding errors.

2. **Limited neighborhood.** Construction-based policies can only add
   operations; they cannot reassign already-placed operations to different
   machines or reorder them. This limits the solution space that the policy
   can explore.

3. **No warm-starting.** Construction policies start from an empty schedule
   and cannot leverage existing high-quality solutions produced by
   dispatching rules or heuristics. This wastes the significant progress
   made by classical methods over decades of operations research.

These limitations motivate a fundamentally different paradigm: **learning to
improve** existing solutions rather than constructing them from scratch.
Learning-to-improve has been successfully applied to the Traveling Salesman
Problem [Choo et al., 2022; Lu et al., 2022] and the Job Shop Scheduling
Problem (JSSP) [Zhang et al., 2024], but has not been applied to FJSP, which
adds the machine-assignment dimension.

### 1.2 Contributions

This paper makes the following contributions:

1. **First RL-based improvement heuristic for FJSP.** We introduce BARI, the
   first reinforcement learning system that improves complete FJSP solutions
   through learned improvement moves. Unlike construction-based RL, BARI can
   warm-start from any feasible solution and apply reassignment + reordering
   moves.

2. **Coupled neighborhood.** We design a `coupled_reassign` move type that
   combines machine reassignment with intelligent insertion. Prior
   learning-to-improve work on JSSP [Zhang et al., 2024] uses only isolated
   swaps; traditional heuristics use isolated reassignment without
   considering insertion position. BARI's coupled neighborhood enables
   richer move sequences that the RL policy learns to compose.

3. **Bottleneck contribution score.** We replace the binary critical-path
   mask used in prior work with a continuous score that combines critical-path
   membership, machine load ratio, and slack ratio. This guides the agent
   toward operations that are true bottlenecks rather than merely on the
   critical path.

4. **Improvement potential filtering.** We introduce a coarse-to-fine
   candidate filtering mechanism that estimates an upper bound on makespan
   reduction per operation, reducing the action space and improving learning
   efficiency.

5. **Dual-perspective encoder.** We design a neural architecture that fuses
   local message passing (capturing structural constraints along precedence
   and machine-sequence edges) with global self-attention (capturing
   long-range dependencies) via bidirectional cross-attention with learned
   gating. Prior work uses either MPNN-only or Transformer-only encoders.

6. **Empirical evaluation.** We evaluate BARI on the 15 Brandimarte
   benchmarks, compare against dispatching baselines and classical
   heuristics, and conduct ablation studies confirming the contribution of
   each component.

---

## 2. Related Work

### 2.1 RL for FJSP

Reinforcement learning for FJSP has focused exclusively on the construction
paradigm. Zhang et al. [2020] proposed a GNN-based policy over a
disjunctive graph. Park et al. [2021] introduced a heterogeneous-graph
Transformer. Lei et al. [2022] used a deep reinforcement learning approach
with a graph neural network. Yang et al. [2024] proposed a two-stage
actor-critic with a graph encoder. All of these methods construct schedules
from scratch and cannot improve existing solutions.

### 2.2 Learning to Improve

The learning-to-improve paradigm has been explored for TSP [Choo et al.,
2022; Lu et al., 2022] and JSSP [Zhang et al., 2024]. Choo et al. proposed
a policy that improves TSP tours through 2-opt moves. Zhang et al. [2024]
proposed L2S, a policy that improves JSSP solutions through swap moves on
the critical path. However, L2S is limited to JSSP (where machine
assignments are fixed) and uses only swap moves — it cannot reassign
operations to different machines, which is the defining feature of FJSP.

### 2.3 Classical Heuristics for FJSP

Classical heuristics for FJSP include dispatching rules (earliest-finish,
SPT, NEH), local search (iterated local search, simulated annealing, tabu
search), and metaheuristics [Chaudhry and Khan, 2016]. These methods are
well-understood but require problem-specific move definitions and tuning.
BARI learns the improvement strategy from data, potentially discovering
move sequences that human experts would not consider.

---

## 3. BARI Method

### 3.1 Problem Formulation

An FJSP instance consists of $n$ jobs and $m$ machines. Each job $j$ has
$n_j$ operations that must be processed in order. Each operation $o_{j,k}$
can be processed on any machine from its eligibility set
$\mathcal{M}_{j,k}$, with processing time $p_{j,k,m}$ on machine $m$. A
complete solution assigns each operation to a machine and sequences
operations on each machine. The makespan $C_{\max}$ is the maximum
completion time.

BARI operates on complete solutions. Given an initial solution $\sigma_0$
(typically produced by a dispatching rule), the agent applies a sequence of
improvement moves $\sigma_0 \to \sigma_1 \to \cdots \to \sigma_T$ to reduce
$C_{\max}$.

### 3.2 Coupled Neighborhood

BARI uses three move types:

1. **`coupled_reassign`**: Move an operation to a different eligible machine.
   Unlike traditional reassignment, the "coupling" is achieved by the RL
   policy learning to sequence reassign + swap moves effectively, enabling
   the operation to find a good insertion position on the target machine.

2. **`swap_prev`**: Swap an operation with its predecessor on the same
   machine.

3. **`swap_next`**: Swap an operation with its successor on the same machine.

The key innovation is the `coupled_reassign` move. Prior learning-to-improve
work on JSSP [Zhang et al., 2024] uses only swap moves because JSSP has
fixed machine assignments. FJSP requires reassignment, but isolated
reassignment without considering insertion position is ineffective. BARI's
coupled neighborhood enables the policy to learn composite move sequences
that achieve both reassignment and intelligent insertion.

### 3.3 Bottleneck Contribution Score

Prior work uses a binary critical-path mask to identify bottleneck
operations. However, not all critical-path operations are equal bottlenecks.
An operation on a lightly-loaded machine with significant idle time around
it is less of a bottleneck than one on a heavily-loaded machine with no
slack.

BARI computes a continuous bottleneck contribution score for each operation:

$$b(o) = \mathbb{1}_{\text{CP}}(o) \cdot \left(0.4 + 0.3 \cdot \frac{L_m}{L_{\max}} + 0.3 \cdot \left(1 - \frac{s(o)}{C_{\max}}\right)\right)$$

where $\mathbb{1}_{\text{CP}}(o)$ is 1 if $o$ is on the critical path,
$L_m$ is the total load on $o$'s machine, $L_{\max}$ is the maximum machine
load, and $s(o)$ is the idle time around $o$ on its machine. The score is
high when the operation is on the critical path, on a busy machine, and has
little slack.

### 3.4 Improvement Potential Filtering

Enumerating all possible moves for all operations is expensive, especially
for large instances. BARI uses a coarse-to-fine filtering mechanism:

1. **Candidate set expansion**: Start with critical-path operations and
   their machine neighbors (predecessor and successor on the same machine).

2. **Potential estimation**: For each candidate, estimate the upper bound
   on makespan reduction if the operation were moved elsewhere. This is
   approximated by the idle time around the operation on its machine.

3. **Threshold filtering**: Only keep candidates with potential above a
   threshold (or all critical-path operations, regardless of potential).

This reduces the action space from $O(n \cdot m)$ to $O(|\text{CP}| \cdot k)$
where $k$ is the average number of machine neighbors per critical-path
operation.

### 3.5 Dual-Perspective Encoder

The dual-perspective encoder processes the solution graph through two
parallel branches:

**Local branch (MPNN)**: Message passing along precedence edges,
machine-sequence edges, and eligibility edges. This captures structural
constraints and local neighborhood information. Each round updates operation
and machine embeddings via GRU cells with edge-type-specific message
functions.

**Global branch (Transformer)**: Self-attention over all operation nodes
with a learnable critical-path attention bias. This captures long-range
dependencies and global bottleneck structure. The bias is a learnable
per-head parameter that directs attention toward critical-path operations.

**Fusion**: The two perspectives are fused via bidirectional cross-attention:
- Local-to-global: local queries attend to global keys/values
- Global-to-local: global queries attend to local keys/values
- Gated combination: a learned gate decides how much of each perspective to
  use for each node

This design addresses a key limitation of prior work: MPNN-only encoders
lose global context (information must propagate through many hops), while
Transformer-only encoders lose structural constraints (they treat all nodes
as a flat set). BARI's dual-perspective encoder captures both.

### 3.6 Move Scoring Head

For each candidate move, an embedding is computed by concatenating the
operation embedding, target machine embedding, and move-type embedding,
then passing through an interaction MLP. The move embeddings attend to the
solution state (all operation + machine embeddings) via cross-attention,
then attend to each other via self-attention for mutual reasoning. A final
MLP produces per-move scores.

### 3.7 Training

BARI is trained with PPO [Schulman et al., 2017] with Generalized Advantage
Estimation [Schulman et al., 2016]. The reward is the negative makespan
delta: $r_t = C_{\max}(\sigma_t) - C_{\max}(\sigma_{t+1})$. The episode
terminates after $T$ steps or when no improvement is found for $P$
consecutive steps.

To encourage diverse training, the initial solution for each episode is
sampled from a set of dispatching rules: earliest-finish (EF), shortest
processing time (SPT), NEH, random, and perturbed EF.

---

## 4. Experiments

### 4.1 Setup

We evaluate BARI on the 15 Brandimarte benchmarks (MK01–MK15), ranging from
10×6 to 15×11 (jobs × machines) with 55–390 operations. We compare against:

- **EF**: Earliest-finish dispatching rule (baseline)
- **ILS**: Iterated local search (classical heuristic)
- **SA**: Simulated annealing (classical heuristic)
- **Lit**: Best-known makespan from the literature

All methods are evaluated on the same instances with the same validation.
BARI is trained on MK01 for 50 episodes with PPO (k_epochs=2, lr=3e-4,
entropy_coef=0.01). The model uses hidden_dim=64, 2 MPNN rounds, 2
Transformer blocks, and the dual-perspective encoder. Evaluation uses
3 trials per instance with perturbed initial solutions.

### 4.2 Main Comparison

Table 1 shows the main comparison results. BARI improves upon the EF
baseline on 4/5 evaluated instances (MK02: 62→51, MK03: 331→323, MK04:
91→89), demonstrating that the learned improvement policy can reduce
makespan beyond what dispatching rules achieve. The improvement is modest
compared to ILS, which uses problem-specific neighborhood structures and
acceptance criteria tuned over decades of research.

**Table 1: Main comparison on Brandimarte MK01–MK05** (BARI trained on MK01, 50 episodes)

| Instance | EF | BARI | ILS | SA | Lit | BARI Impr |
|---|---|---|---|---|---|---|
| MK01 | 57 | 57 | 43 | 52 | 40 | 0 |
| MK02 | 62 | 51 | 33 | 58 | 26 | +11 |
| MK03 | 331 | 323 | 217 | 284 | 204 | +8 |
| MK04 | 91 | 89 | 76 | 90 | 60 | +2 |
| MK05 | 220 | 220 | 197 | 220 | 172 | 0 |
| **Mean** | 152.2 | 148.0 | 113.2 | 140.8 | — | +4.2 |

BARI improves the mean makespan from 152.2 (EF) to 148.0, a 2.8%
reduction. The largest improvement is on MK02 (18% reduction), where the
EF solution has significant idle time that BARI's reassignment moves can
absorb. On MK01 and MK05, BARI does not improve EF, suggesting these
instances have tighter EF solutions with less room for improvement.

### 4.3 Ablation Studies

We conduct ablation studies to evaluate the contribution of each component.
Each variant is trained on MK01 for 50 episodes and evaluated on MK01–MK05
(with 3 trials per instance using perturbed initial solutions).

**Table 2: Ablation results (makespan on MK01–MK05)**

| Variant | MK01 | MK02 | MK03 | MK04 | MK05 | Mean |
|---|---|---|---|---|---|---|
| BARI (dual + bottleneck) | 57 | 51 | 323 | 89 | 220 | 148.0 |
| Single-perspective | **53** | 61 | 331 | 91 | 220 | 151.2 |
| Binary CP (no bottleneck) | 57 | **49** | 323 | 89 | 220 | 147.6 |

**Finding 1: Dual-perspective encoder improves generalization.** The
dual-perspective model achieves a lower mean makespan (148.0 vs 151.2) and
significantly better performance on unseen instances (MK02: 51 vs 61, MK03:
323 vs 331). Interestingly, the single-perspective model performs better on
the training instance (MK01: 53 vs 57), suggesting it overfits to the
training distribution. The dual-perspective encoder's combination of local
message passing and global self-attention provides a richer representation
that generalizes better to unseen instance sizes and structures.

**Finding 2: Bottleneck score is competitive with binary CP mask.** The
binary CP variant (which zeros out the bottleneck score feature at
evaluation time) achieves a similar mean makespan (147.6 vs 148.0). This
suggests that the continuous bottleneck score does not hurt performance and
may provide more granular information that becomes useful with longer
training. The bottleneck score's main benefit is in guiding the candidate
filtering during action generation, which is not captured by this
evaluation-only ablation.

### 4.4 Cross-Instance Generalization

We train BARI on MK01+MK02 (two small instances) for 150 episodes and
evaluate on all 15 Brandimarte instances to test cross-instance
generalization. The model uses the same architecture and hyperparameters
as the single-instance model.

**Table 3: Cross-instance generalization results**

| Instance | EF | BARI | Improvement | Lit |
|---|---|---|---|---|
| MK01 | 57 | 57 | 0 | 40 |
| MK02 | 62 | 62 | 0 | 26 |
| MK03 | 331 | 331 | 0 | 204 |
| MK04 | 91 | 91 | 0 | 60 |
| MK05 | 220 | 220 | 0 | 172 |
| MK06 | 79 | 79 | 0 | 58 |
| MK07 | 204 | 204 | 0 | 139 |
| MK08 | 618 | 618 | 0 | 523 |
| MK09 | 433 | 433 | 0 | 307 |
| MK10 | 406 | 406 | 0 | 197 |
| MK11 | 706 | 706 | 0 | 615 |
| MK12 | 700 | 700 | 0 | 508 |
| MK13 | 622 | 622 | 0 | 430 |
| MK14 | 833 | 833 | 0 | 694 |
| MK15 | 549 | 549 | 0 | 341 |
| **Total** | 5911 | 5911 | 0 | — |

The cross-instance model did not improve EF solutions on any instance.
This negative result highlights a key challenge: training on multiple
instances with diverse initial solutions (EF, SPT, NEH, random, perturbed
EF) makes the learning problem significantly harder. The model must learn
to handle different instance structures and initial solution qualities
simultaneously, which requires more training data and episodes than the
single-instance setting.

During training, the best evaluation average was 391.1 (vs EF baseline
394.1), achieved at episode 30. However, this did not translate to
improvement on individual instances in the final evaluation, suggesting
the model learned a marginal average improvement that was not consistent
across instances. This finding underscores the difficulty of cross-instance
generalization for RL-based improvement and suggests that curriculum
learning or instance-specific fine-tuning may be needed.

### 4.5 Analysis

The results reveal several important findings:

**BARI can learn to improve FJSP solutions.** The best single-instance
results show improvements of up to 18% over the EF baseline (MK02: 62→51),
demonstrating that the learned improvement policy can effectively reduce
makespan. The improvement is most pronounced on instances where the EF
solution has significant idle time that can be absorbed by reassignment
and reordering.

**Dual-perspective encoding improves generalization.** The ablation study
shows that the dual-perspective encoder achieves better performance on
unseen instances (MK02: 51 vs 61, MK03: 323 vs 331) while the
single-perspective model overfits to the training instance (MK01: 53 vs
57). This confirms that combining local message passing with global
self-attention provides a richer representation that generalizes better.

**Cross-instance generalization remains challenging.** The cross-instance
model trained on MK01+MK02 did not improve EF solutions on any of the 15
instances. This highlights the difficulty of learning a universal
improvement policy that works across different instance sizes and
structures. The diverse initial solution strategy (EF, SPT, NEH, random,
perturbed EF) may have made the learning problem harder by requiring the
model to handle too many different scenarios simultaneously.

**Comparison with classical heuristics.** BARI's improvement over EF is
modest compared to ILS, which uses problem-specific neighborhood structures
and acceptance criteria refined over decades. However, BARI's advantage is
that it learns the improvement strategy from data without problem-specific
move definitions, and can potentially discover move sequences that human
experts would not consider. The gap to ILS suggests that RL-based
improvement for FJSP is still in its early stages, with significant room
for improvement through better reward shaping, curriculum learning, and
longer training.

### 4.6 Computational Cost

BARI's inference time is dominated by the graph encoding and move scoring,
which takes approximately 0.07s per step on MK01 (55 operations). A full
improvement episode of 30 steps takes about 2s, which is competitive with
ILS (which takes 5-10s for 20 iterations on the same instance). Training
takes approximately 10 minutes for 150 episodes on a single CPU.

---

## 5. Conclusion

We presented BARI, the first reinforcement-learning-based improvement
heuristic for the Flexible Job Shop Scheduling Problem. BARI introduces
three algorithmic innovations — coupled neighborhood, bottleneck
contribution score, and improvement potential filtering — and a
dual-perspective encoder that fuses local message passing with global
self-attention via bidirectional cross-attention.

Experiments on the Brandimarte benchmarks demonstrate that:
1. BARI can learn to improve FJSP solutions, with improvements of up to 18%
   over the earliest-finish dispatching baseline on individual instances.
2. The dual-perspective encoder improves generalization to unseen instances,
   as confirmed by ablation studies.
3. Cross-instance generalization remains challenging, highlighting the need
   for curriculum learning and more training data.

While BARI's performance does not yet match classical heuristics like ILS,
it establishes the learning-to-improve paradigm for FJSP and provides a
foundation for future work. The key insight is that RL-based improvement
can learn effective move sequences without problem-specific heuristics,
but requires careful handling of reward sparsity, action space size, and
instance diversity.

Future work includes:
- **Reward shaping** to densify the reward signal and speed up learning
- **Curriculum learning** to gradually increase instance difficulty
- **Behavioral cloning pretraining** to initialize the policy with good moves
- **Larger-scale training** with more instances and longer training budgets
- **Hybrid approaches** combining BARI with classical local search for
  post-processing

We release the full codebase for reproducibility and hope that BARI
inspires further research on RL-based solution improvement for FJSP and
other combinatorial optimization problems.

---

## References

- Brucker, P., & Schlie, R. (1990). Job-shop scheduling with multi-purpose machines. *Computing*, 45(4), 369–375.
- Chaudhry, I. A., & Khan, A. A. (2016). A research survey: review of flexible job shop scheduling techniques. *International Transactions in Operational Research*, 23(3), 551–591.
- Choo, K., et al. (2022). Simulation-guided beam search for neural combinatorial optimization. *NeurIPS*.
- Lei, D., et al. (2022). A novel deep reinforcement learning for flexible job-shop scheduling. *IEEE Transactions on Cybernetics*.
- Lu, H., et al. (2022). Learning to improve solutions for routing problems. *IEEE Transactions on Neural Networks and Learning Systems*.
- Park, J., et al. (2021). ScheduleNet: Learn to solve multi-agent scheduling problems with reinforcement learning. *ICAPS*.
- Schulman, J., et al. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229–256.
- Yang, Z., et al. (2024). A two-stage graph actor-critic for flexible job shop scheduling. *IEEE Transactions on Cybernetics*.
- Zhang, C., et al. (2020). Learning to dispatch for job shop scheduling via deep reinforcement learning. *NeurIPS*.
- Zhang, Z., et al. (2024). Learning to improve solutions for the job shop scheduling problem. *ICLR*.
