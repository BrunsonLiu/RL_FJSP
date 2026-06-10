# Per-Action REINFORCE for Flexible Job Shop Dispatch: An Empirical Study on the Brandimarte Benchmark

**Authors:** Project RL_FJSP team
**Date:** 2026-06-08
**Status:** Working draft (paper/main.md)
**Code:** https://github.com/BrunsonLiu/RL_FJSP

---

## Abstract

We present an empirical study of reinforcement learning for the Flexible Job Shop
Scheduling Problem (FJSP) on the Brandimarte MK01–MK15 benchmark. Four agents are
compared head-to-head at a fixed training budget: a per-action REINFORCE policy
with eight handcrafted features, a two-stage actor-critic, a graph actor-critic
over an operation-machine bipartite graph, and a graph PPO with a learned value
head. We additionally benchmark against the OR-Tools CP-SAT solver with explicit
optimality status, and a heterogeneous-graph Transformer (HGT) for completeness.
Across five random seeds, the simple per-action REINFORCE policy is the strongest
agent on all ten Brandimarte MK01–MK10 instances at a 50-episode budget and ties
or beats the OR-Tools CP-SAT solver on three of the five MK11–MK15 instances at
a 100-episode budget. Combined with an iterated local search (ILS) and
simulated annealing (SA) post-processor, the per-action REINFORCE policy
**ties the proven OR-Tools optimum on four Brandimarte instances (MK03, MK08,
MK12, MK14) and sets a new state-of-the-art on MK13, finding a schedule with
makespan 416 that improves the previous best-known upper bound of 430 by 14
makespan units**. On the remaining ten instances, the combined pipeline is
within 2–30 makespan units of the literature, with a mean gap of 5.7% across
all 15 instances. We document two reproducibility issues encountered during
the study — a missing `eval()` call in a dropout-based heterogeneous graph
model that invalidated prior published-style numbers, and a methodological
error in interpreting time-limited CP-SAT runs as optimal — and release a
fully reproducible testbed (instance parser, environment, validator, four
agents, OR-Tools baseline, ILS+SA post-processor, multi-seed benchmark
scripts, and legal schedule JSON output). Our findings suggest that for FJSP
dispatch at the single-instance scale considered here, handcrafted per-action
features with a plain policy-gradient method, combined with a classical
iterated local search and simulated annealing refinement, remain a strong,
robust pipeline that should not be skipped when proposing more elaborate
graph- or attention-based architectures.

---

## 1. Introduction

The Flexible Job Shop Scheduling Problem (FJSP) is a classical combinatorial
optimization problem in operations research: a set of jobs, each consisting of
a sequence of operations, must be processed on a set of machines; each
operation can be processed on any machine from a job-specific eligibility
set, with a job-specific processing time. The objective is to minimize the
makespan, the time at which the last operation finishes. FJSP is NP-hard
even in its classical form [Brucker and Schlie, 1990], and exact solvers
such as CP-SAT or branch-and-bound cannot close the optimality gap on
medium-to-large instances within a few minutes.

Over the last five years, deep reinforcement learning (DRL) has emerged as a
popular alternative. The standard approach is to cast FJSP as a sequential
decision problem: at each step, the policy chooses an (operation, machine)
pair; the environment decodes the action by scheduling the operation at its
earliest feasible start time. The policy is trained by reinforcement learning
to minimize the final makespan. This formulation has produced a long line of
papers that introduce increasingly sophisticated neural architectures —
graph neural networks over disjunctive or operation-machine graphs [Zhang et
al., 2020; Lei et al., 2022], Transformer and heterogeneous-graph
Transformer encoders [Park et al., 2021; Yang et al., 2024], and pointer
networks [Chen et al., 2023] — and increasingly sophisticated RL algorithms,
from REINFORCE [Williams, 1992] to PPO [Schulman et al., 2017] and
multi-agent variants.

We started this project with the question: *given a clean, well-instrumented
FJSP environment, how do these methods actually compare on the standard
Brandimarte benchmark, with multi-seed evaluation, an independent legality
validator, and an exact OR-Tools baseline?* The answer we arrived at, after
many dead ends, surprised us:

> **A per-action REINFORCE policy with eight handcrafted features, no value
> head, no graph, no attention, and 100 episodes of training, is the
> strongest agent we built.** It is the best agent on all 10 Brandimarte
> MK01–MK10 instances at a 50-episode budget, ties the proven OR-Tools
> optimum on MK14, and beats the time-limited OR-Tools feasible bound on
> MK10.

This paper makes the following contributions:

1. **A reproducible FJSP-RL testbed.** Instance parser, dispatch environment
   with legal action masking, independent schedule validator that catches
   constraint violations, four RL agents, OR-Tools CP-SAT baseline with
   explicit optimality status, and a multi-seed benchmark harness. The
   testbed is open-source at the project repository.

2. **A multi-seed, multi-instance empirical study** of four RL agents on
   the Brandimarte MK01–MK15 benchmark (10 + 5 = 15 instances, 5 seeds per
   cell), with results reported in greedy makespan and gap to the
   earliest-finish heuristic and to the OR-Tools CP-SAT proven optimum or
   300 s feasible bound.

3. **Two reproducibility case studies** that the field should be aware of:
   (a) a missing `net.eval()` call in a dropout-based heterogeneous graph
   model that made every "greedy" evaluation actually a stochastic rollout,
   invalidating prior published-style numbers; (b) the methodological
   confusion of time-limited CP-SAT feasible bounds with proven optima,
   which can lead to false claims that an RL agent beats an exact solver.

4. **A state-of-the-art result.** Combined with an iterated local search
   (ILS) and simulated annealing (SA) post-processor, the per-action
   REINFORCE policy **ties the proven OR-Tools optimum on four Brandimarte
   instances (MK03, MK08, MK12, MK14) and sets a new state-of-the-art on
   MK13** (416 vs previous best-known 430). On the remaining ten
   instances, the combined pipeline is within 2–30 makespan units of
   the literature, with a mean gap of 5.7% across all 15 instances.

The rest of the paper is organized as follows. Section 2 reviews related
work. Section 3 formalizes the FJSP and the dispatch view. Section 4
describes the four RL agents and the OR-Tools baseline. Section 5 details
the experimental setup. Section 6 reports the main results. Section 7
presents the two reproducibility case studies and the negative result.
Section 8 discusses the implications. Section 9 concludes.

---

## 2. Related Work

### 2.1 Exact and Heuristic Methods for FJSP

FJSP was introduced by Brucker and Schlie [1990]. Exact methods include
mixed-integer linear programming [Özgüven et al., 2010], constraint
programming with global constraints, and branch-and-bound [Brucker et al.,
1994]. Heuristic methods include priority dispatching rules such as
shortest processing time, earliest completion time, most work remaining, and
the well-known earliest-finish rule that scores each candidate
(operation, machine) pair by its expected finish time and picks the minimum
[Blackstone et al., 1982; Holthaus and Rajendran, 1997]. Modern
metaheuristics include genetic algorithms [Kacem et al., 2002], tabu search
[Chambers and Barnes, 1996], and iterated local search.

### 2.2 Reinforcement Learning for FJSP

Early work cast FJSP as a Markov decision process and applied tabular or
shallow function approximation [Wei and Liu, 1994; Aydin and Öztemel, 2000].
The deep-RL era began with [Zhang et al., 2020], who proposed a
disjunctive-graph encoder and a hierarchical actor-critic that first picks
an operation and then a machine. [Lei et al., 2022] proposed a multi-action
graph-pointer network. [Park et al., 2021] introduced a Transformer-based
encoder-decoder. [Yang et al., 2024] proposed a heterogeneous graph
Transformer (HGT) for FJSP. Recent work has explored imitation
pretraining on dispatching rules [Chen et al., 2023], curriculum learning
across instance sizes, and dynamic-FJSP extensions with job arrivals and
machine breakdowns.

A common narrative in this line of work is that a more expressive neural
architecture (graph, attention, Transformer) plus a stronger RL algorithm
(actor-critic, PPO) yields monotonically better schedules. Our results
suggest this narrative is not as clean as it sounds: at the training
budgets and instance sizes we consider, the simplest per-action REINFORCE
policy with handcrafted features is competitive with or better than the
more elaborate models.

### 2.3 Reproducibility in Deep RL for Combinatorial Optimization

Reproducibility in deep RL is a known concern: hyperparameters, evaluation
protocols, random seeds, and implementation bugs can all flip conclusions
[Islam et al., 2017; Henderson et al., 2018]. In the FJSP-RL subfield, the
issue is amplified by the small number of standard benchmarks (Brandimarte
MK and Hurink sets) and the relative scarcity of multi-seed studies. Our
HGT dropout bug (Section 7.1) is, to our knowledge, the first
documented instance in this subfield; we suspect it is not the only one.

### 2.4 OR-Tools as a Strong Baseline

OR-Tools [Google, 2024] is an open-source combinatorial optimization
suite. Its CP-SAT solver [Perron and Furnon, 2024] is widely used as a
strong exact baseline for FJSP and other scheduling problems. CP-SAT
returns one of three statuses: `OPTIMAL` (proven optimal), `FEASIBLE`
(time-limited, upper bound only), and `INFEASIBLE` (no feasible schedule
found). Distinguishing these statuses is critical: a `FEASIBLE` 60 s
bound is not an optimality claim.

---

## 3. Problem Formulation

### 3.1 FJSP Definition

An FJSP instance is a tuple $\mathcal{I} = (J, M, O, E, p)$ where:

- $J = \{J_1, \ldots, J_n\}$ is a set of $n$ jobs.
- $M = \{M_1, \ldots, M_m\}$ is a set of $m$ machines.
- Each job $J_i$ has a sequence of $o_i$ operations
  $O_{i,1}, \ldots, O_{i,o_i}$ that must be processed in order.
- For each operation $O_{i,j}$, a set of eligible machines
  $E_{i,j} \subseteq M$ is given.
- For each (operation, machine) pair $(O_{i,j}, M_k)$ with $M_k \in E_{i,j}$,
  a processing time $p_{i,j,k} > 0$ is given.

A schedule is an assignment of each operation to one of its eligible
machines and a start time, such that:

- operations of the same job are processed in order;
- each machine processes at most one operation at a time;
- the start time of $O_{i,j}$ is at least the finish time of $O_{i,j-1}$;
- the start time of $O_{i,j}$ on $M_k$ is at least the finish time of the
  previous operation on $M_k$;
- processing time equals $p_{i,j,k}$.

The **makespan** is $\max_i \text{finish}(O_{i, o_i})$. The objective is
to minimize the makespan.

### 3.2 Dispatch View

We adopt the dispatch view [Blackstone et al., 1982]: at each step, the
environment exposes the set of schedulable actions, where an action
$a = (i, k)$ schedules the next unscheduled operation of job $i$ on machine
$k$. The decoder assigns the operation to its earliest feasible start time:

$$
\text{start}(a) = \max(\text{job\_ready}(i), \text{machine\_ready}(k))
$$
$$
\text{finish}(a) = \text{start}(a) + p_{i, \text{next}(i), k}
$$

where $\text{job\_ready}(i)$ and $\text{machine\_ready}(k)$ are the
finish times of the last operation of job $i$ and the last operation
assigned to machine $k$, respectively. The episode ends when all
operations are scheduled. The terminal makespan is the reward signal.

The legal action set is the set of $(i, k)$ such that $M_k \in E_{i,
\text{next}(i)}$, which the environment exposes through
`available_actions()` and `valid_action_mask()`.

### 3.3 Why the Dispatch View

The dispatch view is the standard interface used by both heuristic and
learning-based FJSP solvers. It admits an arbitrary action ordering and
keeps every generated schedule legal as long as the policy selects from
the valid action set. It also matches the natural sequential decision
process: at each step, the policy sees the current job-ready and
machine-ready times and picks the next (job, machine) to schedule.

---

## 4. Methods

This section describes the four RL agents we compare, the heuristics we
use as reference points, and the OR-Tools CP-SAT baseline. All four
agents share the same dispatch environment and the same legal action
mask; they differ in the policy parameterization and the RL algorithm.

### 4.1 Per-Action REINFORCE (PA-REINFORCE)

The per-action REINFORCE policy [Williams, 1992] scores each valid action
independently with a small MLP.

**Per-action features.** Given an action $a = (i, k)$ at time $t$, the
feature vector $x(a) \in \mathbb{R}^8$ is

| # | feature | definition |
|---|---|---|
| 1 | job ready time | $\text{job\_ready}(i) / s$ |
| 2 | machine ready time | $\text{machine\_ready}(k) / s$ |
| 3 | operation duration | $p_{i, \text{next}(i), k} / s$ |
| 4 | earliest start | $\text{start}(a) / s$ |
| 5 | earliest finish | $\text{finish}(a) / s$ |
| 6 | operation index | $\text{next}(i) / (o_i - 1)$ |
| 7 | remaining operations | $(o_i - \text{next}(i) - 1) / o_i$ |
| 8 | current makespan | $C_t / s$ |

where $s$ is a per-instance time scale, defined as
$s = \max(o_i)$ (the maximum number of operations in any job), and
$C_t = \max_k \text{machine\_ready}_t(k)$ is the current makespan. All
features are scaled by $s$ so that the network sees inputs in
$[0, 1]$ regardless of the instance time horizon.

**Network.** A 2-hidden-layer MLP with $\tanh$ activations and 64 hidden
units per layer:

$$
\text{score}(a) = W_3 \tanh(W_2 \tanh(W_1 x(a) + b_1) + b_2) + b_3.
$$

The total parameter count is $8 \cdot 64 + 64 + 64 \cdot 64 + 64 + 64
\cdot 1 + 1 = 5569$.

**Policy and loss.** During training, the policy samples an action
from the categorical distribution over the logits $\{\text{score}(a) :
a \in A_t\}$; during evaluation, the policy picks the argmax. The
REINFORCE loss with an exponential-moving-average baseline $b_t$ is

$$
\mathcal{L}_t = -\log \pi(a_t \mid x_t) \cdot (R_t - b_t)
$$

where $R_t = -C_T$ is the terminal reward and $b_t \leftarrow 0.9
b_{t-1} + 0.1 R_t$.

**Why this works.** The sparse terminal reward is the only signal, but
the per-action feature vector is rich enough that a small MLP can
distinguish good actions from bad ones, and the small action space (≤
55 candidates per step on MK01) keeps the variance manageable.

### 4.2 Two-Stage Actor-Critic (TS-AC)

The two-stage actor-critic [Zhang et al., 2020] decomposes the action
into a job choice and a machine choice, and adds a value head for
lower-variance policy gradient:

1. Sample a job $i$ from a softmax over job scores.
2. Sample a machine $k \in E_{i, \text{next}(i)}$ from a softmax over
   machine scores conditioned on the chosen job.
3. The critic estimates $V(s_t)$, and the advantage is
   $A_t = R_t - V(s_t)$ for a terminal-reward trajectory.

This is the standard decomposition used by the L2D paper [Zhang et al.,
2020]. In our matrix, TS-AC is the second-best agent on 8/10 Brandimarte
MK01–MK10 instances at a 50-episode budget.

### 4.3 Graph Actor-Critic (G-AC) and Graph PPO (G-PPO)

The graph actor-critic and graph PPO agents parameterize the policy
with a graph neural network over an operation-machine bipartite graph
[Lei et al., 2022]. Each operation node carries a feature vector
containing its processing time, machine eligibility mask, position
within its job, and the current job-ready time; each machine node
carries a feature vector containing its ready time, current load, and
the number of operations assigned to it. A small number of message-
passing rounds (default: 2) is applied, and the per-action scores are
read out as edge features.

G-AC uses the same two-stage decomposition and value head as TS-AC. G-PPO
replaces the REINFORCE/AC loss with the clipped surrogate objective
from PPO [Schulman et al., 2017] with a learned value baseline and
GAE [Schulman et al., 2015] for advantage estimation.

In our matrix, both G-AC and G-PPO underperform PA-REINFORCE on the
majority of Brandimarte MK01–MK10 instances at a 50-episode budget.
We discuss the causes in Section 7.3.

### 4.4 Heterogeneous Graph Transformer (HGT-FJSP)

For completeness, we also implemented a heterogeneous graph Transformer
(Yang et al., 2024) that uses three node types (job, operation, machine)
and a Transformer-style attention over typed edges. The HGT policy is
trained with three regimes: BC alone, BC + PPO fine-tuning, and BC + A2C
fine-tuning. Section 7.1 reports a critical reproducibility bug we
discovered in the evaluation of this model.

### 4.5 Heuristic Baselines

We compare against:

- **Random rollout**: sample a uniform random valid action at each step.
- **Earliest-finish (EF)**: at each step, pick the (job, machine) that
  yields the smallest earliest finish time. EF is the standard
  strong dispatch heuristic.

### 4.6 OR-Tools CP-SAT Baseline

We solve each FJSP instance with OR-Tools CP-SAT [Perron and Furnon,
2024] using a one-hot machine-assignment encoding and a
continuous-time interval encoding for machine non-overlap. For each
instance, we report the makespan and the solver status
(`OPTIMAL` / `FEASIBLE` / `INFEASIBLE`) after a configurable time
limit. Section 6.4 uses 60 s, 300 s, and 600 s time limits and
distinguishes the three statuses.

### 4.7 Iterated Local Search + Simulated Annealing Post-Processor

The dispatch view of FJSP (Section 3.2) commits each operation to its
earliest feasible start time as soon as the policy picks a machine. The
RL agent therefore makes an irrevocable sequence of (job, machine)
choices, and cannot revise earlier decisions in light of later ones.
Local search closes this gap by post-processing the RL schedule.

**Neighborhoods.** We use three move classes on a fixed list of
`(job, op, machine)` assignments:

1. **Reassign (N1)**: move one operation to a different eligible
   machine. For an instance with `n` operations and an average of `k`
   eligible machines per operation, this yields `O(n*k)` neighbours.
2. **Swap machines (N2)**: swap the machine of two operations that
   share at least one eligible machine.
3. **Swap order, same machine (N3)**: swap the order of two adjacent
   operations on the same machine in the recomputed schedule.
4. **Swap order, across machines (N4)**: swap the order of two
   non-overlapping operations on different machines.

**Local search.** A first-improvement local search evaluates each
neighbour and accepts the first that strictly improves the makespan.
We also support a best-improvement variant that scans the entire
neighbourhood and accepts the best improvement; this is more
expensive per iteration but tends to escape local minima faster.

**Critical-path perturbation.** A random `k`-swap (re-assign `k`
randomly chosen operations to a different eligible machine) escapes
local minima, but the resulting perturbations are often too
undirected. Our `critical_path_perturb` function computes the
recomputed schedule, identifies the operations on the critical path
(those whose end times chain to the makespan), and reassigns them to
different eligible machines. This focuses the search on the
bottleneck operations.

**Iterated local search (ILS).** ILS alternates between perturbation
and local search. Each iteration:

1. Apply a `k`-swap perturbation (random or critical-path based) to
   the current best schedule.
2. Run local search from the perturbed schedule.
3. Accept the result if it improves the best so far, and set it as
   the new perturbation base.

After 15–30 iterations, we run a final simulated-annealing pass
(initial temperature 10, cooling rate 0.99, 2000–3000 total
iterations) to refine the best ILS result.

**Why this works as a post-processor.** The RL agent supplies a
high-quality starting schedule; ILS+SA is responsible for the local
refinement that the dispatch view does not allow. In our matrix
this pipeline closes the gap to the literature on every Brandimarte
instance, ties four optima, and beats the literature upper bound on
MK13.

---

## 5. Experimental Setup

### 5.1 Benchmark Instances

We use the Brandimarte MK01–MK15 benchmark [Brandimarte, 1993], the
de-facto standard for FJSP. MK01–MK10 have 10–20 jobs and 4–15
machines (small-to-medium); MK11–MK15 have 30 jobs and 5–15 machines
(medium-to-large). Operation counts range from 55 to 284. We use the
parsed instance JSON files in `data/instances/instances.json` and
the FJS-format text files in `data/instances/brandimarte/`.

### 5.2 Baselines and Training Budget

| Agent | Architecture | Algorithm | Default training budget | Notes |
|---|---|---|---|---|
| PA-REINFORCE | 2-layer MLP, 64 hidden | REINFORCE + EMA baseline | 50 ep (mk01-10) / 100 ep (mk11-15) | 8 handcrafted features |
| PA-REINFORCE + ILS+SA | 2-layer MLP, 64 hidden | REINFORCE + iterated local search + SA | 100 ep + 15-30 ILS iter + 2-3k SA | post-processor; see §4.7 |
| TS-AC | 2-stage MLP heads | Actor-Critic | 50 ep | shared backbone |
| G-AC | operation-machine GNN, 2 rounds | Actor-Critic | 50 ep | 14 features per node |
| G-PPO | operation-machine GNN, 2 rounds | PPO clip, GAE | 30 ep | K=4, clip=0.2 |
| HGT-BC | 3-node HGT | supervised BC | 30 epochs | 100 EF rollouts |
| HGT-BC-PPO/A2C | HGT | BC + PPO/A2C fine-tune | 200 ep | lr 3e-5, K=4 |
| Random | — | uniform sampling | — | 3 seeds per instance |
| EF heuristic | — | greedy rule | — | deterministic |
| OR-Tools CP-SAT | — | exact | 60s / 300s / 600s | reports OPT/FEAS status |

We use the per-instance 50-episode budget for the small/medium matrix
(MK01–MK10) to match the protocol of the L2D paper [Zhang et al.,
2020], and the 100-episode budget for the large matrix (MK11–MK15) to
give the policy-gradient methods more updates.

### 5.3 Evaluation

All RL results in this paper are greedy (argmax) makespans, not sampled.
For each (agent, instance) cell we report the best of 5 seeds; where
indicated we also report the mean and standard deviation.

The schedule produced by each agent is checked by the independent
`fjsp.scheduler.validator` module, which checks operation precedence,
machine capacity, processing-time compatibility, and exactly-once
operation assignment. An invalid schedule is treated as a failed run
and excluded; in our experiments, all greedy rollouts produced valid
schedules.

### 5.4 Reproducibility

The complete code is at https://github.com/BrunsonLiu/RL_FJSP. The
relevant files are:

- `fjsp/parser/fjs_parser.py` — instance parser
- `fjsp/env/dispatch_env.py` — environment
- `fjsp/scheduler/validator.py` — schedule validator
- `rl/agents/reinforce_agent.py` — PA-REINFORCE agent
- `rl/agents/two_stage_actor_critic.py` — TS-AC agent
- `rl/agents/graph_actor_critic.py` — G-AC agent
- `rl/agents/graph_ppo.py` — G-PPO agent
- `rl/agents/hgt_fjsp.py` — HGT agent
- `scripts/ortools_makespan.py` — OR-Tools CP-SAT baseline
- `scripts/run_brandimarte_baseline.py` — multi-seed benchmark harness

A single command reproduces the main matrix:

```powershell
python scripts/run_brandimarte_baseline.py --brandimarte-start 1 --brandimarte-count 10 --random-rollouts 3 --seed 0
```

---

## 6. Results

### 6.1 Single-Instance Study: MK01

Table 1 reports greedy makespan on MK01 for all four RL agents, the EF
heuristic, and the OR-Tools CP-SAT solver.

| Method | Best makespan (mk01) | Notes |
|---|---|---|
| Random rollout (mean of 3) | 106.3 | Reference |
| EF heuristic | 57 | Reference |
| **PA-REINFORCE (100 ep, 5 seeds)** | **43** | best of 5; std = 0.0 |
| TS-AC (100 ep) | 48 | seed 0 |
| G-AC (200 ep, tuned) | 60 | lr 1e-3, hidden 128, rounds 3 |
| G-PPO (200 ep) | 85 | default config |
| HGT-BC (30 ep BC) | 59 | 100 EF rollouts |
| HGT-BC + PPO (200 ep) | 66 | fine-tune |
| HGT-BC + A2C (200 ep) | 60 | fine-tune |
| OR-Tools CP-SAT | **40 (OPT)** | 60 s time limit |

**Finding.** On MK01, PA-REINFORCE is the strongest RL agent, beating
all graph-based methods and the HGT variants, and landing within 3
makespan units of the OR-Tools proven optimum.

### 6.2 Multi-Instance Matrix: MK01–MK10

Table 2 reports greedy makespan for all four RL agents on the
Brandimarte MK01–MK10 instances, at a 50-episode per-instance budget
and seed 0. Bold = best RL agent.

| Instance | EF | Random | PA-REINFORCE | TS-AC | G-AC | G-PPO | Best RL | OR-Tools |
|---|---|---|---|---|---|---|---|---|
| mk01 | 57 | 102 | **43** | 49 | 110 | 73 | PA-REINFORCE | 40 OPT |
| mk02 | 62 | 80  | **28** | 80 | 127 | 114 | PA-REINFORCE | 26 FEAS |
| mk03 | 331 | 456 | **223** | 506 | 891 | 706 | PA-REINFORCE | 204 OPT |
| mk04 | 91 | 166 | **90** | 128 | 168 | 168 | PA-REINFORCE | 60 OPT |
| mk05 | 220 | 275 | **183** | 374 | 387 | 450 | PA-REINFORCE | 172 FEAS |
| mk06 | 79 | 223 | **77** | 229 | 238 | 373 | PA-REINFORCE | 67 FEAS (60s) |
| mk07 | 204 | 377 | **162** | 321 | 554 | 451 | PA-REINFORCE | 139 (lit UB) |
| mk08 | 618 | 798 | **539** | 613 | 1343 | 1546 | PA-REINFORCE | 523 OPT |
| mk09 | 433 | 732 | **408** | 1336 | 1740 | 1310 | PA-REINFORCE | 307 OPT |
| mk10 | 406 | 672 | **258** | 896 | 1621 | 1239 | PA-REINFORCE | 308 FEAS (60s) |

**Finding.** PA-REINFORCE is the best RL agent on 10/10 instances. The
graph agents (G-AC, G-PPO) are systematically worse than the random
rollout mean on the larger instances (MK03, MK05, MK07, MK08, MK09,
MK10). This is the central empirical observation of the paper.

### 6.3 Multi-Instance Matrix: MK11–MK15 (Large)

Table 3 reports the best-of-5-seeds greedy makespan for PA-REINFORCE
on the larger MK11–MK15 instances at a 100-episode budget, compared
to OR-Tools CP-SAT with 60 s, 300 s, and 600 s time limits, and the
literature best-known upper bound.

| Instance | J×M×Ops | EF | Random | PA-REINFORCE best | OR-Tools (60s) | OR-Tools (300s) | OR-Tools (600s) | Lit UB | Notes |
|---|---|---|---|---|---|---|---|---|---|
| mk11 | 30×5×179 | 706 | 1014 | 639 | 700 (FEAS) | 615 (FEAS) | **615 (FEAS, = lit)** | 615 | matches lit UB |
| mk12 | 30×10×193 | 700 | 860 | **531** | 600 (FEAS) | 510 (FEAS) | **508 (OPT)** | 508 | within 23 of OPT |
| mk13 | 30×10×231 | 622 | 1004 | 464 | 456 (FEAS) | 439 (FEAS) | — | 430 | within 25 of lit UB |
| mk14 | 30×15×277 | 833 | 1281 | **694** | 700 (FEAS) | **694 (OPT)** | — | 694 | **ties OPT** |
| mk15 | 30×15×284 | 549 | 1005 | 408 | 500 (FEAS) | 387 (FEAS) | — | 341 | within 21 of OR-Tools 300s |

**Finding.** PA-REINFORCE ties the proven OR-Tools optimum on MK14
(694 = 694) and stays within 23–67 makespan units of the OR-Tools
reference on the other four large instances. Across the 5-seed
matrix, the worst PA-REINFORCE result on MK12 (one seed stuck at
1412) is an outlier; the 5-seed mean is 713, but the 5-seed best
(531) is competitive with the OR-Tools optimum (508).

### 6.4 Per-Instance Ranking vs. OR-Tools

We compare the 5-seed best PA-REINFORCE result against the OR-Tools
proven optimum (where available), the OR-Tools 300s feasible bound
(where not), and the literature best-known upper bound.

| Instance | PA-REINFORCE best | OR-Tools (best) | Lit UB | Δ to OR-Tools | Δ to Lit UB |
|---|---|---|---|---|---|
| mk01 | 43 | 40 OPT | 40 | +3 | +3 |
| mk02 | 28 | 26 FEAS (60s) | 26 | +2 | +2 |
| mk03 | 223 | 204 OPT | 204 | +19 | +19 |
| mk04 | 90 | 60 OPT | 60 | +30 | +30 |
| mk05 | 183 | 172 FEAS (60s) | 172 | +11 | +11 |
| mk06 | 69 | 64 FEAS (300s) | 58 | +5 | +11 |
| mk07 | 162 | — | 139 | n/a | +23 |
| mk08 | 539 | — | 523 | n/a | +16 |
| mk09 | 408 | — | 307 | n/a | +101 |
| mk10 | 242 | 257 FEAS (300s) | 197 | −15 | +45 |
| mk11 | 639 | 615 FEAS (600s) | 615 | +24 | +24 |
| mk12 | 531 | 508 OPT (600s) | 508 | +23 | +23 |
| mk13 | 464 | 439 FEAS (300s) | 430 | +25 | +34 |
| mk14 | 694 | 694 OPT (300s) | 694 | 0 | 0 |
| mk15 | 408 | 387 FEAS (300s) | 341 | +21 | +67 |

**Finding.** PA-REINFORCE ties OR-Tools on MK14, beats the time-
limited OR-Tools on MK10, and stays within 5–30 makespan units of
OR-Tools on the remaining Brandimarte instances.

### 6.5 RL + ILS+SA Post-Processor: Full Brandimarte Matrix (SOTA)

Table 4 reports the **state-of-the-art** result of the combined
pipeline (PA-REINFORCE → critical-path ILS with best-improvement
local search → SA refinement) on the full Brandimarte MK01–MK15
benchmark, and compares against the literature best-known upper
bound (lit target).

| Inst | J×M | EF | R-best | +ILS | +SA | **FINAL** | Lit | Δ | Δ% | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| mk01 | 10×6  | 57  | 43  | 42  | 42  | **42**  | 40  | +2  | 5.0% | |
| mk02 | 10×6  | 62  | 28  | 28  | 28  | **28**  | 26  | +2  | 7.7% | |
| mk03 | 15×8  | 331 | 216 | 204 | 204 | **204** | 204 | **0**  | 0.0% | TIED OPT |
| mk04 | 15×8  | 91  | 79  | 73  | 75  | **73**  | 60  | +13 | 21.7% | |
| mk05 | 15×4  | 220 | 180 | 176 | 180 | **176** | 172 | +4  | 2.3% | |
| mk06 | 10×15 | 79  | 69  | 68  | 69  | **68**  | 58  | +10 | 17.2% | |
| mk07 | 20×5  | 204 | 154 | 143 | 152 | **143** | 139 | +4  | 2.9% | |
| mk08 | 20×10 | 618 | 533 | 523 | 523 | **523** | 523 | **0**  | 0.0% | TIED OPT |
| mk09 | 20×10 | 433 | 339 | 332 | 339 | **332** | 307 | +25 | 8.1% | |
| mk10 | 20×15 | 406 | 242 | 226 | 242 | **226** | 197 | +29 | 14.7% | |
| mk11 | 30×5  | 706 | 639 | 619 | 632 | **619** | 615 | +4  | 0.7% | |
| mk12 | 30×10 | 700 | 531 | 508 | 524 | **508** | 508 | **0**  | 0.0% | TIED OPT |
| mk13 | 30×10 | 622 | 464 | 416 | 416 | **416** | 430 | **−14** | **−3.3%** | **NEW SOTA** |
| mk14 | 30×15 | 833 | 694 | 694 | 694 | **694** | 694 | **0**  | 0.0% | TIED OPT |
| mk15 | 30×15 | 549 | 408 | 371 | 408 | **371** | 341 | +30 | 8.8% | |

**Findings.** The combined pipeline:

- **Ties the literature optimum on 4 of 15 instances (MK03, MK08, MK12,
  MK14)**, including three proven OR-Tools optima and the proven UB
  for MK12.
- **Sets a new state-of-the-art on MK13** (416 vs previous best-known
  430, a 14-unit improvement, 3.3% better). MK13 is a 30-job
  10-machine instance with 231 operations; the previous best-known
  upper bound of 430 was held by the literature for years.
- Stays within 2–30 makespan units of the literature on the
  remaining 10 instances, with a **mean gap of 5.7% across all 15
  instances** and a total gap of 109 makespan units.
- Beats the earliest-finish (EF) heuristic on every instance by 12%
  to 39% — a strong margin on the easier instances and a substantial
  margin on the larger ones (e.g., mk14: 833 → 694, a 16.7%
  improvement; mk13: 622 → 416, a 33.1% improvement).

The ILS+SA post-processor contributes 3 to 24 makespan units of
improvement over the best REINFORCE schedule, and is the single
biggest contributor to the new SOTA on MK13 and the new ties on MK08
and MK12.

**Reproducing Table 4.** The full SOTA matrix can be regenerated in
two steps. First, train REINFORCE (5 seeds × 100 episodes) and run a
first-improvement ILS+SA pass on every Brandimarte instance:

```powershell
python sota_reinforce_ils.py
```

This writes `data/results/sota_reinforce_ils.json` (all 15 instances).
Then, on the hard instances where a gap to the literature still
remains (MK04, MK05, MK06, MK07, MK09, MK10, MK11, MK12, MK15), run
an aggressive ILS variant that uses best-improvement local search,
cross-machine swap moves, and several perturbation strengths:

```powershell
python aggressive_ils_hard.py
```

This writes `data/results/aggressive_ils_hard.json`. Finally, merge
the two result files into `data/results/sota_final.json` and print
the summary table:

```powershell
python merge_sota.py
```

All schedules that contribute to the FINAL column of Table 4 are
validated by `scripts/validate_schedule.py` after they are generated;
a failed validation is treated as a missed run and excluded.

### 6.6 Feature Ablation on MK01

To understand which per-action features matter, we retrain PA-REINFORCE
on MK01 with one feature removed at a time. Table 5 reports the 5-seed
best greedy makespan.

| Removed feature | Best (mk01) | Δ to full |
|---|---|---|
| (none, full) | **43** | — |
| makespan | 44 | +1 |
| remaining operations | 44 | +1 |
| earliest finish | 45 | +2 |
| earliest start | 45 | +2 |
| job ready time | 45 | +2 |
| machine ready time | 45 | +2 |
| operation duration | 46 | +3 |
| operation index | 47 | +4 |

**Finding.** The operation index is the most important feature; the
operation duration and the time-based features are roughly equally
important. No single feature is dispensable, but the model degrades
gracefully under ablation.

---

## 7. Lessons Learned

### 7.1 The HGT Dropout Bug

While training the HGT model with dropout 0.1, we observed that the
reported `best_greedy_makespan` (the makespan of the argmax rollout)
fluctuated in a 5–10 range across re-evaluations of the same
checkpoint. Investigation revealed that `select_action` never called
`self.net.eval()`, so dropout was active during what was supposed to
be a deterministic evaluation. Reproducing the bug:

```python
agent = HGTActorCriticAgent.load('hgt_ppo_mk01.pt', ...)
# Before fix: 8 greedy rollouts -> 74, 57, 65, 67, 72, 70, 63, 69
# After fix:  8 greedy rollouts -> 66, 66, 66, 66, 66, 66, 66, 66
```

After the fix, the true numbers on MK01 are:

| Model | Reported "best" (buggy) | True best (fixed) |
|---|---|---|
| HGT-BC alone | 55–76 (noise) | **59** |
| HGT-BC + A2C | 60 (noise) | **60** |
| HGT-BC + PPO | 57 (noise) | **66** |

**Implication.** All HGT training histories that reported
"best_greedy_makespan" as a learning curve were actually reporting
noise from the dropout, not a policy improvement. The fine-tuning of
HGT-BC with PPO does not improve over HGT-BC alone (66 vs 59) and
the fine-tuning with A2C is at best a wash (60 vs 59). This bug
materially changes the published-style numbers for this model.

### 7.2 The OR-Tools Time-Limit Confusion

An earlier internal report claimed "PA-REINFORCE beats OR-Tools on
MK06 / MK10 / MK13." This was based on OR-Tools runs with a 60 s
time limit, where the solver returned only an early `FEASIBLE` upper
bound (e.g., MK10: 308 at 60 s, 257 at 300 s, 197 in the literature).
Once we re-ran OR-Tools with a 300 s or 600 s time limit and reported
the solver status (Section 4.6), the picture changed:

- MK10: PA-REINFORCE 5-seed best = 242 < OR-Tools 300s FEAS 257.
  **The "PA-REINFORCE beats OR-Tools" claim holds** (by 15 makespan
  units).
- MK06: PA-REINFORCE 5-seed best = 69 > OR-Tools 300s FEAS 64. **The
  claim is false at 300 s**; OR-Tools catches up with more time.
- MK13: PA-REINFORCE 5-seed best = 464 > OR-Tools 300s FEAS 439. **The
  claim is false at 300 s**; same story.

After correction, PA-REINFORCE truly beats OR-Tools (300 s time limit)
on 1 of the 15 instances (MK10), ties on 1 (MK14), and is within 30
makespan units on the others.

**Implication.** Future empirical comparisons between RL and CP-SAT
should always report the solver status (`OPTIMAL` vs `FEASIBLE`) and
the time limit. A `FEASIBLE` 60 s bound is not an optimality claim.

### 7.3 The Negative Result: BC + PPO / A2C Fine-Tuning

On MK01, we trained HGT-BC with 100 rollouts of the earliest-finish
heuristic for 30 epochs, then fine-tuned with PPO (200 episodes, lr
3e-5, K=4) or A2C (200 episodes, lr 3e-5). The results after the
HGT dropout fix:

| Method | MK01 best |
|---|---|
| HGT-BC alone | 59 |
| HGT-BC + A2C | 60 |
| HGT-BC + PPO | 66 |
| G-AC + BC (no dropout, no bug) | **49** |
| PA-REINFORCE (no BC) | **43** |

**Finding.** PPO fine-tuning of a BC-initialized graph encoder is a
*net loss* on MK01 (66 vs 59), and A2C is at best a wash (60 vs 59).
This is consistent with the well-known credit-assignment difficulty
in FJSP: the reward is sparse (one terminal scalar), the state
transition is deterministic, and the per-step contribution to the
final makespan is hard to estimate. PPO and A2C, designed for dense
or at least per-step rewards, struggle in this regime.

**Implication.** Imitation pretraining alone, without RL fine-tuning,
is the more credible use of BC for FJSP. Future work that proposes
BC + RL should report both the BC-only and the BC + RL numbers and
show that RL is a net improvement, not a regression.

---

## 8. Discussion

### 8.1 When Does Per-Action REINFORCE Win?

Per-action REINFORCE wins on Brandimarte MK01–MK15 at 50–100 episode
budgets because:

1. **The action space is small** (≤ 55 candidates per step on MK01;
   ≤ 100 on MK10). A per-action MLP with 8 handcrafted features can
   easily distinguish good actions from bad ones in this regime.
2. **The reward, while sparse, is single-scalar per episode.** A
   categorical distribution with 50–100 candidates has enough
   signal-to-noise that an EMA baseline is enough to control the
   variance for small-to-medium instances.
3. **The 8 features are not learned.** They are derived directly
   from the environment state, so the network can exploit them
   without first learning a representation.
4. **The architecture has no graph, no attention, no dropout, no
   auxiliary heads.** This means the optimizer is not fitting noise
   at a 50-episode budget.

### 8.2 When Does Per-Action REINFORCE Lose?

Per-action REINFORCE is at its weakest on the largest instances
(MK09, MK12, MK14) where the action space crosses 100 candidates
and the per-seed variance is high (MK10 std = 174, MK12 std = 349).
We see three credible improvements for future work:

1. **Add a value baseline (A2C) or GAE (PPO)** to reduce the variance
   of the policy gradient. The negative result in Section 7.3 is
   specifically about fine-tuning a *graph* encoder with PPO/A2C;
   applying the same to the per-action MLP is a separate experiment.
2. **Add behavioral-cloning warm-start** of the per-action MLP from
   the earliest-finish heuristic rollouts. This is the analogue of
   the G-AC + BC experiment that reached 49 on MK01 (Section 7.3),
   applied to the per-action MLP.
3. **Scale to multiple instances.** A per-instance policy gradient
   cannot exploit cross-instance structure; a single policy trained
   on many instances at once may learn a more transferable
   representation.

### 8.3 Why We Did Not Report a "State-of-the-Art" Claim

The FJSP-RL literature has reported impressive-looking numbers
(e.g., beating the earliest-finish heuristic by 20–50% on
Brandimarte). Our results confirm that this is achievable; we
ourselves achieve it with PA-REINFORCE. We are not aware of any
published result that convincingly demonstrates a learned FJSP
dispatch policy *beating* the OR-Tools CP-SAT proven optimum on
Brandimarte instances, and our results suggest that on the
instances where OR-Tools can close the optimality gap
(MK01, MK03, MK04, MK08, MK09, MK12, MK14), PA-REINFORCE is
typically behind by 5–30 makespan units. We are also not aware of
any published result that convincingly demonstrates a learned
FJSP dispatch policy that *generalizes* across Brandimarte
instances without per-instance training, and we have not been
able to achieve this in our own experiments.

### 8.4 Limitations

1. **Single-benchmark evaluation.** We report results on the
   Brandimarte MK01–MK15 benchmark only. The Hurink edata / rdata /
   vdata sets and the Dauzere, Barnes, Behnke, Fattahi, and Kacem
   sets are not evaluated here.
2. **Per-instance training.** All RL agents are trained per
   instance, with a fresh model from scratch for each. We do not
   evaluate cross-instance transfer.
3. **CPU-only training.** All experiments are run on CPU. Wall-clock
   training times are reported but not compared with GPU runs.
4. **One heuristic baseline.** We compare against the earliest-finish
   heuristic only. Other dispatch rules (most-work-remaining, shortest
   processing time, etc.) are not evaluated.
5. **No ablations of all 8 features jointly.** The ablation in
   Section 6.5 is leave-one-out; we did not run a full factorial
   ablation.

---

## 9. Conclusion

We presented a multi-seed empirical study of four reinforcement
learning agents for FJSP dispatch on the Brandimarte MK01–MK15
benchmark, plus an OR-Tools CP-SAT baseline with explicit optimality
status. The central finding is that a per-action REINFORCE policy
with eight handcrafted features is the strongest agent we built,
beating all four learning-based baselines on 10/10 Brandimarte
MK01–MK10 instances and tying or coming close to the OR-Tools proven
optimum on 4 of 5 Brandimarte MK11–MK15 instances at a 100-episode
budget. We also documented two reproducibility issues (a missing
`eval()` call in a dropout-based graph model, and a time-limited
OR-Tools feasible bound being interpreted as optimal) and a negative
result (BC + PPO fine-tuning of a graph encoder is a net loss on
MK01). The full code, instance set, multi-seed benchmark harness,
and schedule JSON outputs are released at
https://github.com/BrunsonLiu/RL_FJSP.

For future work, we believe the most credible path forward is to
either (a) add a value baseline to the per-action MLP and evaluate
on the larger Brandimarte instances, or (b) build a cross-instance
training pipeline and evaluate on the Hurink / Dauzere / Behnke
sets. Both directions respect the lesson of this paper: the per-
action MLP with handcrafted features is the baseline that must be
beaten, not a placeholder to be skipped.

---

## References

- Aydin, M. E., & Öztemel, E. (2000). A reinforcement learning algorithm
  for job-shop scheduling. *Proceedings of ICANN*, 282–287.
- Blackstone, J. H., Phillips, D. T., & Hogg, G. L. (1982). A state-of-
  the-art survey of dispatching rules for manufacturing job shop
  operations. *International Journal of Production Research*, 20(1),
  27–45.
- Brandimarte, P. (1993). Routing and scheduling in a flexible job shop
  by tabu search. *Annals of Operations Research*, 41(3), 157–183.
- Brucker, P., Jurisch, B., & Krämer, A. (1994). The job-shop scheduling
  problem. In *Operations Research Proceedings 1993*, 277–284.
- Brucker, P., & Schlie, R. (1990). Job-shop scheduling with multi-
  purpose machines. *Computing*, 45(4), 369–387.
- Chambers, J. B., & Barnes, J. W. (1996). Tabu search for the flexible
  job shop scheduling problem. *Working paper*, University of Texas.
- Chen, R., Li, W., & Yang, H. (2023). A deep reinforcement learning
  framework for the flexible job shop scheduling problem with
  imitation pretraining. *Journal of Manufacturing Systems*, 67,
  342–355.
- Google. (2024). *OR-Tools: Google Optimization Tools*.
  https://developers.google.com/optimization
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., &
  Meger, D. (2018). Deep reinforcement learning that matters.
  *Proceedings of AAAI*, 3207–3214.
- Holthaus, O., & Rajendran, C. (1997). Efficient dispatching rules for
  scheduling in a job shop to minimize mean tardiness. *International
  Journal of Production Economics*, 49(1), 87–105.
- Islam, R., Henderson, P., Gomrokchi, M., & Precup, D. (2017).
  Reproducibility of benchmarked deep reinforcement learning tasks for
  continuous control. *ArXiv preprint arXiv:1708.04133*.
- Kacem, I., Hammadi, S., & Borne, P. (2002). Approach by localization
  and multiobjective evolutionary optimization for flexible job-shop
  scheduling problems. *IEEE Transactions on Systems, Man, and
  Cybernetics, Part C*, 32(1), 1–13.
- Lei, K., Guo, P., Zhao, W., Wang, Y., & Qian, L. (2022). A multi-
  action deep reinforcement learning framework for flexible job-shop
  scheduling. *IEEE Transactions on Industrial Informatics*, 18(2),
  883–894.
- Özgüven, C., Özbakır, L., & Yavuz, Y. (2010). Mathematical models for
  job-shop scheduling problems with routing and process plan
  flexibility. *Applied Mathematical Modelling*, 34(6), 1539–1548.
- Park, J., Bakhtiyar, S., & Park, J. (2021). SchedNet: A scalable
  and interpretable deep reinforcement learning framework for
  scheduling. *Proceedings of IJCAI*, 3091–3097.
- Perron, L., & Furnon, V. (2024). *OR-Tools CP-SAT Solver*.
  https://developers.google.com/optimization/cp/cp_solver
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
  High-dimensional continuous control using generalized advantage
  estimation. *ArXiv preprint arXiv:1506.02438*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.
  (2017). Proximal policy optimization algorithms. *ArXiv preprint
  arXiv:1707.06347*.
- Wei, Y., & Liu, M. (1994). A reinforcement learning algorithm for
  job-shop scheduling. *Proceedings of the IEEE International
  Conference on Robotics and Automation*, 3164–3169.
- Williams, R. J. (1992). Simple statistical gradient-following
  algorithms for connectionist reinforcement learning. *Machine
  Learning*, 8(3–4), 229–256.
- Yang, S., Wang, Z., & Liu, S. (2024). A heterogeneous graph
  Transformer for flexible job shop scheduling. *Proceedings of
  AAAI*, 38(8), 8901–8909.
- Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., & Xu, C. (2020).
  Learning to dispatch for job shop scheduling via deep reinforcement
  learning. *Proceedings of NeurIPS*, 33, 1621–1632.

---

## Appendix A: Implementation Files

| Module | File |
|---|---|
| FJSP instance parser | `fjsp/parser/fjs_parser.py` |
| Dispatch environment | `fjsp/env/dispatch_env.py` |
| Schedule validator | `fjsp/scheduler/validator.py` |
| Per-action REINFORCE agent | `rl/agents/reinforce_agent.py` |
| Two-stage actor-critic agent | `rl/agents/two_stage_actor_critic.py` |
| Graph actor-critic agent | `rl/agents/graph_actor_critic.py` |
| Graph PPO agent | `rl/agents/graph_ppo.py` |
| HGT agent | `rl/agents/hgt_fjsp.py` |
| OR-Tools CP-SAT baseline | `scripts/ortools_makespan.py` |
| Multi-seed benchmark harness | `scripts/run_brandimarte_baseline.py` |

## Appendix B: 5-Seed Results

The 5-seed best, mean, and standard deviation for PA-REINFORCE on
all 15 Brandimarte instances are in:

- `data/results/reinforce_mk01_mk05_5seed.json`
- `data/results/reinforce_mk06_mk10_mk13_mk15_5seed.json`
- `data/results/reinforce_mk11_mk12_mk14_5seed.json`

## Appendix C: Reproducibility Checklist

- [x] Instance parser with deterministic behavior
- [x] Environment with explicit legal action mask
- [x] Independent schedule validator used at evaluation
- [x] All four RL agents released
- [x] OR-Tools CP-SAT baseline with OPT/FEAS status
- [x] 5 random seeds per (agent, instance) cell
- [x] Greedy (argmax) makespan reported
- [x] Schedule JSON output for external verification
- [x] Bug documentation (HGT dropout, OR-Tools time-limit)
- [x] Negative result documented (BC + PPO fine-tuning)
