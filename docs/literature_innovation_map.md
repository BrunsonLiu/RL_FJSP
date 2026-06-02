# FJSP-RL Literature Innovation Map

## Short Answer

Most recent FJSP-RL papers innovate in one or more of these modules:

1. Problem representation.
2. Neural architecture.
3. Action decomposition.
4. RL algorithm and training stability.
5. Dynamic or realistic shop-floor constraints.
6. Generalization and large-scale performance.
7. Hybridization with heuristics or expert rules.

Our current project only has a basic dispatch environment and a vanilla REINFORCE policy. It is a baseline, not a paper-level method yet.

## Common Innovation Areas

### 1. Representation Innovation

Papers often formulate FJSP as a graph:

- disjunctive graph;
- operation-machine bipartite graph;
- heterogeneous graph;
- meta-path graph;
- dynamic graph that changes during scheduling.

Why it matters:

- FJSP has two coupled decisions: operation sequencing and machine assignment.
- Graphs naturally encode precedence constraints, machine eligibility, and machine competition.

Project gap:

- We currently use handcrafted action features, not a graph representation.

### 2. Network Architecture Innovation

Common choices:

- Graph Neural Network;
- Graph Attention Network;
- heterogeneous graph attention;
- dual attention over operations and machines;
- Transformer encoder;
- residual graph attention.

Why it matters:

- Operations and machines have different roles.
- Machine competition and operation precedence are different relation types.
- Attention helps the model focus on critical operations or bottleneck machines.

Project gap:

- We currently use a small MLP action scorer.

### 3. Action-Space Innovation

FJSP actions are naturally multi-part:

```text
choose operation/job + choose machine
```

Papers often design:

- two-stage policies;
- multi-action policies;
- collaborative job-agent and machine-agent policies;
- pointer networks for job and machine selection;
- masked action decoding.

Project gap:

- We currently use one flat action `(job, machine)`.
- The mask is correct, but not architecturally sophisticated.

### 4. RL Algorithm Innovation

Common algorithms:

- PPO;
- actor-critic;
- SAC;
- DQN variants;
- multi-agent RL;
- hierarchical RL.

Why it matters:

- Vanilla REINFORCE has high variance.
- Large FJSP instances are unstable under weak policy-gradient training.

Project gap:

- We currently use vanilla REINFORCE with a moving baseline.

### 5. Dynamic and Realistic Constraints

Many newer papers move beyond static FJSP:

- random job arrivals;
- machine breakdowns;
- transport constraints;
- AGV scheduling;
- setup times;
- due dates;
- energy or carbon objectives;
- multi-objective scheduling.

Why it matters:

- Static makespan minimization is classic but crowded.
- Dynamic constraints are closer to smart manufacturing.

Project gap:

- We currently solve static makespan-only FJSP.

### 6. Generalization Innovation

Better papers often emphasize:

- training on generated instances;
- testing on unseen benchmark instances;
- scale generalization;
- cross-distribution generalization;
- fast inference compared with metaheuristics.

Project gap:

- We currently train per instance.
- We have not yet built multi-instance training.

### 7. Hybrid / Expert-Guided Innovation

Some papers combine learning with:

- priority dispatching rules;
- expert demonstrations;
- tabu search or local search;
- genetic algorithms;
- heuristic rewards;
- repair or improvement operators.

Why it matters:

- Pure RL can be unstable.
- Heuristics provide useful prior knowledge.

Project gap:

- We currently compare against a heuristic but do not learn from it.

## Where We Can Innovate

Best directions for this project:

1. Graph-based state encoder for operations and machines.
2. Two-head action policy: one head for job/operation, one head for machine.
3. PPO or actor-critic instead of REINFORCE.
4. Imitation pretraining from earliest-finish and other dispatch rules.
5. Multi-instance training over Brandimarte plus Hurink/Fattahi.
6. Dynamic FJSP extension with random job arrivals or machine breakdowns.

Most practical next step:

```text
Graph encoder + PPO + two-stage masked action policy
```

That would be much closer to current literature while still leaving room for project-specific contribution.

