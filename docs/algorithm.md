# Algorithm Notes

## Current Status

The current RL method is a minimal baseline, not a novel FJSP algorithm yet.

It is useful because it gives the project a complete learning loop:

```text
FJSP instance -> dispatch environment -> legal action mask -> policy network -> schedule -> validator -> makespan
```

This baseline should be treated as the first reproducible reference point before adding stronger or more original methods.

## Method Name

Working name:

```text
REINFORCE Dispatch Policy for FJSP
```

## Decision Model

At each step, the environment exposes all currently schedulable actions:

```text
action = (job, machine)
```

The selected action schedules the next unscheduled operation of `job` on `machine`.

The environment then decodes the operation at its earliest feasible start time:

```text
start = max(job_ready_time[job], machine_ready_time[machine])
end = start + processing_time(job, operation, machine)
```

This keeps every generated schedule legal as long as the policy selects from the valid action set.

## Action Mask

An action is valid only if:

- the job still has an unscheduled next operation;
- the selected machine is eligible for that operation.

Invalid actions are masked out by `available_actions()` and `valid_action_mask()`.

## Policy Network

The policy scores each valid action independently with a small MLP.

Current per-action features:

- job ready time;
- machine ready time;
- operation duration on the selected machine;
- earliest start time;
- earliest finish time;
- operation index within the job;
- remaining operations in the job;
- current makespan.

The action is sampled during training and chosen greedily during evaluation.

## Reward

Training optimizes terminal makespan through REINFORCE.

For policy-gradient update:

```text
episode_reward = -makespan
advantage = episode_reward - moving_baseline
loss = -sum(log_prob(action_t)) * advantage
```

The environment also exposes a step reward:

```text
reward_t = -(new_makespan - old_makespan)
```

but the current trainer uses terminal makespan for the update.

## What Is Original Here?

Current version:

- not a new algorithm;
- not yet a paper-level contribution;
- a clean baseline implementation specialized for this project.

Current project-specific strengths:

- legal schedule generation is built into the environment;
- every output can be checked by an independent validator;
- benchmark scripts record random, heuristic, and RL results together;
- Brandimarte MK instances are already integrated.

## Difference From Stronger Literature Methods

Many stronger FJSP-RL papers use combinations of:

- graph neural networks over operation-machine graphs;
- PPO or actor-critic instead of vanilla REINFORCE;
- disjunctive graph representations;
- attention/Transformer encoders;
- curriculum learning over generated instances;
- learned dispatching rules that generalize across instance sizes;
- multi-objective rewards such as makespan, machine load, delay, or energy.

Our current method is simpler:

- MLP action scorer;
- handcrafted action features;
- vanilla REINFORCE;
- earliest-start deterministic decoder;
- fixed benchmark instance training.

## First Upgrade: Two-Stage Actor-Critic

The project now includes a first step beyond the basic REINFORCE baseline:

```text
TwoStageActorCriticAgent
```

It decomposes the action into:

```text
1. select job
2. select eligible machine for that job
```

and adds a value critic for lower-variance policy-gradient training.

This is still not a full graph/PPO method, but it is closer to common FJSP-RL action decompositions than the flat `(job, machine)` scorer.

Initial MK01 check:

```text
Two-stage actor-critic, seed=0, 100 episodes: makespan 48
```

The schedule was validated by the independent schedule validator.

## Recommended Next Innovations

Good next research directions:

1. Replace handcrafted action features with graph features.
2. Use PPO or actor-critic for lower-variance training.
3. Train across multiple Brandimarte instances instead of one instance.
4. Add operation-machine bipartite graph encoding.
5. Add dispatching-rule imitation pretraining.
6. Add curriculum from tiny instances to MK01-MK15.
7. Compare against OR-Tools, tabu search, and classic dispatching rules.

## Verified Results

### Brandimarte MK01 (single instance)

```text
known optimum: 40
earliest-finish heuristic: 57
random mean: 104.8
REINFORCE dispatch policy, 3 seeds x 200 episodes: 43 mean / 43 best / 0.0 std
Two-stage AC, seed 0, 100 episodes: 48
Graph AC, default config, 500 episodes: 54
Graph AC, tuned config (lr 1e-3, hidden 128, rounds 3), 200 episodes: 60
Graph PPO, default config, 200 episodes: 85
```

Full single-instance study: `data/results/comparison_three_agents_mk01_100ep.md`.

### Brandimarte MK01–MK10 (10 instances, 4 agents)

```text
script: scripts/run_brandimarte_baseline.py
budget: REINFORCE 50 ep / AC 50 ep / GAC 50 ep / GPPO 30 ep
seed: 0
```

Greedy makespan, lower is better. The full table is in
`data/results/baseline_brandimarte_matrix.md`. Summary:

- **REINFORCE is the best agent on 10/10 instances** at this budget.
- Two-stage AC is 2nd on 8/10 instances.
- Graph AC and Graph PPO are systematically the worst — at 50 ep the graph
  encoder is under-trained, not just lower-variance.
- REINFORCE beats the earliest-finish heuristic on 10/10 instances
  (gap from −1.1% on mk04 to −54.8% on mk02).
- REINFORCE is within 10% of the known optimum on mk01 (+7.5%),
  mk03 (+9.3%) and mk08 (+3.1%).

The matrix is the new single-source-of-truth for the per-instance ranking and
is the basis for the next-round decisions in `docs/large_instances.md`.
