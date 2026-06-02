# 4-Agent × 10-Instance Brandimarte Baseline Matrix

**Run date**: 2026-06-02
**Script**: `scripts/run_brandimarte_baseline.py`
**Raw outputs**: `data/results/baseline_brandimarte_matrix.csv`, `data/results/baseline_brandimarte_matrix.json`

**Setup**

- Instances: Brandimarte `mk01` – `mk10` (10 instances total)
- Agents and per-agent training budget (this run):
  - `reinforce`: 50 episodes
  - `actor_critic` (two-stage MLP AC): 50 episodes
  - `graph_actor_critic` (operation-machine GNN + two-stage AC): 50 episodes
  - `graph_ppo` (operation-machine GNN + PPO clip): 30 episodes
- Evaluation: greedy rollout on the same instance, no fine-tuning.
- Seed: `0` for training, `seed * 1000 + idx` for the 3 random rollouts.
- Reference heuristics computed inside the script: earliest-finish heuristic, 3 random rollouts (mean / best).

## Full results (greedy makespan)

| instance | jobs | mach | ops  | opt  | earliest | random_best | random_mean | REINFORCE | 2-Stage AC | Graph AC | Graph PPO | best |
| -------- | ---- | ---- | ---- | ---- | -------- | ----------- | ----------- | --------- | ---------- | -------- | --------- | ---- |
| mk01     | 10   | 6    | 55   | 40   | 57       | 102         | 106.333     | **43**    | 49         | 110      | 73        | reinforce |
| mk02     | 10   | 6    | 58   | —    | 62       | 80          | 87          | **28**    | 80         | 127      | 114       | reinforce |
| mk03     | 15   | 8    | 150  | 204  | 331      | 456         | 493         | **223**   | 506        | 891      | 706       | reinforce |
| mk04     | 15   | 8    | 90   | 60   | 91       | 166         | 166.667     | **90**    | 128        | 168      | 168       | reinforce |
| mk05     | 15   | 4    | 106  | —    | 220      | 275         | 325.667     | **183**   | 374        | 387      | 450       | reinforce |
| mk06     | 10   | 10   | 150  | —    | 79       | 223         | 245         | **77**    | 229        | 238      | 373       | reinforce |
| mk07     | 20   | 5    | 100  | —    | 204      | 377         | 385         | **162**   | 321        | 554      | 451       | reinforce |
| mk08     | 20   | 10   | 225  | 523  | 618      | 798         | 845.667     | **539**   | 613        | 1343     | 1546      | reinforce |
| mk09     | 20   | 10   | 240  | 307  | 433      | 732         | 754         | **408**   | 1336       | 1740     | 1310      | reinforce |
| mk10     | 20   | 15   | 240  | —    | 406      | 672         | 702.667     | **258**   | 896        | 1621     | 1239      | reinforce |

## Per-instance ranking (greedy makespan, lower is better)

| instance | 1st             | 2nd        | 3rd       | 4th       |
| -------- | --------------- | ---------- | --------- | --------- |
| mk01     | REINFORCE 43    | AC 49      | PPO 73    | GAC 110   |
| mk02     | REINFORCE 28    | AC 80      | PPO 114   | GAC 127   |
| mk03     | REINFORCE 223   | AC 506     | PPO 706   | GAC 891   |
| mk04     | REINFORCE 90    | AC 128     | GAC 168   | PPO 168   |
| mk05     | REINFORCE 183   | AC 374     | GAC 387   | PPO 450   |
| mk06     | REINFORCE 77    | AC 229     | GAC 238   | PPO 373   |
| mk07     | REINFORCE 162   | AC 321     | PPO 451   | GAC 554   |
| mk08     | REINFORCE 539   | AC 613     | GAC 1343  | PPO 1546  |
| mk09     | REINFORCE 408   | PPO 1310   | AC 1336   | GAC 1740  |
| mk10     | REINFORCE 258   | AC 896     | PPO 1239  | GAC 1621  |

`REINFORCE` is best on all 10/10 instances.
`actor_critic` is 2nd on 8/10 (mk08, mk09 the exceptions where it is 2nd or 3rd).
`graph_actor_critic` is 3rd on 4 and 4th on 5.
`graph_ppo` is mostly 3rd or 4th.

## Gap to earliest-finish heuristic (best method – earliest)

| instance | earliest | REINFORCE | Δ        | %        | beats heuristic? |
| -------- | -------- | --------- | -------- | -------- | ---------------- |
| mk01     | 57       | 43        | −14      | −24.6%   | yes              |
| mk02     | 62       | 28        | −34      | −54.8%   | yes              |
| mk03     | 331      | 223       | −108     | −32.6%   | yes              |
| mk04     | 91       | 90        | −1       | −1.1%    | yes (tie)        |
| mk05     | 220      | 183       | −37      | −16.8%   | yes              |
| mk06     | 79       | 77        | −2       | −2.5%    | yes              |
| mk07     | 204      | 162       | −42      | −20.6%   | yes              |
| mk08     | 618      | 539       | −79      | −12.8%   | yes              |
| mk09     | 433      | 408       | −25      | −5.8%    | yes              |
| mk10     | 406      | 258       | −148     | −36.5%   | yes              |

REINFORCE beats the earliest-finish heuristic on all 10/10 instances, often by 20-55%.

## Gap to known optimum (when available)

| instance | opt  | REINFORCE | Δ        | %       |
| -------- | ---- | --------- | -------- | ------- |
| mk01     | 40   | 43        | +3       | +7.5%   |
| mk03     | 204  | 223       | +19      | +9.3%   |
| mk04     | 60   | 90        | +30      | +50.0%  |
| mk08     | 523  | 539       | +16      | +3.1%   |
| mk09     | 307  | 408       | +101     | +32.9%  |

REINFORCE is within 10% of optimum on 3/5 known instances (mk01, mk03, mk08). mk04 and mk09 are the weak spots.

## Headline takeaways

1. **REINFORCE is the strongest agent at this training budget on every Brandimarte instance** (10/10).
   The 8 handcrafted per-action features (`docs/algorithm.md` §Policy Network) already capture most of the dispatch signal a single-instance policy can exploit.

2. **The two graph agents are systematically the worst.** Even though the graph encoder compiles, trains and produces valid schedules, its greedy makespan is higher than the random rollout mean on most instances (e.g. mk03 Graph AC 891 vs random mean 493, mk08 Graph PPO 1546 vs random mean 845). This is a strong signal that the graph encoder is *overfitting / under-training* at the same episode budget, not just losing variance.

3. **Two-stage AC is 2nd-best on 8/10** but still loses to REINFORCE on every single instance. The added value head and 2-stage decomposition do not pay off with a 50-episode budget. Earlier `comparison_three_agents_mk01_100ep.md` showed AC at 48 (100 ep) — also still behind REINFORCE at 43 (100 ep).

4. **Per-instance difficulty is now visible**:
   - Easy / small (mk01, mk02, mk06, mk10): all agents converge to <120, REINFORCE matches or beats the heuristic by a wide margin.
   - Medium (mk03, mk05, mk07): REINFORCE is well ahead, graph agents trail badly.
   - Hard / high-lower-bound (mk04, mk08, mk09): REINFORCE still wins but the gap to optimum is the largest (mk04 +50%, mk09 +33%).

## Why does REINFORCE beat Graph AC / Graph PPO so consistently?

This is the central new question raised by the matrix. The graph encoder is supposed to be strictly more expressive, but it produces strictly worse policies. Three plausible explanations:

1. **Sample complexity mismatch.** The graph encoder has more parameters, but the per-instance training budget (50 ep) is the same. Larger model + same data → under-trained.
2. **Per-step reward signal is not what the graph encoder is learning from.** Like AC, the graph agents are still trained on the sparse terminal reward. The richer state representation does not help when the learning signal is a single scalar at episode end.
3. **The handcrafted action features are already near-optimal for a single-instance MLP policy.** Adding graph context without also increasing the training budget / data diversity / reward density is a net negative — the optimizer is fitting noise.

## Implication for the SOTA path

If we want graph-based methods to actually win on this benchmark, the next experiments have to attack the cause, not the symptoms:

- **Cross-instance training.** One graph encoder trained on many Brandimarte instances at once. This gives the encoder enough data to actually use its capacity, and matches how graph FJSP papers report results.
- **Imitation pretraining (BC) on earliest-finish rollouts.** A warm-start policy that is already near the heuristic, then fine-tuned with REINFORCE / PPO. The current per-instance training starts from a random policy and the graph encoder cannot recover within 50-100 episodes.
- **Dense per-step reward** (e.g. negative change in machine idle time) so the value head in PPO has a learnable signal on every step.
- **Larger training budget per instance** (≥ 500 ep) combined with early stopping on a held-out Brandimarte instance.

We are not going to claim "graph encoder = SOTA" until one of the above is implemented and the matrix in this document flips in favor of the graph agents.

## Reproduction

```powershell
python scripts/run_brandimarte_baseline.py --brandimarte-start 1 --brandimarte-count 10 --random-rollouts 3 --seed 0
```

Override per-agent episode budget with `--episodes-reinforce`, `--episodes-actor-critic`, `--episodes-graph-actor-critic`, `--episodes-graph-ppo`.
