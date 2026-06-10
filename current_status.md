# Current Status (2026-06-10)

This document is a snapshot of where the project stands today. It is
deliberately kept concise so a new reader can understand the state
of the work in a few minutes.

## 1. Headline result

Combined REINFORCE + ILS + SA + (on the three largest instances)
tabu search, applied to the Brandimarte MK01–MK15 benchmark:

- **New state-of-the-art on MK13** (416 vs previous best 430, –14
  makespans, –3.3 %).
- **Ties the literature best on 4 of 15 instances**: MK03, MK08,
  MK12, MK14.
- **Mean gap to literature: 5.64 %** across all 15 instances
  (total absolute gap 106 makespan units).
- **Beats the earliest-finish (EF) heuristic on every instance**
  by 12–39 %.

The full SOTA matrix is in `paper/main.md` §6.5 and the underlying
JSON in `data/results/sota_final.json`. The pipeline is fully
reproducible; the commands are listed in `AGENTS.md` §"SOTA
Pipeline".

## 2. Repository structure (the parts that matter)

```
paper/main.md                    # full paper
paper/outline_zh.md              # Chinese outline of the paper
AGENTS.md                        # agent guide, including SOTA commands

fjsp/
  parser/                        # FJSP instance parser, deterministic
  scheduler/                     # baselines, dispatchers, validators
    local_search.py              # LS, ILS, SA, TS, critical-path perturb
  env/                           # RL environment (dispatch view)

rl/
  agents/                        # REINFORCE, AC, PPO, graph-AC, graph-PPO,
                                 # BC, BC+PPO, BC+AC
  models/                        # MLP, GNN, HGT

data/
  instances/brandimarte/         # MK01..MK15
  results/                       # schedules, metrics, SOTA JSON

scripts/                         # validation, smoke tests, plotting,
                                 # benchmark driver
```

## 3. Three post-processors, in order of when they enter the pipeline

1. **Iterated local search (ILS)** — uses four neighbourhoods
   (reassign, swap-machine, swap-order-same-machine,
   swap-order-across-machines) with first-improvement local search,
   random swap perturbation, and best-improvement + critical-path
   perturbation for the aggressive variant. This is the workhorse
   and contributes the bulk of the makespan reduction.
2. **Simulated annealing (SA)** — final refinement of the best
   ILS result, 2000–3000 iterations, cooling rate 0.99.
3. **Tabu search (TS)** — applied only on the three largest
   instances (MK09, MK10, MK15). With a tabu tenure of
   `ceil(sqrt(N))`, candidate sampling of 200 neighbours per
   iteration, and an aspiration criterion. Closes an extra 7–10
   makespan units on MK10 and MK15.

Full implementation is in `fjsp/scheduler/local_search.py`.
Public entry points: `local_search`, `iterated_local_search`,
`simulated_annealing`, `tabu_search`,
`critical_path_perturb`.

## 4. RL agents compared

- **PA-REINFORCE** (per-action REINFORCE, our strong baseline).
- **AC** (vanilla actor-critic with shared backbone).
- **Graph-AC** (two-stage actor-critic with a heterogeneous graph
  encoder).
- **Graph-PPO** (two-stage PPO with the same encoder).
- **HGT** (heterogeneous graph Transformer) — kept for
  completeness; we document two reproducibility issues we found
  with the published-style setup in §2 of the paper.
- **BC + AC / BC + PPO** — behavioural cloning on the
  earliest-finish dispatch rule, then fine-tuning. BC + AC reaches
  greedy 49 on MK01; BC + PPO stays at 57 because PPO cannot break
  the BC initialization with the default sparse reward.

PA-REINFORCE is the strongest of the four learned agents on every
Brandimarte instance at the budgets we tested.

## 5. Validation

- All schedules that contribute to the FINAL column of Table 4 are
  validated by `scripts/validate_schedule.py` immediately after
  they are generated. A failed validation is treated as a missed
  run and excluded.
- The instance parser is independent from the RL code
  (`fjsp/parser/`) and the schedule validator is independent from
  the model code (`fjsp/scheduler/validator.py`). The validator
  is used as a judge.

## 6. Reproducibility

End-to-end SOTA run, in five commands:

```powershell
python sota_reinforce_ils.py
python aggressive_ils_hard.py
python merge_sota.py
python tabu_hard_v2.py
python merge_ts_sota.py
```

Output JSONs:

- `data/results/sota_reinforce_ils.json`
- `data/results/aggressive_ils_hard.json`
- `data/results/tabu_results.json`
- `data/results/sota_final.json` (the merged SOTA matrix)

The paper commands are listed in `AGENTS.md` §"SOTA Pipeline" and
in `paper/main.md` §6.5.

## 7. Known limitations and where the next push could go

- **Large instances (MK15, MK10, MK09) still have a 25–30 unit gap
  to the literature.** TS closes 7–10 of those units but cannot
  close the whole gap. Likely need qualitatively new
  neighbourhoods (block re-insertion of critical-path subsequences,
  cross-job block moves).
- **Per-action feature ablation (Section 6.6) is only on MK01.**
  Extending it to MK05, MK10, MK15 would strengthen the
  ablation story.
- **The literature best-known on MK11 (615) is still
  unconfirmed.** We achieve 619, very close, but we have not been
  able to close that last 4-unit gap.

## 8. What is NOT in this document

- Detailed per-instance training curves (those live in
  `data/results/`, with one JSON per run).
- The HGT and behavioural-cloning narrative (see `paper/main.md`
  §4 and §6.7).
- The OR-Tools baseline methodology discussion (see
  `paper/main.md` §2, §4, and §6.2).
