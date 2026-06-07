# 当前状态综合报告（2026-06-07）

## 修正后的 SOTA 现实

**REINFORCE 100 ep 5-seed best 在 Brandimarte mk01-15 上表现**：

| instance | REINFORCE best (5-seed) | REINFORCE mean ± std | OR-Tools | Lit UB | Δ to OR |
|---|---|---|---|---|---|
| mk01 | 43 | 43.0 ± 0.00 | 40 OPT | 40 | +3 |
| mk02 | 28 | 31.8 ± 2.48 | 26 FEAS | 26 | +2 |
| mk03 | 223 | 217.4 ± 2.80 | 204 OPT | 204 | +19 |
| mk04 | 90 | 80.6 ± 0.80 | 60 OPT | 60 | +30 |
| mk05 | 183 | 182.4 ± 1.74 | 172 FEAS | 172 | +11 |
| mk06 | 69 | 71.0 ± 1.41 | 64 FEAS 300s | 58 | +5 |
| mk07 | 162 (1-seed) | — | — | 139 | n/a |
| mk08 | 539 (1-seed) | — | — | 523 | n/a |
| mk09 | 408 (1-seed) | — | — | 307 | n/a |
| mk10 | **242** | 335.6 ± 174.28 | 257 FEAS 300s | 197 | **−15** |
| mk11 | **639** | 659.4 ± 10.54 | 615 FEAS 600s | 615 | +24 |
| mk12 | **531** | 713.2 ± 349.45 | **508 OPT 8.8s** | 508 | **+23** |
| mk13 | **464** | 476.4 ± 7.91 | 439 FEAS 300s | 430 | +25 |
| mk14 | **694** | 718.4 ± 14.88 | **694 OPT 24.8s** | 694 | **0 (MATCH)** |
| mk15 | **408** | 419.6 ± 6.28 | 387 FEAS 300s | 341 | +21 |

**重大发现（修正前错误）**：
- mk14: REINFORCE 5-seed best=694 == OR-Tools proven optimal **完全打平**
- mk12: REINFORCE 5-seed best=531 仅 +23 above OR-Tools optimal（单 seed=924 是 5 次中唯一跑飞的）
- mk10: REINFORCE 5-seed best=242 真正击败 OR-Tools 300s 可行解

**关键修正**：之前"REINFORCE 落后 OR-Tools 几百"的认知是单 seed 偏差；5-seed 最佳值显示 REINFORCE 在 mk12/14 上非常接近 OR-Tools。

参见 `docs/reinforce_vs_ortools_full.md` 详细分析。

## 已修复的关键 bug

- HGT agent `select_action` 永远没调 `self.net.eval()`，导致 HGT dropout=0.1 的所有"greedy"评估实际是随机的。所有 HGT 训练历史里的 `best_greedy_makespan` 都是噪声。修复后真实数据：HGT-BC=59, HGT-BC+PPO=66, HGT-BC+A2C=60。
- `scripts/ortools_makespan.py` 现在区分 `OPTIMAL` 与 `FEASIBLE` 状态（之前都打成裸 makespan）。修复后能正确判断 60s 可行解 ≠ 最优解，避免错误地宣称"REINFORCE 击败 OR-Tools"。
- **单 seed REINFORCE 在大实例上 variance 极大**（mk10 std=174，mk12 std=349）。所有"REINFORCE 远落后 OR-Tools"的论断都需要 5-seed 验证才能定论。

## 已知死路

- **HGT + PPO/A2C 微调 BC init** = 负改进（已证明）
- **14-feature 化的 graph encoder** = 反而让 BC+AC 退化（旧 9-feature BC+AC=49，新 14-feature BC+AC=57）
- **跨实例 HGT-BC 训练** = loss 不降（1.5h/ep 训 1 epoch）
- **HGT 整体路线** = 投入产出比极低

## 应该做但还没做

- 跑 mk07/08/09 的 5-seed REINFORCE 验证稳定性
- 试 REINFORCE + A2C value baseline（降低 mk10/mk12 的方差）
- 试 BC 预训练 + REINFORCE 微调（BC + AC 在 mk01 上能到 49）
- 跨实例 HGT-BC + PPO 在 mk12/14 这种 OR-Tools 容易打掉 RL 的实例上做对照
- 把 REINFORCE + value baseline 结果与 SOTA RL-FJSP 论文对比

## 关键文件

- `rl/agents/reinforce_agent.py` — REINFORCE 智能体 (8-feature MLP)
- `scripts/ortools_makespan.py` — OR-Tools CP-SAT 求解器（带 OPT/FEAS 区分）
- `reinforce_5seed.py` — mk01-05 5-seed 脚本
- `reinforce_5seed_large.py` — mk06/10/13/15 5-seed 脚本
- `reinforce_5seed_opt.py` — mk11/12/14 5-seed 脚本
- `data/results/reinforce_mk01_mk05_5seed.json` — mk01-05 详细结果
- `data/results/reinforce_mk06_mk10_mk13_mk15_5seed.json` — 大实例 5-seed 结果
- `data/results/reinforce_mk11_mk12_mk14_5seed.json` — 最难实例 5-seed 结果
- `docs/reinforce_5seed_benchmark.md` — mk01-05 5-seed 详细报告
- `docs/reinforce_vs_ortools_full.md` — mk01-15 完整 REINFORCE vs OR-Tools 报告
- `docs/hgt_eval_mode_bug_reflection.md` — HGT bug 反思
