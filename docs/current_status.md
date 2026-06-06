# 当前状态综合报告（2025-11-15）

## SOTA

**REINFORCE 100 ep on Brandimarte mk01-05**:
- 5 seed 平均 0 std（mk01 完美稳定 = 43）
- mk01: 43, OR-Tools=40, gap +3
- mk02: 28, OR-Tools=27, gap +1
- mk03: 216, OR-Tools=204, gap +12
- mk04: 79, OR-Tools=60, gap +19
- mk05: 180, OR-Tools=173, gap +7

详见 `reinforce_5seed_benchmark.md`。

## 已修复的关键 bug

- HGT agent `select_action` 永远没调 `self.net.eval()`，导致 HGT dropout=0.1 的所有"greedy"评估实际是随机的。所有 HGT 训练历史里的 `best_greedy_makespan` 都是噪声。修复后真实数据：HGT-BC=59, HGT-BC+PPO=66, HGT-BC+A2C=60。修复 commit: `60570ca`。

## 已知死路

- **HGT + PPO/A2C 微调 BC init** = 负改进（已证明）
- **14-feature 化的 graph encoder** = 反而让 BC+AC 退化（旧 9-feature BC+AC=49，新 14-feature BC+AC=57）
- **跨实例 HGT-BC 训练** = loss 不降（1.5h/ep 训 1 epoch）
- **HGT 整体路线** = 投入产出比极低

## 应该做但还没做

- 把当前 SOTA（REINFORCE 100ep = 43/28/216/79/180）整理成论文 baseline
- 跑 mk06-15 看泛化
- 试 REINFORCE + A2C value baseline，看能否缩 mk04/mk03 的 gap
- 用 BC 预训练 + REINFORCE 微调
- 写 README / 文档整合

## 关键文件

- `rl/agents/reinforce_agent.py` — SOTA 智能体 (8-feature MLP + EMA baseline)
- `scripts/ortools_makespan.py` — OR-Tools CP-SAT 真下界
- `reinforce_5seed.py` — 5-seed benchmark 脚本
- `docs/reinforce_5seed_benchmark.md` — 详细结果
- `docs/hgt_eval_mode_bug_reflection.md` — HGT bug 反思
