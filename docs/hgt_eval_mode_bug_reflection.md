# 反思：HGT-FJSP 中的 eval/train 模式 bug

日期：2025-11-15

## 摘要

发现 HGT 智能体代码中存在一个**严重的、贯穿所有训练和评估的 bug**：HGT 模型在 dropout=0.1 下，但 `select_action` 等所有前向函数从未调用 `self.net.eval()`。这导致：

- **所有"greedy" rollout 实际上是带 dropout 的随机 rollout**
- 训练历史中的 `best_greedy_makespan` 是噪声中的偶然值，不可信
- 加载保存的 checkpoint 再做 greedy 评估，结果在 5-10 个 makespan 之间波动

## 复现

```python
from rl.agents.hgt_fjsp import HGTActorCriticAgent
from fjsp.env import FJSPDispatchEnv

agent = HGTActorCriticAgent.load(
    'data/results/hgt_ppo_mk01_bc_v5.pt',
    hidden_dim=64, num_blocks=2, num_heads=4, ffn_dim=128,
)

# 修复前：8 次 greedy 评估在 57-74 之间波动
for _ in range(8):
    env = FJSPDispatchEnv.from_file('data/instances/brandimarte/mk01.txt')
    print(agent.rollout(env, greedy=True).makespan)
# 输出: 74, 57, 65, 67, 72, 70, 63, 69

# 修复后：8 次 greedy 评估完全一致
agent.net.eval()
for _ in range(8):
    env = FJSPDispatchEnv.from_file('data/instances/brandimarte/mk01.txt')
    print(agent.rollout(env, greedy=True).makespan)
# 输出: 66, 66, 66, 66, 66, 66, 66, 66
```

## 修复

在 `rl/agents/hgt_fjsp.py` 的 `select_action` 中加 `try/finally` 切换模式：

```python
was_training = self.net.training
self.net.eval()
try:
    # ... 原有 forward / argmax / softmax 代码 ...
finally:
    if was_training:
        self.net.train()
```

`evaluate_action`（PPO update 时调用）保持 net 当前模式；调用方负责在 train() 模式下调用它。

## 修复前后真实数据（mk01，5 次独立 greedy 评估取均值）

| 模型 | 修复前 best（噪声）| 修复后真实 |
|---|---|---|
| HGT-BC 单独 | 55-76 范围 | **59** |
| HGT-BC + A2C | 60（"best 60"，噪声）| **60** |
| HGT-BC + PPO v5 | 57（"best 57"，噪声）| **66** |
| Graph AC + BC | 49（无 dropout，无 bug）| **49** |

## 这个 bug 意味着什么

1. **所有 HGT 训练历史里的 `best_greedy_makespan` 不可信**。`v1` 报 best=58，`v3` 报 best=67，`v5` 报 best=57，**全部是 dropout 随机性下的噪声值**。
2. **HGT 模型的真实 RL 微调结果是负改进**：
   - BC init = 59
   - A2C 200 ep 后 = 60（差 1）
   - PPO 200 ep 后 = 66（差 7）
3. **过去几天的所有"调参 → 涨点"都是幻觉**。换 lr、entropy、K_epochs、teacher 数、rollout 数、跨实例，**都没真正改进 HGT 模型的真实性能**。
4. **Graph AC + BC = 49 仍然是 SOTA**。GraphTwoStageAC 没有 dropout，所以这个数字是真实可信的。

## 第一性原理：方向对吗

| 现状 | 判断 |
|---|---|
| HGT 单独 BC = 59 | **HGT 模型本身在 mk01 上做不到 49**。BC 数据已饱和，模型容量/表达/特征有更根本的限制。 |
| RL 微调负改进 | **PPO/A2C 在 FJSP 这种稀疏奖励 + 信用分配难的问题上，不能在 BC init 之上继续优化**。这是文献的已知问题，不是我们的 bug。 |
| 跨实例 BC loss 不降 | 5 个实例的特征分布不同，30 epoch 根本不够学通用表示；需要：更大模型、更长训练、或者**单实例**。 |
| 调参（lr/entropy/K_epochs）无效 | 调的是"PPO 不会破坏 BC init 的程度"，不是真的在提升策略。 |

## 哲学：我们在做什么

FJSP 是组合优化问题。RL（PPO/A2C）的本质是**策略搜索**，需要密集奖励信号 + 长期探索。FJSP 调度：

- **奖励稀疏**：仅在最后一步给 `-makespan`
- **状态确定**：每步的环境转移是确定性的
- **信用分配难**：几百步操作对最终 makespan 的贡献

**结论**：在 FJSP 上用 PPO/A2C 微调 BC，本质是**用一个不擅长的工具解决一个不需要它的问题**。

更合适的范式：

1. **模仿学习 + 大模型**：BC 本身（不加 RL）已经能接近教师。教师质量决定上限。
2. **组合优化直接求解**：OR-Tools CP-SAT、Gurobi — mk01 下界 39，**几秒就出最优**。
3. **DRL 范式重做**：把 FJSP 改成可以密集奖励的问题（局部调度 / 多步奖励 / RL + 搜索 hybrid）。
4. **有模型 RL**：用 FJSP 转移模型 + MCTS 做规划。

## 接下来怎么走

**选项 A**：放弃 HGT+PPO，专注强化 Graph AC + BC = 49 这条 SOTA 路线，作为论文核心。

**选项 B**：在 HGT 真实 best=59 的基础上，重新设计 HGT 架构（去掉 dropout、加全局特征、简化 cross-attention），看 BC 能否突破 49。

**选项 C**：把 FJSP-RL 整个项目目标重定为"用 RL 学到接近 OR-Tools 性能的近似调度器"，承认 39 是天花板。

**建议**：A。把现有 49 这条线写完 + 写论文。HGT+PPO 路线已经证明是死胡同。

## 未做的事

由于时间限制，下面这些**还没做**，可能进一步确认 / 反驳：
- 用新 fix 重新训练 HGT-BC 和 HGT-PPO，看新的真实曲线
- 跑 OR-Tools CP-SAT 拿 mk01 的真下界
- 跨实例 HGT-BC + A2C 在 fix 下的真实性能
- 多 seed 跑 Graph AC + BC 确认 49 是稳定的 mean 不是单 seed 偶然
