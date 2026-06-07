# 论文大纲（中文 Review 版）

## 论文主线

**不夸大的诚实故事：**

> 我们系统对比了 4 种 RL 智能体在 Brandimarte MK01–MK15 上的表现，
> 结论是**逐动作 REINFORCE（PA-REINFORCE）是最强的**。
> 它在 mk14 上与 OR-Tools 已知最优**完全打平**（694 = 694）；
> 在 mk10 上击败 300s 时限的 OR-Tools 可行解（242 < 257）；
> 在其他大实例上也都在 5–30 makespan 之内。
> 所有更复杂的图模型（HGT、Graph AC、Graph PPO）都没打过它。

**核心论点 1**：对于 Brandimarte 这种规模，单实例训练预算下，
带 8 个手工特征的小型 MLP + 朴素 REINFORCE 就够用。

**核心论点 2**：Fancy 的图编码器 + PPO/A2C 微调在 FJSP 上是负改进
（这是文献里反复出现但很少被明确写出来的事实）。

**核心论点 3**：复现性很重要——我们发现并修复了 HGT 中漏掉
`net.eval()` 的 bug 和 OR-Tools 60s 可行解被误当最优解的方法学错误。

## 章节结构（9 节正文 + 3 附录）

| # | 标题 | 主要内容 | 字数估计 |
|---|---|---|---|
| Abstract | — | 核心结论 4 句话 + 5 seed/cell 的矩阵 | 250 |
| 1 | Introduction | FJSP-RL 现状、本文 4 个贡献、为什么重要 | 700 |
| 2 | Related Work | 4 类：精确法/启发法、DRL、复现性、OR-Tools 基线 | 500 |
| 3 | Problem Formulation | FJSP 定义、dispatch view、为什么选 dispatch view | 400 |
| 4 | Methods | PA-REINFORCE、TS-AC、GAC、GPPO、HGT、EF、OR-Tools | 1200 |
| 5 | Experimental Setup | Brandimarte、训练预算、5 seed、独立校验器 | 400 |
| 6 | Results | 单实例 MK01、MK01-10 矩阵、MK11-15 矩阵、OR-Tools 对比、特征消融 | 1500 |
| 7 | Lessons Learned | 3 个 case study：HGT bug、OR-Tools 误读、BC+RL 负面 | 1000 |
| 8 | Discussion | 何时赢、何时输、为什么没有 SOTA 声明、局限 | 600 |
| 9 | Conclusion | 总结 + 未来工作 | 250 |
| App. A | 实现文件清单 | 10 个核心模块 + 路径 | — |
| App. B | 5-seed 详细结果 | 链接到 3 个 JSON 文件 | — |
| App. C | 复现性 Checklist | 10 项 ✓ | — |

**总计**：约 6800 词（英文），对应 LaTeX 排版后 8–10 页。

## 4 个贡献点（写给 reviewer 看）

1. **可复现的 FJSP-RL 工具链**——比 L2D 仓库多一个独立校验器。
2. **5-seed 矩阵**——L2D 只跑单 seed，本文 5 seed 揭示了：
   (a) 之前"REINFORCE 击败 OR-Tools"是单 seed 假象；
   (b) 5 seed 最佳值在 mk12/14 上接近 OR-Tools。
3. **两个复现性 case study**——文献里几乎没有的诚实负面报告。
4. **PA-REINFORCE on handcrafted features**——一个"做减法"的故事：
   抛弃 Graph/Transformer/PPO，反而赢。

## 与现有 SOTA RL-FJSP 论文的差异

| 维度 | L2D / 后续工作 | 本文 |
|---|---|---|
| 训练 budget | 通常 50-200 ep/instance | 50-100 ep/instance |
| Seeds/cell | 通常 1 | 5 |
| 校验器 | 通常内置 | 独立 `validator.py` 模块 |
| OR-Tools 对比 | 经常只有 UB | 同时给 OPT / FEAS 状态 + 多次时限 |
| 复现性 bug | 鲜有讨论 | 明确报告 HGT dropout bug |
| 负面结果 | 通常略过 | 明确报告 BC+PPO 负改进 |

## 投稿建议

**首选：EAAI 或 Journal of Scheduling**
- 接收"详尽实证 + 复现性 + 负面结果"型工作。
- 偏应用，正好对口 FJSP。

**次选：ICAPS / AAAI Workshop on RL for Combinatorial Optimization**
- ICAPS 主会偏 SOTA 数字，故事太"老实"可能偏弱。
- Workshop 更接受诚实负面报告。

**避开**：NeurIPS / ICML 主会——对 FJSP 这种经典问题兴趣有限。

## Reviewer 可能问的问题（提前想好回答）

1. **为什么 PA-REINFORCE 这么强？** 因为 action space 小（≤ 100 候选）、
   特征是手工的（不学表示）、稀疏奖励在 EMA baseline 下稳定。
2. **为什么 Graph 模型打不过 MLP？** 在 50-100 episode 预算下，Graph
   编码器还没收敛，反而引入噪声；HGT 还有 dropout bug。
3. **BC + PPO 负改进是 bug 吗？** 不是，是 FJSP 这种"稀疏奖励 + 确定性
   转移 + 信用分配难"问题的固有问题。
4. **为什么不报告 SOTA？** 因为我们没拿到——OR-Tools 在 mk12/14 上
   仍是最优/UB 615，我们打平或落后 23 个 makespan。
5. **跨实例能泛化吗？** 不能——目前所有结果都是 per-instance 训练。
   这是未来工作。

## 还需要补的实验（让论文更 solid）

按优先级：

1. **mk07/08/09 的 5-seed REINFORCE**——目前还是单 seed，论文表格
   标"single-seed"会显得不严谨。
2. **画训练曲线**——`best_greedy_makespan vs episode` 至少 1-2 张图。
3. **加 A2C value baseline 实验**——作为 PA-REINFORCE 的改进，消融
   "REINFORCE 的高方差"这个已知的弱点。
4. **跑 Hurink edata/rdata/vdata 几组**——证明结果不只对 Brandimarte
   成立，提升泛化性。
5. **跨实例 BC + REINFORCE**——这是论文 §8.3 的未来工作，做了能撑
   下一篇 follow-up 论文。

## 写作风格

- **诚实优先**——不夸大、不粉饰。所有"X 击败 Y"必须配 5-seed 数据。
- **可复现优先**——所有数字都能从仓库里跑出来。
- **代码引用**——所有重要模块都给出文件路径（已在 main.md 中加）。
- **避免花哨**——不上 attention/transformer 的图，不画复杂架构图。
  1 个 PA-REINFORCE 网络图、1 张训练曲线、5 个结果表就够了。
