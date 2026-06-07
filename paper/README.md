# 论文目录说明

| 文件 | 内容 |
|---|---|
| `main.md` | 论文全文（英文，Markdown 格式） |
| `outline_zh.md` | 论文大纲与核心论点（中文，便于你 review） |
| `README.md` | 本文件 |

## 论文题目（待定）

英文：*Per-Action REINFORCE for Flexible Job Shop Dispatch: An Empirical Study on the Brandimarte Benchmark*

中文：*逐动作 REINFORCE 用于柔性作业车间调度：在 Brandimarte 基准上的实证研究*

## 论文主线（核心故事）

**诚实的故事，不夸大、不粉饰：**

> 经过多轮实验（4 个 RL 智能体 × 15 个 Brandimarte 实例 × 5 个随机种子），
> 一个带 8 个手工特征的小型逐动作 REINFORCE 策略，
> **比所有图模型（GAC、GPPO、HGT+BC、HGT+BC+PPO、HGT+BC+A2C）都强**，
> **在 mk14 上与 OR-Tools 已知最优完全打平**，
> **在 mk10 上击败 300s 时限的 OR-Tools 可行解**。
> 简单方法就是赢家。

## 论文卖点（4 个贡献点）

1. **可复现的 FJSP-RL 工具链**——解析器、环境、合法动作 mask、独立
   调度合法性校验器、4 个 RL 智能体、OR-Tools 基线、多 seed benchmark
   脚本、合法调度 JSON 输出。
2. **多 seed、多实例的实证对比**——Brandimarte MK01–MK15，5 seed/cell。
3. **两个复现性 case study**——
   (a) HGT 智能体中漏掉 `net.eval()` 导致 dropout 在评估时仍生效，
       所有"greedy"评估其实是随机的；
   (b) 把 60s OR-Tools 可行解误当最优解，错误地宣称"REINFORCE 击败
       OR-Tools"。
4. **诚实的负面结果**——BC + PPO / A2C 在图编码器上微调，反而不如
   纯 BC，更不如逐动作 REINFORCE。

## 论文长度

`main.md` 全文约 843 行 Markdown（英文），预计对应 LaTeX 排版后的
8–10 页（ICAPS / EAAI 风格）。

## 建议投稿方向

| 期刊/会议 | 适配度 | 理由 |
|---|---|---|
| **EAAI (Engineering Applications of AI)** | ★★★★★ | 偏应用 + 实证，故事+复现性+负面结果都吃 |
| **Journal of Scheduling** | ★★★★ | 偏调度算法，接受详细 benchmark |
| **ICAPS** | ★★★★ | 偏规划/调度，正好对口 |
| **NeurIPS REsAM workshop** | ★★★ | 复现性 + bug 报告对口 |
| **AAAI / IJCAI** | ★★ | 偏 SOTA，本文的"简单胜出"故事偏负面，不太对口 |

## 后续补完工作

按性价比排序：

1. **跑 mk07/08/09 的 5-seed REINFORCE**——目前还是单 seed，把论文
   表格填满。
2. **加 A2C value baseline 实验**——把"PA-REINFORCE + A2C"作为消融，
   看能否缩小 mk10/mk12 方差。
3. **做 cross-instance BC + REINFORCE**——这是论文 §8.3 的未来工作
   之一，做了就是一篇 follow-up 论文。
4. **画训练曲线图**——目前只有表格，需要把 best_greedy_makespan vs
   episode 曲线画成图。
5. **写 LaTeX 版本**——Markdown 草稿写成后转 LaTeX 排版，便于投稿。
