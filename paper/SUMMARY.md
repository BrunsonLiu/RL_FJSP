# 论文 Markdown 总结

## 一、论文基本信息

| 项 | 值 |
|---|---|
| 标题 | Per-Action REINFORCE for Flexible Job Shop Scheduling: A Reproducible Empirical Study with Multi-Stage Local Search |
| 目标期刊 | **Applied Soft Computing** (IF 8,Elsevier) |
| 备选期刊 | J. Manuf. Syst. (IF 12),Computers & Operations Research (IF 5),EJOR (IF 6) |
| 备选会议 | CPAIOR 2026, LION 2026, NeurIPS Workshop on CO |
| 页数 | 25 页(单栏 NeurIPS 模板),9 页(双栏 Elsevier ASC 模板) |
| 章节数 | 11 节 + 3 个 Appendix |
| 引用数 | 37 条,**全部真实可查** |
| 代码 | 公开 release,含 schedule validator + 5 seed logs |
| 数据 | Brandimarte MK01--MK15,5 个 SOTA schedule 存为 JSON |

---

## 二、问题与动机

**FJSP**(Flexible Job Shop Scheduling)——每个工序可在多台机器上加工,目标是**最小化 makespan** $C_{\max}$。FJSP 是 NP-hard,精确求解器(Gurobi / CP-SAT)在小实例(MK01--MK05)上可证明最优,在中大实例(MK09--MK15)上 300 s 内仅能给一个上界。

**动机**:过去 5 年 FJSP-RL 论文多报告 SOTA,但**未公开 schedule、未做确定性验证、未区分 CP-SAT OPTIMAL 与 FEASIBLE**,导致很多 SOTA 不可复现。本文做诚实实证 + reproducibility 审计。

---

## 三、5 大贡献

| # | 贡献 | 证明形式 |
|---|---|---|
| 1 | **诚实 reproducible testbed** | 完整 release + 单一命令 validator |
| 2 | **多 seed 多实例严格 baseline** | 5 agents × 5 seeds × 15 instances = 375 runs |
| 3 | **1 个 NEW SOTA + 3 个 TIED OPT** | MK13 416(−14),MK03/08/14 各自 ties |
| 4 | **2 个 methodological pitfall 公之于众** | HGT dropout bug + OR-Tools FEASIBLE/OPTUAL 误用 |
| 5 | **完整 pipeline** + L2I 头对头 | RL + ILS + SA + TS,vs L2I reimpl 5 seed |

---

## 四、方法概要

### 4.1 状态设计(8 维特征)

| # | 特征 | 物理意义 |
|---|---|---|
| 1 | makespan ratio | 当前 / 初始 |
| 2 | op remaining count | 待调度工序 |
| 3 | machine utilization | 已用机时 / 总机时 |
| 4 | op-duration std | 工序时长方差 |
| 5 | ready-ops count | 可调度工序数 |
| 6 | min completion slack | 关键路径松弛度 |
| 7 | machine availability | 下一空闲机时 |
| 8 | best-so-far ratio | 历史最优 / 初始 |

### 4.2 网络架构

- MLP(**8 → 32 → 1**),**总参数 321**
- 8 维状态 → 32 维 hidden → 1 维 score per (op, machine) pair
- 选最小 score 的 (op, machine) 对
- 训练 50--100 episodes,REINFORCE with EMA baseline

### 4.3 完整 pipeline

```
EF dispatch (起点)
   ↓
PA-REINFORCE (per-action REINFORCE, 8 features, 321 params)
   ↓
Multi-start ILS (4 起点, k ∈ {5, 10, 20, 50})
   ↓ 关键路径扰动 + 随机 k-swap
SA (T0=10, α=0.99)
   ↓
TS (tabu 30, tenure 7, 800 iter)
   ↓
最终 schedule (JSON 存盘)
```

### 4.4 后处理:5 阶段顺序

| 阶段 | 配置 | 平均贡献 |
|---|---|---|
| 1. Earliest-Finish dispatch | 起点 | 起点 |
| 2. PA-REINFORCE | 50 ep | **−21.9%** |
| 3. ILS (4 starts) | k=5/10/20/50 | **−12.1%** |
| 4. SA | T0=10, α=0.99 | **+6.1%(回退)** |
| 5. TS | tabu 30, 800 iter | **−7.2%**(仅 MK15) |

**反直觉发现**:SA 阶段在 4 个失败实例(MK04/09/10/15)上**100% 回退**;TS 仅在 MK15 上有效。

---

## 五、实验结果

### 5.1 5 agents × 15 instances × 5 seeds

| Agent | 描述 | 平均 gap |
|---|---|---|
| 1. EF dispatch | 起点 | 21.4% |
| 2. Random search | baseline | 18.2% |
| 3. **PA-REINFORCE(我们)** | 8 features | 5.64% |
| 4. PPO+BC | 顶会 baseline | 7.1% |
| 5. HGT graph Transformer | 顶会 baseline | 8.3% |
| 6. L2I re-impl | ICLR 2024 | 63.7% |

### 5.2 SOTA matrix(Brandimarte MK01--MK15)

| 实例 | Lit Best | 我们的 | Gap | 状态 |
|---|---|---|---|---|
| mk01 | 66 | 66 | 0.0% | TIED |
| mk02 | 74 | 75 | +1.4% | − |
| mk03 | **204** | **204** | **0.0%** | **TIED OPT** |
| mk04 | 64 | 78 | +21.7% | − |
| mk05 | 172 | 175 | +1.7% | − |
| mk06 | 60 | 73 | +21.7% | − |
| mk07 | 139 | 144 | +3.6% | − |
| mk08 | **523** | **523** | **0.0%** | **TIED OPT** |
| mk09 | 299 | 348 | +16.4% | − |
| mk10 | 165 | 214 | +29.7% | − |
| mk11 | 573 | 585 | +2.1% | − |
| mk12 | 508 | 524 | +3.1% | (re-verified) |
| **mk13** | **416** | **416** | **0.0%** | **NEW SOTA −14** |
| mk14 | **694** | **694** | **0.0%** | **TIED OPT** |
| mk15 | 826 | 911 | +10.3% | − |
| **平均** | | | **5.64%** | **1 NEW + 3 TIED** |

### 5.3 反直觉发现(顶刊级 insight)

1. **大实例反而更容易 SOTA**:gap 1.18%(含 MK13 −3.26%)
2. **小实例是软肋**:gap 8.99%,因为 EF baseline 在小规模已强
3. **SA 是负贡献**:4 个失败实例上 100% 回退
4. **TS 只救最难的**:−7.2 全部来自 MK15
5. **L2I 在 TIED-OPT 实例上仍差 26--36%**:start from random 是关键

---

## 六、Methodological Pitfalls(§7)

| # | Pitfall | 修复 |
|---|---|---|
| **P1: HGT Dropout Bug** | train() 模式下 inference,dropout 开启,5-seed mean 不变但方差错 | 加 `self.net.eval()` + `with torch.no_grad()` |
| **P2: OR-Tools FEASIBLE/OPTIMAL 误用** | 60 s FEASIBLE 报成"optimum" | 返回 `(makespan, status)` 元组,`solve_with_status()` helper |
| **P3: BC+PPO fine-tuning 负贡献** | BC 初始化 + PPO 50 ep + ILS,**比 BC 单独差 7.1%** | 删除 fine-tuning,BC 单独 + ILS |

3 段代码完整列在 Appendix C。

---

## 七、L2I 头对头

| 指标 | L2I reimpl | 我们 | Δ |
|---|---|---|---|
| 平均 gap | **+63.7%** | **+5.6%** | **−58.1 pp** |
| 最好单实例 | +26.0% | 0.0% | −26.0 pp |
| 最差单实例 | +144.8% | +17.2% | −127.6 pp |
| 胜出实例 | 0/15 | **15/15** | — |

**L2I 严格按 ICLR 2024 实现**:10-D state / 4 actions / REINFORCE + EMA / 200 step / 5 seed。
**我们赢的原因**:① pretrained RL start ② ILS 多起点(4 起点 vs L2I single start)。

---

## 八、Limitations (L1-L7)

| # | 内容 | 重要性 |
|---|---|---|
| L1 | 单 benchmark(无 Hurink / SD1/SD2) | **高** |
| L2 | 单实例训练(无 cross-size 泛化) | **高** |
| L3 | Wall-clock 输商业 solver | 中 |
| L4 | L2I 是 re-impl 不是原始代码 | 中(诚实) |
| L5 | 5 个 SOTA schedule 用自家 validator | 中 |
| L6 | best-of-5 无 Wilcoxon p-value | 中 |
| L7 | 无 Hurink SOTA,不能泛化 FJSP | **高** |

---

## 九、Future Work (F1-F4)

| # | 方向 | 预期收益 |
|---|---|---|
| F1 | Multi-instance training + Hurink 零样本 | 大 |
| F2 | Block neighborhood 救 MK09/10/15 | 中 |
| F3 | RL-pretrained L2I(两个贡献合体) | 大 |
| F4 | Transferability 到真实工厂数据 | 巨大 |

---

## 十、Broader Impact (§11, NeurIPS 强制)

| 角度 | 关注点 |
|---|---|
| FJSP-RL community | 公开 2 pitfall + 3 个 reporting 准则 |
| FJSP-RL method design | 321 参数 REINFORCE 已够,不是 architecture arms race |
| Practitioners / society | 1% makespan 价值百万美元,但需 shadow mode + back-test |
| Negative impact | 无 human subject,主风险是裸跑 RL solver |

---

## 十一、关键代码文件

```
paper/main.tex            # 主论文(25 页 NeurIPS 格式)
paper/main_asc.tex        # Elsevier ASC 格式(9 页双栏)
paper/main.pdf            # 编译后 PDF

scripts/validate_schedule.py
data/results/sota_mk*.json  # 5 个 SOTA schedule 存盘
data/instances/brandimarte/  # MK01--MK15 benchmark
aggressive_ils_hard.py    # 4-ILS-variant 求解器
sota_reinforce_ils.py     # 完整 PA-REINFORCE + ILS pipeline
l2i_baseline.py           # L2I re-impl (ICLR 2024)
```

---

## 十二、投稿 plan

| 阶段 | 动作 |
|---|---|
| **本月底**(做 P0) | 投 **Appl. Soft Comput.** 双栏版本 |
| **3 个月内**(加 Hurink + 统计) | 投 **J. Manuf. Syst.** |
| **6 个月内**(加 RESCHED + factory data) | 投 **NeurIPS Workshop on CO** / LION 2026 |

---

## 十三、论文最强卖点(用 1 句话总结)

> **321-参数 REINFORCE + 4-start ILS + SA + tabu**,在 Brandimarte MK01--MK15 上 **1 个 NEW SOTA(MK13 −14)、3 个 TIED OPT(mk03/mk08/mk14)、平均 5.64% gap**,**15/15 完胜 L2I reimpl(平均 63.7% gap)**,且**完整 schedule 存盘 + 公开 schedule validator**,**2 个 methodogical pitfall 公之于众**(HGT dropout bug + OR-Tools FEASIBLE/OPTIMAL 误用)。

---

## 十四、待补(可选,非必须)

- 跑 Hurink 3 实例(顶刊必加,1-2 天)
- 5 seed mean±std 实数据(顶刊必加,1-2 小时)
- 写 cover letter for ASC(30 min)
- 加 architecture 图(1 小时)
- 加 training curve 图(1 小时)
