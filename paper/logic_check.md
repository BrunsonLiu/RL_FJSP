# 论文逻辑自检报告

> 检查对象:`paper/main.tex` (15 页, 50+ 引用)
> 检查时间:2026-06-11
> 评价框架:用户提出的 11 项问题清单
> 总评:**结构完整,但"实证细节"严重不足,缺图严重**

---

## 1. 场景 / 意义 / 背景

| 状态 | 项 |
|---|---|
| ✅ 概念清晰 | FJSP 是 NP-hard,OR-Tools 跑不出大实例最优 |
| ❌ **缺真实工业场景** | 没说 FJSP 在哪里出现,市场规模多大,经济影响 |
| ❌ 缺 "motivating example" | 没有"假设一个半导体工厂..." |
| ⚠️ 背景抽象 | 开头太学术化,缺"industry pull" |

**顶刊做法:** Nature 系列开头给一个具体场景 ——
e.g. "在汽车制造车间,30 台机器、100 个 job 的排产延迟 1 小时损失 ¥10K"

---

## 2. 前人研究铺垫

| 状态 | 项 |
|---|---|
| ✅ 50+ 引用 | L2D/L2I/RESCHED/SMG-DRL/DAN 谱系完整 |
| ✅ §2.3 显式空白 | "no published RL method has reported MK13 NEW SOTA" |
| ✅ §2.5 复现问题 | HGT dropout, CP-SAT 误读 |
| ❌ **缺定量失败证据** | 没引用具体数字证明前人差 |
| ❌ 缺 "research gap" 段落 | Nature 风有 "Open Questions" 子节 |

---

## 3. 我们做了什么 / 为什么 / 贡献

| 状态 | 项 |
|---|---|
| ✅ 4 个 contribution | 在 §1 列出 |
| ✅ 假设驱动 | 抽象中"添加了反数据点" |
| ❌ **Why-we-did-this 弱** | 读起来像"我们试了"而非"我们假设 X,验证 X" |
| ❌ 贡献与实验对应不清 | "Reproducible testbed" 在 Results 没验证 |

---

## 4. 算法设计

| 状态 | 项 |
|---|---|
| ✅ MDP 形式化 | §3 |
| ✅ 8 维特征 | §3.2 |
| ✅ 2 个算法伪代码 | Algorithm 1, 2 |
| ✅ 4 邻域 N1-N4 | §4.2 |
| ❌ **没有流程图** | "流程图都要画" — 没有 TikZ/PDF |
| ❌ **Why-this-design 缺** | 为什么 8 特征够?为什么 EMA 基线? |
| ❌ 算法优势没列 | 没和 Transformer/PPO 逐条对比 |
| ❌ 创新点不显式 | 没 "conceptual innovation" 列表 |

---

## 5. 实验设置

| 状态 | 项 |
|---|---|
| ✅ 5 seed | 顶刊基线 |
| ✅ Brandimarte MK01-15 | 标准 |
| ✅ OR-Tools 300s | 显式 OPT/FEASIBLE |
| ✅ 复现命令 | App B |
| ❌ **缺"实验协议"小节** | Nature 有"Statistical Analysis"子节 |
| ❌ 缺 compute 资源 | 没 CPU/GPU 型号、内存 |
| ❌ 缺 significance level | 没说 α=0.05 |
| ⚠️ 5 seed 是下限 | 部分顶刊用 10-20 |

---

## 6. 对比算法

| 状态 | 项 |
|---|---|
| ✅ 4 RL + EF + OR-Tools | 6 个 baseline |
| ❌ **没跑 L2I / RESCHED** | 致命 — 引了不跑 |
| ❌ 缺 Hurink benchmark | 顶刊通常 ≥2 套 |
| ❌ 缺 SD1/SD2 生成集 | RESCHED 主战场 |
| ⚠️ 4 agents 选取理由弱 | 没说为什么是这 4 个 |

---

## 7. 可视化

| 状态 | 项 |
|---|---|
| ✅ 2 个 booktabs 表 | 表格风格合规 |
| ✅ 2 个 algorithm | 伪代码 |
| ❌ **0 张图** | Nature 风 4-6 张图 |
| ❌ 缺 pipeline 流程图 | 核心叙事图 |
| ❌ 缺 learning curve | 5 seed 收敛 |
| ❌ 缺 bar chart | SOTA 对比可视化 |
| ❌ 缺 ablation 可视化 | 各组件贡献饼图/条形图 |
| ⚠️ 颜色无意义 | 没用到颜色编码信息 |

---

## 8. 结果

| 状态 | 项 |
|---|---|
| ✅ TIED 4 + NEW SOTA MK13 | 数字层面强 |
| ✅ 5.64% 平均 gap | 强 |
| ❌ **"能证明什么"不显式** | 每个表后缺 interpretation |
| ❌ **缺 mean ± std** | 5 seed 没汇报离散度 |
| ❌ **缺 p-value** | 没 t-test/Wilcoxon |
| ❌ 缺 wall-clock 对比 | OR-Tools vs RL 时间 |
| ❌ 缺逐级 ablation | RL / +ILS / +SA / +TS |
| ❌ 缺 best/worst/mean | 没汇报分布 |

---

## 9. 其他发现

| 状态 | 项 |
|---|---|
| ✅ HGT dropout bug | 真实复现问题 |
| ✅ CP-SAT 误读 | 方法学 |
| ✅ BC+PPO 负结果 | 重要 negative result |
| ❌ **缺"管理洞见"** | 没讲工厂经理能从中学到什么 |
| ❌ 缺 transferability | 训完能否换规模 |
| ❌ 缺 scalability | 训练时间 vs 实例规模 |
| ⚠️ "PA-REINFORCE > Transformer" 震撼但没强调 | 应该做 key finding |

---

## 10. 结论

| 状态 | 项 |
|---|---|
| ✅ 1 段总结 | 简短 |
| ❌ **§10 Limitations 缺** | 应该独立成节 |
| ❌ **§11 Future Work 缺** | 3-5 个具体方向 |
| ❌ **§12 Broader Impact 缺** | 顶刊越来越重视 |
| ⚠️ Conclusion 1 段太短 | 3-4 段 |

---

## 11. 整体逻辑链

**当前串联:**
- §1 引 → §2 文献 → §3 形式化 → §4 算法 → §5 实验 →
  §6 结果 → §7 教训 → §8 讨论 → §9 结论 → References

**问题:**
- §1 列 4 贡献,§6/§7/§8 没显式"兑现"
- §7 教训与 §6 结果衔接不紧
- §8 讨论未 link 回 §2.7 定位

---

## 优先级清单

### P0 — 致命(必做,投前)
1. **加 pipeline 流程图**(TikZ,RL → ILS → SA → TS)
2. **加 SOTA bar chart** 可视化(Matplotlib 输出 PDF)
3. **加 learning curve** 可视化(5 seed 收敛)
4. **跑 / 引用 L2I + RESCHED on Brandimarte**(对比表)
5. **5 实例 schedule 存盘 + validate** (mk03/08/12/13/14)
6. **所有数字补 mean ± std + Wilcoxon p-value**

### P1 — 重要(影响评分)
7. **加真实工业场景段** in §1
8. **加 §10 Limitations** 独立节
9. **加 §11 Future Work** 3-5 方向
10. **加逐级 ablation 表** (RL / +ILS / +SA / +TS)
11. **加 wall-clock 对比表**
12. **加 compute 资源** in §5
13. **加 transferability 实验** (小规模泛化)

### P2 — 加分
14. **Hurink benchmark** 完整跑
15. **SD1/SD2 生成集** 验证
16. **管理洞见段落** in §8
17. **TikZ 架构对比图** (PA-REINFORCE vs Graph vs Transformer)
18. **Broader Impact** 子节

---

## 总评

| 维度 | 评分 (1-10) |
|---|---|
| 逻辑完整性 | 7 |
| 实验完整度 | 5 |
| 可视化 | **3** |
| 复现性 | 7 |
| 顶刊规范 | 6 |
| 创新性 | 7 |
| **综合** | **6** (B- 偏 C+) |

**主线逻辑 OK,关键缺图、P0 实验、P1 讨论节。**
