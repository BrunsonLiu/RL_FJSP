# 3-Paper Digest: LPNR 必须先读的关键文献

> 目标：写 LPNR 之前，必须先吃透这 3 篇。它们分别定义了我们必须使用的"事实标准"、必须对比的"最直接竞争者"、必须借鉴的"机制范式"。

---

## Paper 1: Vieira, Herrmann, Lin (2003) — "Rescheduling Manufacturing Systems: A Framework of Strategies, Policies, and Methods"
**Journal of Scheduling, 6(1): 35-58**
**威胁等级: ★☆☆ (基础工具)**
**作用: stability 度量的事实标准**

### 论文核心 (1 分钟读懂)

这是反应式调度的"圣经级"论文（Google Scholar 引用 1500+），把过去 20 年的 rescheduling 文献整理成 5 个维度：
- **Goal**: makespan / stability / robustness / 混合
- **Strategy**: Predictive-Reactive / Dynamic (Dispatching) / Robust
- **Policy**: Right-Shift / Affected-Operations / Partial / Matchup
- **Method**: Local Search / Heuristics / Metaheuristics / Exact
- **Rescheduling frequency**: Event-driven / Periodic / Hybrid

### 必须吃的 4 个定义 (论文 §3.1-3.3)

| 概念 | 公式 | 含义 |
|---|---|---|
| **Schedule stability** | Δ = 1/C_old × ΣⱼΣᵢ \|S_new(j,i) − S_old(j,i)\| | 归一化的开始时间总偏差 |
| **Makespan deviation** | ΔC = (C_new − C_old) / C_old | 重调度后完工时间变化率 |
| **Operation deviation** | D_op = #(changed ops) / #(total ops) | 改动工序比例 |
| **Machine-assignment change** | D_ma = #(re-assigned ops) / #(total ops) | 机器分配改动比例 |

### 为什么 LPNR 必须引用

- **任何"稳定性"评估都必须用这 4 个公式**，否则审稿人会说"为什么不和经典 reactive scheduling 框架对比"
- Vieira 2003 给出了 RSR/AOR/Matchup 的明确分类，我们要论证"为什么 RL-based repair 比 AOR/RSR 更好"
- 论文 35-40 页详细讨论了 "stability vs. makespan trade-off"，是我们 motivation 的直接支撑

### 读这一篇你要回答的问题

1. 我设计的 schedule deviation 公式是上面 4 种中的哪一种组合？为什么是这几种？
2. 我的 RSR/AOR baseline 如何用论文中的定义实现？可不可能进一步和 Matchup Scheduling 对比？
3. 论文讨论的 rescheduling frequency（event-driven vs. periodic）怎么对应到我的"扰动到达时刻"？

---

## Paper 2: Lv, Fan, Zhang, Shen (2025) — "A multi-agent reinforcement learning based scheduling strategy for flexible job shops under machine breakdowns"
**Robotics and Computer-Integrated Manufacturing, 93:102923**
**威胁等级: ★★★ (最直接竞争者)**
**作用: 必须显式区分的 SOTA**

### 论文核心 (1 分钟读懂)

华中科技大学 + NRC Canada 团队，把 dynamic FJSP under machine breakdown 形式化为 **multi-agent MDP**，提出 **type-aware MADRL**：
- **State**: 异构图 (machine 节点 + operation 节点 + 边表示工艺约束)
- **Encoder**: meta-path type-aware (MPTA) RNN + 异构图注意力 (HAN) + 超网络 (hypernetwork) 做参数自适应
- **Action**: 每个 machine agent 从候选 operation 集合中选 1 个（cross-attention）
- **Reward**: makespan + stability（首次显式引入 stability objective）

**Highlight 3（论文自己强调的）**: "The movement of unaffected operations has expanded the action space" — 这正是我们要反对的：**他们让"未受影响工序"也进入动作空间，导致 action space 爆炸**。

### LPNR 怎么和它区分

| 维度 | Lv 2025 | LPNR (我们) |
|---|---|---|
| **Action 空间** | O(m × n × T) 巨大（未受影响工序也可移动）| O(\|region\| × \|operator\|) 小（先选 region）|
| **Region 选择** | 隐式（没有显式 region head）| **显式 RL head** 输出 region |
| **约束处理** | action mask + soft penalty | **hard constraint decoder**（必合法）|
| **扰动类型** | 仅 machine breakdown | breakdown + new job + processing delay |
| **Stability 显式建模** | 是 | 是 |
| **可解释性** | 黑箱 | region + operator + budget 三层可解释 |

### 论文自己承认的局限 (§5)

- 仅处理 machine breakdown，未涉及 new job arrival
- 异构图注意力 + MPTA 训练时间 > 24h/instance
- Action space 包含"未受影响工序"组合爆炸
- 缺少与经典 reactive scheduling baseline（RSR/AOR）的对比

### 读这一篇你要回答的问题

1. 我的"显式 region head"相对 Lv 2025 的"隐式 full-graph action"在**样本效率**上有什么优势？
2. Lv 2025 没有 baseline 对比 RSR/AOR，我要不要把这个补上？这是 easy gain 还是 reviewer 必问？
3. 我能不能"反击" Lv 2025 的 MADRL？比如证明 LPNR 在**小训练数据**下也能超过它（看 EL-DRL NeurIPS 2023 的思路）？

---

## Paper 3: Reijnen, Zhang, Lau, Bukhsh (2024) — "Online Control of Adaptive Large Neighborhood Search Using Deep Reinforcement Learning"
**ICAPS 2024, pp. 475-483**
**威胁等级: ★★☆ (机制范式)**
**作用: 必须借鉴的 RL+ALNS 融合模式**

### 论文核心 (1 分钟读懂)

TU/e + SMU 团队，提出 **DR-ALNS**：用 DRL 实时控制 ALNS 的 3 个超参：
- **Operator selection**: 选哪个 destroy / repair operator
- **Operator parameter**: destroy ratio、repair ratio
- **Acceptance criterion**: SA 温度、阈值

**关键 insight**: DRL 看到的 state 不是 problem instance，而是**当前 ALNS 的搜索状态**（operator 历史得分、当前解 diversity、温度等），**与问题解耦**。

### LPNR 怎么借鉴它

| 要素 | DR-ALNS | LPNR (我们) |
|---|---|---|
| **State 抽象** | 与 problem 解耦 → 跨问题迁移 | **应该借鉴**：state 包含"当前解的 makespan、stability、扰动后剩余约束"等，不直接是 raw problem features |
| **Action** | 选 operator + 调参数 | **升级**：选 region + operator + budget |
| **Reward** | improvement（makespan 下降）| improvement + stability |
| **Online control** | 边搜边学 | **直接用** |

### 为什么这篇"威胁"是中等

- DR-ALNS 是 **problem-agnostic**（在 orienteering 上验证），**未在 FJSP 上验证**
- LPNR 比 DR-ALNS 多两个**新维度**：
  1. **Region selection**（DR-ALNS 没有）
  2. **Problem-specific FJSP decoder**（DR-ALNS 也没有）

但如果**有人**已经做过"DR-ALNS on FJSP"，那我们的 novelty 就**只剩 region selection**。

### 读这一篇你要回答的问题

1. LPNR 的 state 怎么设计才能既包含 FJSP-specific 信息（region 候选、扰动类型）又保持 problem-agnostic 的可迁移性？
2. 我们的 RL training 是在 "静态" instance 上训练，还是 "扰动 → repair" 的 episode 上训练？后者更接近 DR-ALNS 的范式
3. DR-ALNS 用了 self-play / curriculum 吗？我们用什么训练范式？

---

## 综合消化结论

读完这 3 篇，你应该能写出一个**完整的对比表**：

```
              Problem-Agnostic  Problem-Specific
            +-----------------+------------------+
No Region   | DR-ALNS (Reijnen)| (一般元启发式)   |
            |  ICAPS 2024     |                  |
            +-----------------+------------------+
Region      | (无 — 这是真空) | LPNR (我们的位置) |
            +-----------------+------------------+
```

**这个 2×2 矩阵是论文 Related Work 的核心框架。**

**LPNR 的 uniqueness**:
- 比 DR-ALNS 多: region selection + problem-specific decoder
- 比 Lv 2025 少: action space 爆炸；多: legality guarantee + multi-perturbation

**可投稿目标**:
- **TEVC / SWAM / J. Manufacturing Systems / RCIM**: 这个 novelty 足够（实证导向，35 天决策周期）
- **NeurIPS / ICLR**: 需要更强 theoretical contribution（regret bound / 收敛性）

---

## 接下来你读这 3 篇时，建议的阅读顺序

1. **先读 Vieira 2003**（30 分钟，搞清 4 个 stability 公式，会写 baseline）
2. **再读 Lv 2025**（60 分钟，搞清现状 SOTA 的完整设计，找可以攻击的弱点）
3. **最后读 Reijnen 2024 DR-ALNS**（45 分钟，搞清 RL+ALNS 融合范式，决定怎么 borrow）

读完后，我们再开 1 个 session 讨论：你的 reading notes + 你的 design proposal。

