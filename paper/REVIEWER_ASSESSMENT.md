# 期刊审稿人视角：完整审稿评估报告

**评估日期**: 2026-06-13
**目标期刊**: Applied Soft Computing (Elsevier, IF 8.0) / J. Manuf. Syst. (IF 12)
**审稿框架**: Nature reviewer 三审稿人制 + ARS 五审稿人制 + 代码级验证
**审稿立场**: 残酷诚实，不美化，不回避

---

## 评估设置

| 项目 | 内容 |
|------|------|
| **输入范围** | 完整论文 (25页 NeurIPS 单栏 + 9页 Elsevier 双栏) + 完整代码库 (~50+ .py 文件) + 内部自评 (HONEST_ASSESSMENT.md) + 所有实验结果 JSON |
| **评估边界** | 可评估：方法、实验设计、代码质量、结果有效性、声称贡献的实质。不可评估：第三方独立复现、真实工业部署效果 |
| **论文声称核心** | 1 NEW SOTA (MK13 416, −14) + 3 TIED OPT + 5.64% mean gap + 3 methodological pitfalls + 15/15 胜 L2I + 321-param beats complex architectures |
| **自评定论** | 项目自己的 HONEST_ASSESSMENT.md 已经承认：**无算法创新、无能力创新、无问题创新**。本质是 "reproducibility study"。 |

---

# PART A: Nature-Style 三审稿人报告

## Reviewer 1 — 技术严谨性重点

### 总体评估

这篇论文在工程执行层面做得**相当扎实**——schedule validator、多 seed 实验、JSON 存盘、一站命令复现，这些在 FJSP-RL 领域都是罕见的良好实践。但当我从"方法贡献"的角度审视时，**论文的实质性创新极为有限**。PA-REINFORCE = Williams 1992 的标准 REINFORCE with EMA baseline，321-param MLP 是最普通的前馈网络，8 维 handcrafted features 没有学习成分。后处理 pipeline（ILS + SA + TS）是教科书级的标准组合。严格来说，这篇论文**没有提出任何新算法、新架构或新训练方法**。

### 谁会关心这些结果，为什么

- **FJSP-RL 社区**：会关心 3 个 methodological pitfalls（尤其是 HGT dropout bug 和 CP-SAT FEASIBLE/OPTIMAL 误用），因为这直接质疑已发表文献的可信度。
- **Reproducibility 倡导者**：会欢迎完整的 schedule release + validator，这是该领域稀缺的实践。
- **工业从业者**：可能关心简单 MLP + ILS 是否足以匹敌复杂神经网络，从而降低部署门槛。
- **期刊编辑/审稿人**：大概率不认为这是一篇 A 档期刊的实质性贡献。

### 主要优点

1. **Reproducibility infrastructure 是真实的贡献**：Schedule validator（`fjsp/scheduler/validator.py`）检查 precedence、eligibility、no-overlap、makespan —— 这四个维度是 FJSP 合法性的充要条件。这是该领域第一次有人公开这样做。
2. **375 runs 的实验量**（5 agents × 5 seeds × 15 instances）提供了足够的统计基础，尽管作者只报告了 best-of-5 而非 mean±std。
3. **L2I re-implementation** 是一个诚实的 effort。复现别人的方法并做头对头比较是值得鼓励的。
4. **CP-SAT status wrapper**（`solve_with_status()`）修复了一个真实的文献级错误——将 FEASIBLE 报成 OPTIMAL 会颠倒结论。
5. **Code quality** 整体可读：`FJSPDispatchEnv` 的设计干净，`validate_schedule()` 的实现正确且完整。

### 主要担忧

1. **报告 best-of-5 而非 mean±std**：这是顶刊评审的硬伤。best-of-5 天然偏向乐观估计，无法区分"方法真的更好"和"运气更好"。没有 Wilcoxon signed-rank test 或任何假设检验，5.64% 的 mean gap 是否统计显著无法判断。
2. **L2I 比较是结构性不公平的**：作者的 pipeline 是 5-stage（REINFORCE + ILS + SA + TS，4 starting points），L2I 是 1-stage（单一随机起点 + 200 steps RL policy）。15/15 全胜的 headline 本质是 "5-stage pipeline > 1-stage method"，而不是 "PA-REINFORCE > L2I"。审稿人会立刻指出这个问题。
3. **没有 cross-instance 泛化实验**：每个 instance 单独训练一个 agent。这是 RL-for-CO 领域最基本的泛化问题——训练一个模型，泛化到未见过的实例。没有这个实验，"321-param beats 32K" 的声称只能局限在 single-instance 范围内，外部有效性为零。
4. **MK04 和 MK06 的 gap 高达 21.7%，MK10 高达 29.7%**：这些失败实例没有被充分讨论。如果 mean gap 5.64% 是因为 3 个 TIED OPT 拉低了平均值，而 hard instances 全部失败，那么论文的实际贡献更小。
5. **Wall-clock 比较缺失**：声称 321-param REINFORCE "beats" 复杂架构，但没有 wall-clock 对比。32K-param HGT 训练 50 episodes 可能只需 2 分钟，而 5-stage pipeline 可能需要 20 分钟。纯参数计数不是性能比较。
6. **只有 Brandimarte benchmark**：没有 Hurink、SD1/SD2、DPData 等标准 FJSP benchmark。Brandimarte 已经用了 30 年，单一 benchmark 的结论无法泛化。

### 需要解决的技术缺陷

| # | 缺陷 | 严重度 | 修复建议 |
|---|------|--------|----------|
| T1 | best-of-5 无 mean±std, 无统计检验 | **致命** | 报告 5-seed mean ± std，加 Wilcoxon signed-rank test 或 Friedman test |
| T2 | L2I 比较不公平 | **致命** | 要么给 L2I 同样的 ILS+SA+TS 后处理，要么只比较纯 RL 阶段，清晰标注比较范围 |
| T3 | 无 cross-instance 泛化 | **高** | 在 MK01-05 上联合训练，零样本测试 MK06-15（即使结果差也是诚实的数据点） |
| T4 | MK04/06/10 失败未解释 | **高** | 逐个分析失败原因（instance 结构特征？action space 大小？EF baseline 强度？），加入 Discussion |
| T5 | 无 wall-clock 对比 | **中** | 加一张表：each agent 的训练时间 + 推理时间，与最终 makespan 一起呈现 |
| T6 | 单 benchmark | **中** | 至少加 Hurink 3 个代表性实例，或者诚实声明此限制 |

### 对 Nature 标准的评估

| 维度 | 评分 | 说明 |
|------|------|------|
| Originality | **弱 (2/5)** | 无新算法、新架构、新训练方法。贡献在实证层面而非方法层面 |
| Scientific importance | **中弱 (2/5)** | 3 pitfalls 对领域有帮助，但属于 engineering hygiene 而非科学突破 |
| Interdisciplinary readership | **弱 (1/5)** | 仅 FJSP-RL 子社区关心，Outside 该领域几乎无影响 |
| Technical soundness | **中 (3/5)** | 工程执行规范，但实验设计有结构性问题 (best-of-5, unfair comparison) |
| Readability | **强 (4/5)** | 论文结构清晰，写作流畅。26 页 NeurIPS 版适合会议，9 页 ASC 版足够紧凑 |

### 建议姿态

**Major Revision → 降级投稿定位**。如果作者坚持投 ASC/A 档，需要补 T1-T6。如果接受 reproducibility study 定位，投 Workshop（NeurIPS CO Workshop, ReScience）则当前版本已经足够且可能直接接受。

---

## Reviewer 2 — 原创性与科学重要性重点

### 总体评估

这篇论文最诚实的描述是：**"A reproducible empirical benchmark study of RL for FJSP with open schedules and a schedule validator."** 它试图用 "5 contributions" 的框架包装，但当剥开框架看实质：

- **Contribution 1 (NEW SOTA MK13)**：一个 instance 的 3.3% 改进。小。
- **Contribution 2 (L2I head-to-head)**：不公平比较（5-stage vs 1-stage）。即使赢 15/15，证明的不是方法好而是 pipeline 长。
- **Contribution 3 (3 pitfalls)**：HGT dropout → 任何写 PyTorch 的人都知道 train/eval 模式切换；CP-SAT FEASIBLE ≠ OPTIMAL → 任何学过 OR 的人都知道；SA 回退 → 这是配置选择，不是发现。三个都是 basic engineering hygiene。
- **Contribution 4 (321 beats 32K/110K/180K)**：negative result —— 在 single-instance 的设置下，encoder 不是瓶颈。有趣但不构成正面贡献。
- **Contribution 5 (5-stage pipeline + ablation)**：标准 ablation study。SA 回退是唯一的新发现，但已经被归入 Pitfall 3。

### 谁会关心这些结果，为什么

- **Reproducibility 运动** (ReScience, ML Reproducibility Challenge)：这是高质量的目标受众。
- **FJSP-RL 新入门研究者**：可以作为可靠的 baseline testbed 使用。
- **方法论审查者**：3 个 pitfalls 可以作为 checklist 用于审查未来的 FJSP-RL 论文。
- **顶刊审稿人**：大概率不认为任何一项 contribution 达到 A 档标准。

### 主要优点

1. **诚实性**：HONEST_ASSESSMENT.md 的存在本身就是一种罕见的学术诚实。作者清楚自己的贡献边界。
2. **Open testbed 是真实的 infrastructure 贡献**：如果这个 testbed 被社区采用，它的长期影响可能超过任何单篇 SOTA 论文。
3. **L2I re-implementation effort**：复现别人的 ICLR 论文并诚实报告结果（包括自己的优势和不公平之处），值得尊重。
4. **写作品质良好**：论文结构符合 IMRaD，37 条引用全部真实可查（已通过 reference cleanup 验证）。

### 主要担忧

1. **"NEW SOTA" 叙事与实质贡献不匹配**：Abstract 以 "new best-known makespan of 416 on MK13" 开头，但这只是一个 instance 的 −14（3.3%）。在 FJSP 文献中，单个 instance 的改进通常不构成 headline contribution —— 除非方法本身是新的。而这里方法不是新的。

2. **"5 contributions" 框架是贡献通胀（contribution inflation）**：
   - Empirical finding ×2：一个 instance 的 SOTA + 不公平的 head-to-head
   - Methodological clarification ×1：基本工程规范
   - Architectural insight ×1：negative result
   - Algorithmic ×1：标准 ablation
   - 实际上只有 1 个真正的贡献：**一个 reproducible FJSP-RL testbed + 375 runs 的 benchmark 数据**。

3. **缺乏对领域文献的系统性覆盖**：§7 只讨论了 2-3 个 pitfall（HGT dropout, CP-SAT），但没有系统性地审计已发表的 FJSP-RL 论文。如果要声称 "methodological pitfall"，应该列一个完整的文献审计表：哪些论文可能存在这些问题。

4. **Discussion 部分过弱**：没有深入分析为什么 MK04/06/10 失败（这些 instance 有什么结构特征让 pipeline 失效？），也没有分析为什么 321-param MLP 在 single-instance 设置下足够。Discussion 停留在 "SA reverts" 的现象描述，缺乏机制性解释。

5. **"Architectural arms race" 的声称过强**：single-instance 设置下 simple model 足够 ≠ arms race 整体失败。Cross-instance 泛化才是 encoder 真正发挥作用的地方。作者知道这个限制（L1, L2），但仍然在 Abstract 中做了全局性声称。

### 对 Nature 标准的评估

| 维度 | 评分 | 说明 |
|------|------|------|
| Originality | **弱 (1.5/5)** | 核心方法（REINFORCE + MLP + ILS + SA + TS）没有原创性 |
| Scientific importance | **弱 (2/5)** | Reproducibility infrastructure 有价值，但达不到"杰出科学重要性" |
| Interdisciplinary readership | **弱 (1/5)** | 极度领域内部 |
| Technical soundness | **中 (3/5)** | 工程执行好，实验设计有结构性缺陷 |
| Readability | **强 (4/5)** | 写作流畅 |

### 建议姿态

**如果目标期刊是 ASC (IF 8) → Major Revision with high risk of rejection**。
**如果目标期刊是 ReScience / Workshop → 直接接受**。
**如果目标期刊是 JMS (IF 12) → 几乎必定被拒**。

对 ASC 来说，一个诚实且高工程质量的 empirical study 可能在 "application-oriented soft computing" 的定位下发表，但前提是：(a) 补 mean±std + 统计检验，(b) 修正 L2I 比较或降级其声称，(c) 明确定位为 "reproducibility benchmark" 而非 "novel method"。

---

## Reviewer 3 — 跨学科可读性与影响力重点

### 总体评估

这篇论文的写作质量是好的——Abstract 传达了核心结果，Introduction 提供了清晰的动机（三个 concerns），方法部分（§4）足够详细。但当我试图向一个不做 FJSP 的同事解释这篇论文的重要性时，我遇到了困难。

> "我们在一个 30 年的 benchmark 上用 1992 年的算法 + 教科书级后处理赢了一个 instance 的 3.3%，并发现了一些基本工程错误。"

这不是一个跨学科读者会觉得"重要"的故事。

### 谁会关心这些结果，为什么

- **工业工程师/运筹学从业者**：一个简单的 MLP + ILS pipeline 如果能部署到真实工厂，维护成本远低于复杂的 GNN/Transformer。但论文没有展示真实工厂数据。
- **AI reproducibility 社区**：FJSP-RL 领域的 "replication crisis" case study。
- **方法论教师**：3 个 pitfalls 可以作为研究生课程的教学案例。
- **一般 AI/ML 研究者**：不太可能关心。这是一个细分应用领域。

### 主要优点

1. **论文的可读性良好**：即使对 FJSP 不熟悉的读者，Abstract + Introduction 也能理解问题背景和主要发现。
2. **代码开源 + 一键复现**：这是跨学科读者能理解并尊重的实践。
3. **3 个 pitfalls 有教育价值**：HGT dropout bug 和 CP-SAT misuse 是具体的、可操作的发现。

### 主要担忧

1. **"So what?" 问题没有得到满意回答**：即使 321-param MLP 在 Brandimarte 上表现好，这对真实制造业意味着什么？没有工厂数据，没有成本分析，没有部署经验。Paper 在 §11 Broader Impact 中提到了 "1% makespan = 百万美元"，但没有量化自己的方法在真实场景中的收益。

2. **跨学科吸引力几乎为零**：这是 FJSP 领域内部的工作。没有与 climate/sustainability（调度 → 节能）、healthcare（手术室调度）、logistics（车辆路径）等更广泛领域的联系。即使是"RL + combinatorial optimization"这个大领域，这篇论文也没有提出可迁移到其他 CO 问题的 insight。

3. **5-stage pipeline 的"非单调性"发现虽然有趣，但解释不足**：SA 为什么在 4/15 实例上回退？是因为 SA 的参数（T0=10, α=0.99）不适合那些实例？还是因为 instance 结构使得 SA 的随机扰动有害？如果是前者，这是配置问题不是发现；如果是后者，应该深入分析。

4. **缺乏可视化**：论文声称有 Gantt chart 和 architecture diagram，但作为审稿人我没有看到。对于非专业读者来说，一张好的 Gantt chart 比三个表格更有说服力。

5. **Appendix C 的代码列表价值有限**：3 个 pitfalls 的代码片段已经够小，直接放在正文中更好。Appendix C 占了一页但信息密度低。

### 对 Nature 标准的评估

| 维度 | 评分 | 说明 |
|------|------|------|
| Originality | **弱 (1.5/5)** | 无方法原创 |
| Scientific importance | **弱 (1.5/5)** | 领域内部，难以跨学科 |
| Interdisciplinary readership | **弱 (1/5)** | 无跨学科吸引力 |
| Technical soundness | **中 (3/5)** | 工程好，实验设计有缺陷 |
| Readability | **中强 (3.5/5)** | 对领域内读者友好，对领域外有门槛 |

### 建议姿态

**如果作者想发表跨学科影响力更大的论文，需要**：
- 连接 FJSP → broader CO/RL community
- 添加 factory case study 或 at minimum 真实制造数据
- 将 "simple method beats complex method" 包装成一个可迁移到其他 CO 问题的 insight

**当前版本**：仅适合领域内部期刊或 reproducibility venue。

---

## 跨审稿人综合

### 共识优点

1. **Reproducibility infrastructure 是真实且有价值的贡献**：schedule validator + open testbed + JSON release。三个审稿人都同意这是在 FJSP-RL 领域罕见且值得称赞的实践。
2. **Code quality 和工程执行良好**：375 runs 的实验量、clean API design、正确的 validator 实现。
3. **3 个 methodological pitfalls 对社区有用**：虽然不构成"科学突破"，但作为工程规范提醒有价值。
4. **写作流畅，结构清晰**：IMRaD + 11 节 + 3 appendix 的组织合理。

### 共识技术风险

1. **实验报告不完整（best-of-5, 无统计检验）**：这是三个审稿人的共识 red flag，是顶刊硬伤。
2. **L2I 比较不公平**：15/15 胜的 headline 不反映方法对比，反映 pipeline 长度对比。
3. **单一 benchmark (Brandimarte only)**：限制结论的外部有效性。
4. **无 cross-instance 泛化**：限制了 "321-param beats 32K" 声称的适用范围。
5. **MK04/06/10 的高 gap 未充分分析**：削弱了 mean gap 5.64% 的可信度。

### 审稿人强调差异

| 差异点 | R1 (技术) | R2 (原创性) | R3 (影响力) |
|--------|-----------|-------------|-------------|
| 最大担忧 | 统计报告 + 不公平比较 | 无方法创新 + 贡献通胀 | 跨学科影响力为零 |
| 对 pitfalls 的评价 | 工程规范，有用但小 | 不是科学贡献 | 有教育价值 |
| 对 L2I 比较的评价 | 技术不公平 | 结构性不公平 | 不影响核心故事 |
| 对发表可能性的判断 | ASC 需要 Major Revision | ASC 50-60% 概率，需降级定位 | ASC 勉强，但 cross-disciplinary appeal 不足 |

### 最重要的问题（发表前必须解决）

按优先级排序：

1. **[P0 - 致命]** 报告 5-seed mean ± std + Wilcoxon/Friedman 统计检验。用 mean 而非 best-of-5 重算所有表格。
2. **[P0 - 致命]** 修正 L2I 比较 —— 要么给 L2I 相同的 ILS+SA+TS 后处理，要么只比较纯 RL 阶段，并在 Abstract 中清晰标注比较范围。
3. **[P1 - 高]** 补至少 Hurink 3 个代表性实例（edata, rdata, vdata），或诚实声明 single-benchmark 限制并在标题/Abstract 中体现。
4. **[P1 - 高]** 加 wall-clock comparison table（训练时间 + 推理时间 vs makespan）。
5. **[P1 - 高]** 深入分析 MK04/06/10/15 失败原因，提供 instance-level 的诊断。
6. **[P2 - 中]** 重新 frame paper 为 "reproducible empirical benchmark study"，降级 "NEW SOTA" 叙事。
7. **[P2 - 中]** 补 cross-instance 训练实验（即使在 held-out 上结果差，也是诚实的数据点，且构成 F1 的初步工作）。
8. **[P2 - 中]** 加 architecture diagram + Gantt chart + training curve。

---

# PART B: ARS-Style 五审稿人评分

## EIC (Editor-in-Chief) — Applied Soft Computing 视角

| 维度 | 分数 (0-100) | 评语 |
|------|-------------|------|
| Originality (20%) | **32** | No novel algorithm, architecture, or training method. Contribution is empirical, not methodological. |
| Methodological Rigor (25%) | **55** | Engineering execution is clean; experimental design has structural gaps (best-of-5, unfair comparison, no stats). |
| Evidence Sufficiency (25%) | **50** | 375 runs is substantial volume but wrongly reported. Missing mean±std, wall-clock, cross-instance data. |
| Argument Coherence (15%) | **65** | Paper structure is logical; but "5 contributions" framing overclaims. |
| Writing Quality (15%) | **72** | Clear and fluent academic English; minor issues only. |
| **Weighted Total** | **52.3** | |

**决策**: **Major Revision**（勉强）

ASC 的 scope 包括 "applied soft computing methods with practical relevance"。这篇论文的 "practical relevance" 较弱（无真实数据），"soft computing method" 较弱（REINFORCE 不 soft，MLP 太小）。但 reproducibility contribution 是真实的，如果作者补充 P0-P1 项并诚意降级叙事，有一定发表概率。

**关键 EIC 评论**：
> "This is a well-executed empirical study that provides useful infrastructure for the FJSP-RL community. However, the claimed contributions do not constitute methodological innovation. I recommend the authors reframe the paper as a reproducibility benchmark study and address the statistical reporting gaps before resubmission."

---

## Reviewer 1 — Methodology Reviewer (研究设计 + 统计 + 可复现性)

| 维度 | 分数 | 评语 |
|------|------|------|
| Originality | **25** | No method novelty. |
| Methodological Rigor | **45** | Best-of-5 reporting is a critical flaw. No hypothesis testing. L2I comparison is structurally unfair. |
| Evidence Sufficiency | **42** | Single benchmark. No mean±std. No wall-clock. No cross-instance. |
| Argument Coherence | **60** | Logic flow is fine; contribution inflation is the main issue. |
| Writing Quality | **70** | Acceptable academic prose. |

**关键评论**：
> "Reporting best-of-5 across 5 seeds without mean, standard deviation, or any statistical test is unacceptable for a journal submission. The L2I comparison compares a 5-stage pipeline against a 1-stage method — this is not a head-to-head of RL methods. Both issues must be corrected before the paper can be evaluated on its merits."

---

## Reviewer 2 — Domain Reviewer (FJSP/OR 领域专业知识)

| 维度 | 分数 | 评语 |
|------|------|------|
| Originality | **30** | 3 pitfalls are useful but basic; MK13 SOTA is one instance. |
| Methodological Rigor | **55** | Dispatch env design is correct; ILS/SA/TS are standard recipes. |
| Evidence Sufficiency | **50** | Brandimarte is standard but only one benchmark. No comparison with Hexaly commercial solver. |
| Argument Coherence | **60** | "Architectural arms race" criticism is interesting but overclaimed — single-instance setup doesn't test what encoders are for. |
| Writing Quality | **75** | Good FJSP terminology usage. |

**关键评论**：
> "The paper's most interesting finding — that a 321-param MLP matches or beats 32K–180K graph/Transformer models — is confounded by the single-instance training setup. Graph and Transformer encoders are designed for cross-instance generalization. The claim that 'the architectural arms race fails' is not supported by the evidence presented. This should be presented as a specific data point under specific conditions, not a general conclusion."

---

## Reviewer 3 — Perspective Reviewer (跨学科 + 实际影响)

| 维度 | 分数 | 评语 |
|------|------|------|
| Originality | **35** | Open testbed is the real innovation here, though it's infrastructure not method. |
| Methodological Rigor | **50** | Validator is a genuine contribution; pipeline ablation is standard. |
| Evidence Sufficiency | **45** | No factory data, no cost analysis, no comparison with commercial solvers on wall-clock. |
| Argument Coherence | **55** | "1% makespan = millions" in Broader Impact is not connected to own results. |
| Writing Quality | **70** | Readable for someone outside FJSP; could use more explanatory figures. |

**关键评论**：
> "The paper misses an opportunity to connect its findings to broader themes: (1) when do simple methods suffice in combinatorial optimization? (2) what is the right balance between learned and handcrafted features? These questions matter beyond FJSP. The current framing is too narrow."

---

## Devil's Advocate — 核心论点挑战

### 最强反论点

**"这篇论文的 5 个贡献，如果独立拆开看，没有任何一个达到 ASC/A 档期刊的发表标准。"**

逐个审视：

1. **MK13 SOTA (−14)**：一个 instance 的 3.3%。FJSP 文献中，单个 instance 的改进（没有新方法）通常不足以单独发表。
2. **L2I head-to-head 15/15 win**：5-stage pipeline vs 1-stage method。好比拿一辆装了涡轮增压的汽车和一辆原厂车比速度，然后声称发动机更好。不成立。
3. **3 methodological pitfalls**：HGT dropout 是 PyTorch 101；CP-SAT FEASIBLE 是 OR 101；SA revert 是参数调优 101。这些是 "engineering notes"，不是 "research contributions"。
4. **321-param beats 32K/110K/180K**：在 single-instance 设置下。大 encoder 的价值在于 cross-instance 泛化。好比在一条熟路上测试 —— 不需要地图，但在陌生城市就需要。测试条件消除了 encoder 的需要，然后声称 encoder 没用 —— 循环论证。
5. **5-stage pipeline + ablation**：ILS + SA + TS 的 stage-wise ablation 确实没有人系统做过，但 ILS、SA、TS 各自的效果在 OR 文献中已经充分研究过了。这是 "confirmation of known effects" 而非 "discovery of new effects"。

### 被忽略的替代解释

1. **MK13 的 −14 可能来自 ILS 而非 REINFORCE**：论文显示 REINFORCE contributes −21.9%，ILS contributes −12.1%。但 REINFORCE 的 −21.9% 是从 EF baseline 算的，ILS 的 −12.1% 是从 REINFORCE 的结果算的。如果直接从 EF + ILS 开始（跳过 REINFORCE），gap 可能是多少？没有做这个 ablation。

2. **321-param MLP 在 single-instance 上足够 ≠ 简单方法更好**：single-instance 的 action space 很小（几十到几百个 candidate actions），任何合理的 function approximator 都能学会。这不是 MLP 的胜利，而是问题设置的胜利。

3. **5 stage pipeline 的收益可能来自 compute budget 而非 algorithm design**：REINFORCE 50 episodes + ILS 4 starts × k∈{5,10,20,50} + SA 2000 iters + TS 800 iters —— 总 compute 远大于任何单个 baseline。如果给 Random Search 同等的 compute budget，gap 会是多少？

### 缺失的利益相关者视角

- **工业调度软件供应商**（Hexaly, IBM CPLEX, Siemens）：他们关心 wall-clock performance + robustness + ease of deployment。这篇论文没有提供 wall-clock 比较，single-instance per model 的部署成本太高（15 instances = 15 models）。
- **OR 理论研究者**：他们关心 why the method works。这篇论文只提供了 what works，没有 why。
- **AI 伦理研究者**：全 AI 生成的项目管理（"全托管 AI 做的"）的 authorship 归属问题。这本身就是一个值得讨论的问题，但论文没有涉及。

### "So What?" 测试

> 假设这篇论文发表了，5 年后有人引用它。他们会引用什么？
>
> 大概率不是 MK13 的结果（会被后续方法超越），不是 "321-param beats 32K"（设置太特殊），而是：
> - "Schedule validation and release, as advocated by [this paper]"
> - "The HGT dropout pitfall, as documented by [this paper]"
>
> 也就是说，这篇论文的遗产是 **reproducibility infrastructure**，不是任何算法或方法贡献。论文的 framing 应该与这个遗产对齐。

---

## 编辑决策综合

### 五位审稿人的共识与分歧

| 议题 | EIC | R1 (方法) | R2 (领域) | R3 (跨学科) | DA | 共识？ |
|------|-----|-----------|-----------|-------------|-----|--------|
| Reproducibility infrastructure 有价值 | ✅ | ✅ | ✅ | ✅ | ✅ | **强共识** |
| 实验报告需要 mean±std | ✅ | ✅ | ✅ | ⚠️ | ✅ | **强共识** |
| L2I 比较不公平 | ✅ | ✅ | ✅ | — | ✅ | **强共识** |
| 无方法创新 | ✅ | ✅ | ✅ | ✅ | ✅ | **强共识** |
| 需要 cross-instance | ⚠️ | ✅ | ✅ | — | ✅ | **共识** |
| 适合 ASC | ⚠️ | ⚠️ | ⚠️ | ❌ | ❌ | **分歧** |

### 最终编辑决定

**Decision**: **MAJOR REVISION** (如果在 ASC/EAAI/COR 这类 B+ 期刊)
**Decision**: **REJECT with encouragement to resubmit as reproducibility study** (如果在 JMS/EJOR 等 A 档期刊)

**如果作者选择投 ASC，需要满足以下条件才能进入 Minor Revision → Accept 路径**：

1. 补充 mean±std + Wilcoxon test（不可协商）
2. 修正 L2I 比较的不公平性（不可协商）
3. 补充 Hurink 代表性实例（强烈建议）
4. 补充 wall-clock comparison（强烈建议）
5. 降级 Abstract 中的 "NEW SOTA" 叙事，突出 reproducible testbed（强烈建议）
6. 深入分析 hard instance 失败原因（建议）
7. 补充 architecture diagram + Gantt chart + training curve（建议）

---

# PART C: 代码级验证

## 代码结构评估

```
RL_FJSP/
├── fjsp/
│   ├── parser/fjs_parser.py          ✅ 标准 Brandimarte 解析器，干净
│   ├── env/dispatch_env.py           ✅ 正确的 MDP 建模
│   ├── scheduler/validator.py        ✅ 四个维度的 schedule 验证，完整
│   ├── scheduler/dispatch_rules.py   ✅ EF/greedy baseline
│   ├── scheduler/local_search.py     ⚠️ ILS/SA/TS 是标准实现
│   └── utils/scaling.py              ✅ 时间缩放
├── rl/
│   ├── agents/                       ✅ 6 agents (REINFORCE, AC, Graph AC, HGT, PPO, Imitation)
│   ├── models/                       ✅ MLP action scorer + graph variants
│   └── train*.py                     ✅ 训练循环
├── scripts/                          ✅ 实验运行脚本
├── data/
│   ├── instances/brandimarte/        ✅ MK01-MK15
│   └── results/                      ✅ JSON 存盘 (schedule + makespan)
└── paper/                            ✅ 完整 LaTeX 源码
```

### 代码质量评价

| 维度 | 评分 | 说明 |
|------|------|------|
| 可读性 | 良好 | 类型注解完整，docstring 清晰 |
| 模块化 | 良好 | parser/env/scheduler/RL 分层合理 |
| 正确性 | 良好 | Validator 实现 4 项检查，逻辑正确 |
| 可复现性 | 良好 | 固定 seed，JSON 存盘 |
| 测试覆盖 | 弱 | 有 smoke test 但无单元测试覆盖 |
| 文档 | 一般 | CLAUDE.md + README 可用但不详尽 |

### 关键代码片段审查

**`fjsp/scheduler/validator.py` — Schedule Validator**

```
✅ Precedence check: current.start >= prev.end (L103-111)
✅ Eligibility check: option.duration == expected_duration (L84-93)
✅ No-overlap check: right.start >= left.end (L117-125)
✅ Completeness check: all expected operations present (L95-101)
```

这个 validator 的四个维度是 FJSP schedule 合法性的充要条件。**这是该领域第一次有人公开提供这样的 validator。** 单独拿出来可以作为一个 mini-tool 被社区使用。

**`rl/agents/reinforce_agent.py` — PA-REINFORCE Agent**

```
✅ EMA baseline (L155): baseline = 0.9 * baseline + 0.1 * reward
✅ Gradient clipping (L161): max_norm=1.0
✅ Greedy evaluation during training (L165-180)
✅ Best-model checkpointing (L166-169)
```

这是标准的 REINFORCE 实现，无 bug，但也无创新。

**`experiments/rl_baselines/l2i_baseline.py` — L2I Re-implementation**

```
⚠️ 从 random schedule 开始（不是 RL pretrained）
⚠️ 只有 200 steps（不是 full episode-based training）
⚠️ 4 operators × max 50 trials per operator
⚠️ 没有 ILS/SA/TS 后处理
```

L2I re-implementation 是按照 ICLR 2024 论文的原意做的（learning to select improvement operators）。但正是因为它缺少后处理，而作者的 pipeline 有 4 阶段后处理，所以比较不公平。

**⚠️ `experiments/analysis/check_gaps.py` — 数据一致性检查**

```python
data = json.load(open(r'd:\desktop2\RL_FJSP\data\results\sota_final.json'))
data.sort(key=lambda x: -x['gap_to_lit'])
```

注意到这个文件是 `git status` 中的 untracked file（`?? experiments/analysis/check_gaps.py`）。它与 `paper/SUMMARY.md` 中声称的 SOTA 数据可能存在时间差——内部数据一致性需要检查。

---

# PART D: 你真的能发哪里 —— 诚实刊物评估

## 按创新水平排序的投稿选项

### ✅ 高概率 (70-90%)

| 目标 | 类型 | 预期 | 工作量 |
|------|------|------|--------|
| **NeurIPS Workshop on Learning and Combinatorial Optimization** | Workshop | 几乎必中 | 1 周（改 framing） |
| **ReScience** | Journal (reproducibility-focused) | 高概率 | 1 周（改 framing） |
| **ICML Workshop on PODS/PRB** | Workshop | 高概率 | 1 周 |

### ⚠️ 中等概率 (40-60%)

| 目标 | 类型 | 预期 | 工作量 |
|------|------|------|--------|
| **Applied Soft Computing (IF 8)** | Elsevier Journal | 50-60%（需补 P0-P1） | 2-3 周 |
| **Engineering Applications of AI (IF 7)** | Elsevier Journal | 50-60%（同上） | 2-3 周 |
| **Computers & Operations Research (IF 5)** | Elsevier Journal | 40-50% | 2-3 周 |
| **Expert Systems with Applications (IF 8)** | Elsevier Journal | 40-50% | 2-3 周 |

### ❌ 低概率 (10-30%)

| 目标 | 类型 | 预期 | 原因 |
|------|------|------|------|
| **J. Manuf. Syst. (IF 12)** | Elsevier A 档 | 20-30% | 需要 method innovation + factory data |
| **EJOR (IF 6)** | OR 顶刊 | 20-30% | 需要 method innovation |
| **IEEE TII / TNNLS** | IEEE A 档 | 10-20% | 需要方法贡献 |
| **NeurIPS / ICML / ICLR 主会** | ML 顶会 | <5% | 需要 ML 方法论贡献 |

---

# PART E: 修正路线图 (Revision Roadmap)

## 路线 A: 快速路径 → Workshop (1-2 周，低风险)

**做了什么**：
1. 改 framing 为 "A Reproducible Empirical Benchmark Study of RL for FJSP"
2. 弱化 "NEW SOTA" 叙事，强化 "open testbed + schedule validator" 叙事
3. 改 Abstract：**第一句不是 "new best-known makespan"**，而是 **"We release the first fully reproducible open testbed for RL-based FJSP"**
4. 补 5-seed mean±std（纯计算，1 小时）
5. 加 Wilcoxon p-value 表（1 小时）
6. 改标题为 **"FJSP-RL Testbed: A Reproducible Benchmark with Schedule Validator and 375-Run Empirical Study"**

**投稿**: NeurIPS CO Workshop / ReScience
**接受率**: 70-80%

## 路线 B: 中风险路径 → ASC/EAAI (2-3 周)

在路线 A 的基础上 **额外补**：
1. **L2I 公平比较**：给 L2I 加相同的 ILS+SA+TS 后处理，重新报告 head-to-head
   - 或者：只比较纯 RL 阶段 (REINFORCE vs L2I RL)，清晰标注比较范围
2. **Hurink 3 实例**（edata, rdata, vdata）：跑完整 pipeline，补结果表
3. **Wall-clock 对比表**：all agents 的训练时间 + 推理时间 + 最终 makespan
4. **深入分析 MK04/06/10/15 失败**：每个 instance 的结构特征 → why pipeline fails
5. **保留 3 pitfalls 但降级为 §7 Engineering Notes**（不是 §7 Methodological Contributions）
6. **加 architecture diagram + Gantt chart + training curve + pipeline flowchart**
7. **Cover letter 强调整 reproducibility 贡献**

**投稿**: Applied Soft Computing / Engineering Applications of AI
**接受率**: 50-60%

## 路线 C: 高风险路径 → JMS/EJOR (4-6 周)

在路线 B 的基础上 **额外补**：
1. **Cross-instance training**：MK01-05 联合训练 → 零样本测试 MK06-15
   - 即使 0-shot gap 很大，也是诚实的数据点 → 转换叙事为 "we identify cross-instance generalization as the key bottleneck"
2. **至少一个 Hurink full benchmark**（edata/rdata/vdata 全 15+ 实例）
3. **与至少一个商业 solver（Hexaly 或 Gurobi）的 wall-clock vs makespan Pareto 对比**
4. **Method contribution**：需要开发至少一个新组件（adversarial training / learned perturbation strength / multi-objective extension / dynamic feature selection）
5. **加 real factory case study** 或 at minimum simulation-based transfer experiment

**投稿**: J. Manuf. Syst. / EJOR
**接受率**: 30-50%（如果有 strong cross-instance + method novelty）

---

# PART F: 最终建议

## 核心问题，一句话

> 这篇论文最诚实的定位是 **"reproducibility study + benchmark testbed"**，当前版本用 "NEW SOTA + 5 contributions" 的框架包装了一个实质上没有方法创新的工作。审稿人会看穿这个包装。

## 我的推荐

**不要硬投 A 档期刊**。当前版本的实质贡献不足以支撑 JMS/EJOR 的接受标准。硬投几乎必被拒，且会浪费 3-6 个月的审稿周期。

**推荐路径**：

### 短期（本月）：执行路线 A → 投 Workshop

- **目标**: NeurIPS 2026 Workshop on Learning and Combinatorial Optimization (8 月截止)
- **优势**: 几乎必中，快速获得一个 publication
- **风险**: 低
- **代价**: 接受 "这不是顶刊论文" 的现实

### 中期（2-3 月）：执行路线 B → 投 ASC

- **在 Workshop 接受的基础上**，补 cross-instance 实验 + Hurink + wall-clock
- **目标**: Applied Soft Computing (随时可投)
- **优势**: 有两个 publication（Workshop + Journal），且 Workshop 的 feedback 可以改进 Journal 版本
- **风险**: 中

### 长期（6 月+）：执行路线 C → 投 JMS

- **仅在路线 B 成功且 cross-instance 效果好的情况下**
- 加 real factory data 或 multi-objective extension
- 投 J. Manuf. Syst.

## 关于"全托管 AI 做项目"的特别说明

⚠️ 这是一个需要认真对待的问题。论文的作者是 "The RL_FJSP Project Team"，但项目是你 "全托管 AI 做的"。这涉及：

1. **Authorship**: 如果 AI 工具（Claude/其他）实质性参与了代码编写、实验设计、论文撰写，按照 ICMJE authorship 标准，AI 不能作为作者。你需要在论文中明确披露 AI 使用（§11 Broader Impact 或单独的 Disclosure Statement）。

2. **Intellectual contribution**: 如果方法选择、实验设计、结果解读都是由 AI 完成的，你的 intellectual contribution 是什么？审稿人可能会问："What did the human authors actually decide?"

3. **Reproducibility by others**: 如果项目是 AI 生成的，其他人能否以相同方式复现？代码是公开的，但 AI 生成过程的随机性意味着"同样的 prompt 可能产生不同的代码"。这削弱了 reproducibility 声称。

4. **建议**: 在论文中添加 AI Usage Disclosure（参见 ARS `/ars-disclosure`），清晰说明：
   - 哪些部分由 AI 辅助（代码生成、文献搜索、论文初稿）
   - 哪些部分由人类作者决策和验证（方法选择、实验设计、结果验证、论文终稿）
   - AI 工具的具体版本和用途

## 最终, 如果你只做三件事

按优先级：

1. **补 mean±std + Wilcoxon test**（不可协商，任何期刊都需要）
2. **修正 L2I 比较**（要么公平化，要么降级声称）
3. **改 Abstract 和 Title**（从 "NEW SOTA" → "Reproducible Benchmark Study"）

这三件事做完，论文在 ASC 的接受概率从 30% → 55%。
再加 Hurink + wall-clock → 65%。
再加 cross-instance → 75%。

---

*本审稿报告基于 Nature reviewer 框架 + ARS 五审稿人框架 + 代码级验证，于 2026-06-13 完成。所有评估均基于提供的论文手稿和代码库。未评估的部分标为 "Not assessable from provided material"。*
