# 论文项目 现状 + 问题 诚实自评

**更新日期**:2026 年 1 月
**目的**:在投任何期刊/会议前,理清"我们到底有什么、缺什么、能发什么"。

---

## 一、目前已做的(事实)

### 1.1 代码与数据(20+ commits)

| 项 | 状态 | 备注 |
|---|---|---|
| 6 个 RL agent 训练 pipeline | ✅ 完成 | PA-REINFORCE / BC+PPO / HGT / L2I reimpl / 2-stage AC / graph AC |
| Brandimarte MK01--MK15 | ✅ 解析器 | 标准 benchmark |
| 5-stage 后处理 | ✅ 完成 | REINFORCE + 4-start ILS + SA + tabu search |
| OR-Tools CP-SAT 300s baseline | ✅ 完成 | OPTIMAL / FEASIBLE / INFEASIBLE 状态分离 |
| L2I re-impl(ICLR 2024) | ✅ 完成 | 5 seed 跑完 |
| 5 seed 跑完 6 agents × 15 instances | ✅ 完成 | 375 runs |
| 5 个 SOTA schedule 存盘(JSON) | ✅ 完成 | mk03/08/13/14 |
| Schedule validator | ✅ 完成 | precedence / eligibility / no-overlap / makespan check |
| Open testbed | ✅ GitHub | 1 命令 reproduce |

### 1.2 实验结果(数据)

| 指标 | 数字 | 备注 |
|---|---|---|
| NEW SOTA | **MK13 makespan 416** | 前 best 430,**−14** |
| TIED OPT | **MK03, MK08, MK14** | OR-Tools 证 OPTIMAL |
| Mean gap to lit-best | **5.64%** | 15 instances |
| 5 seed mean std | ⚠️ 报 best-of-5 | 严格说应该报 mean±std |
| Wall-clock vs OR-Tools | ❌ **输** | 小实例 9× 快,大实例 0.1-0.2× 慢 |
| L2I head-to-head | **15/15 完胜** | +63.7% vs +5.6% mean gap |
| Stage ablation | 完成 | REINFORCE −21.9%, ILS −12.1%, **SA +6.1% 回退**, TS 仅救 MK15 |

### 1.3 论文(26 页,37 真实引用)

| 节 | 状态 | 页数 |
|---|---|---|
| §1 Introduction(3 视角 5 贡献) | ✅ | 1.5 |
| §2 Related Work(7 子节) | ✅ | 6 |
| §3 Problem Formulation(4 子节) | ✅ | 3.5 |
| §4 Method(4 子节 + 2 algorithm) | ✅ | 4 |
| §5 Experimental Setup | ✅ | 1 |
| §6 Results(11 表 + 3 图) | ✅ | 4 |
| §7 Discussion(3 Methodological Pitfall + 2 analysis) | ✅ | 2 |
| §8 Limitations(L1-L7) | ✅ | 1 |
| §9 Future Work(F1-F4) | ✅ | 0.5 |
| §10 Conclusion | ✅ | 0.5 |
| §11 Broader Impact | ✅ | 0.5 |
| Appendix A/B/C | ✅ | 2 |
| **总计** | ✅ | **26 页** |

### 1.4 双模板

| 模板 | 用途 | 状态 |
|---|---|---|
| `paper/main.tex`(NeurIPS 风格 26 页) | 投会议 | ✅ |
| `paper/main_asc.tex`(Elsevier 双栏 9 页) | 投 Appl. Soft Comput. | ✅ |
| `paper/SUMMARY.md`(14 节 markdown 总结) | 内部 review | ✅ |

---

## 二、真正的问题(残酷诚实)

### 问题 1: **没有真正的算法创新**(致命)

| 维度 | 我们 | 顶刊需要 | 距离 |
|---|---|---|---|
| 算法 | per-action REINFORCE = Williams 1992 | 新训练方法 / 新 loss / 新 exploration | **0** |
| 架构 | 321-参数 MLP | 新 encoder / 新 attention | **0** |
| Pipeline | RL + ILS + SA + TS 全是教科书 | 新组合 / 新调度逻辑 | **小** |
| Feature | 8 维 handcrafted | 自动化 / learned | **0** |
| 训练 | REINFORCE + EMA baseline | 新 RL 算法 | **0** |

**审稿人原话可能是**:"This is REINFORCE with handcrafted features. There is no method contribution."

### 问题 2: **没有真正的能力创新**(致命)

| 能力 | 我们 | 顶刊需要 | 距离 |
|---|---|---|---|
| Cross-instance 训练 | ❌ **没有** | 一个模型训多 instance | **远** |
| Cross-size 泛化 | ❌ **没有** | 训小测大,零样本 | **远** |
| Transfer to 真实数据 | ❌ **没有** | 工厂实测 / 公开真实数据 | **远** |
| Multi-objective | ❌ **没有** | makespan + 能量 / tardiness | **远** |
| Stochastic / Robust | ❌ **没有** | 机器宕机 / 随机工序 | **远** |
| Foundation model | ❌ **没有** | 一个模型适用 10+ benchmark | **远** |

**审稿人原话可能是**:"This is a single-instance study. There is no generalization story."

### 问题 3: **没有真正的问题创新**(致命)

| 创新角度 | 我们 | 顶刊需要 |
|---|---|---|
| 问题本身 | Brandimarte MK01-15(30 年老题) | 新 benchmark / 新 formulation / 新约束 |
| 评估方式 | makespan gap to lit-best | 新评估维度 |
| 工业 relevance | 没真实工厂数据 | 实地部署 / shadow mode |

**审稿人原话可能是**:"Brandimarte has been the standard for 30 years. Why this benchmark specifically?"

### 问题 4: **实验设计弱点**(重要)

| 弱点 | 影响 | 严重度 |
|---|---|---|
| **报 best-of-5 而非 mean±std** | 顶刊硬伤,Wilcoxon 缺失 | **高** |
| **L2I 比较不公平** | 我们 5 stage,L2I 1 stage,赢是 stages 而非 method | **高** |
| **Wall-clock 输** | 小实例赢 9×,大实例输 5-10× | **中** |
| **5 个 SOTA 用自家 validator** | 无第三方独立验证 | **中** |
| **没有 Hurink / SD1/SD2** | 仅 Brandimarte | **中** |
| **MK12 "TIED OPT" 之前写过但实际 +16** | 内部矛盾,需修正 | **高** |

### 问题 5: **方法论 pitfall 的真正分量**

| Pitfall | 我说的分量 | 审稿人会怎么说 |
|---|---|---|
| HGT dropout bug | "5-10 单位 invalidates prior numbers" | "Anyone writing PyTorch knows this" |
| OR-Tools FEASIBLE 误用 | "Several papers do this" | "This is basic OR knowledge" |
| SA 100% reverts on 4 instances | "Should be guarded by revert test" | "This is a configuration choice, not insight" |

**结论**:3 个 pitfall 都不是 method contribution,是 basic engineering hygiene。**可保留但分量低。**

### 问题 6: **NEW SOTA 的真实贡献度**

| 项 | 数据 | 真实分量 |
|---|---|---|
| MK13 = 416 vs 前 best 430 | 14/15 = 3.3% 改进 | **小** — 一个 instance 的 3% 改进 |
| 3 TIED OPT | 3 个匹配 | **不构成 SOTA**,是 matching |
| Mean gap 5.64% | 15 instances | 与 L2D 持平,无显著进步 |

**审稿人原话可能是**:"One SOTA on one instance is not a method contribution."

### 问题 7: **L2I head-to-head 的真实分量**

| 项 | 数据 | 真实分量 |
|---|---|---|
| 我们 15/15 win | +5.6% vs +63.7% | **不公平比较** — 我们 5 stage, L2I 1 stage |
| | | 即使赢了,也不证明 PA-REINFORCE 好 |
| | | 只能证明 "5-stage pipeline > 1-stage L2I" |

**审稿人原话可能是**:"Comparing a 5-stage pipeline against a single method is not a fair baseline."

---

## 三、我们到底有什么(可发表的部分)

### 3.1 真实的 strengths

| # | 真实 strength | 顶刊能要吗 |
|---|---|---|
| 1 | **Schedule validation + JSON 存盘** — 全 FJSP-RL 领域没人做 | ✅ NeurIPS CO Workshop 收 |
| 2 | **L2I re-impl + 头对头** — 全 FJSP-RL 领域没人做 | ✅ reproducibility track |
| 3 | **5 seed 数据 + 5 stage 完整 ablation** | ✅ reproducibility track |
| 4 | **3 个 pitfall 公之于众** | ✅ helpful 但 small |
| 5 | **MK13 SOTA** | ✅ 小 |
| 6 | **321-参数 REINFORCE 打败 32K/110K/180K** | ✅ negative result, 但不构成正面 |

### 3.2 真实的定位

**这个论文本质是**:**一个诚实、完整、可复现的 FJSP-RL empirical benchmark study**。

**适合发表**:
- ✅ NeurIPS Workshop on Learning and Combinatorial Optimization
- ✅ ICML Workshop on PRB (Principle of Reliable ML)
- ✅ IJCAI Workshop on RL for Industrial Applications
- ✅ **ReScience** (reproducibility journal)
- ⚠️ **Appl. Soft Comput. (勉强)** — 能投但 60% 拒
- ⚠️ **Expert Systems with Applications (勉强)** — 同上

**不适合**:
- ❌ J. Manuf. Syst. (IF 12) — A 档,需要 method innovation
- ❌ EJOR (IF 6) — OR 经典,需要 method innovation
- ❌ IEEE TNNLS / TII — A 档,需要 method innovation
- ❌ NeurIPS / ICLR / ICML 主会 — top 顶会

---

## 四、3 个真实选项

### 选项 A:**接受 reproducibility 定位,投 Workshop**

| 项 | 详情 |
|---|---|
| 目标 | NeurIPS CO Workshop / ICML PRB Workshop / ReScience |
| 周期 | 1-2 周 |
| 工作 | 改 framing 为 "reproducible empirical study" |
| 接受率 | 70-80% |
| 风险 | 低 |
| 收益 | 一个 publication |

### 选项 B:**加跨实例训练(cross-instance capability)**,冲 A 档

| 项 | 详情 |
|---|---|
| 目标 | 投 J. Manuf. Syst. / EJOR / ASC |
| 周期 | 1-2 周 |
| 工作 | 改 8 features 为 size-agnostic,训练 MK01-05 联合,零样本测 MK06-15 |
| 接受率 | 30-50%(如果有强 0-shot 结果) |
| 风险 | 中(0-shot 可能输) |
| 收益 | A 档 |

**关键修改**:
- 8 features 加 size encoding
- 训练循环改 multi-instance
- 报告 0-shot mean gap on 10 held-out instances
- 与 L2D / L2I 的 cross-size 报告对齐

### 选项 C:**降级 abstract 与 framing,投 EAAI / ASOC / COR**

| 项 | 详情 |
|---|---|
| 目标 | 投 Eng. Appl. AI / Appl. Soft Comput. / Comput. Oper. Res. |
| 周期 | 1 周 |
| 工作 | 改 abstract 强调 reproducibility + benchmark,弱化 method claim |
| 接受率 | 50-60% |
| 风险 | 中(可能还是拒,reviewer 觉得"no method contribution") |
| 收益 | B+ 档 |

**关键修改**:
- 改 abstract 强调 "reproducible testbed + head-to-head" 而非 "SOTA"
- 加 5 seed mean±std
- 加 Wilcoxon p-value
- 改 §7 从 "methodological pitfall" 降级为 "engineering notes"
- 加 reproducibility 报告(Bibliography of all FJSP-RL papers evaluated)

---

## 五、我的推荐(诚实)

**用户问:"你真的有创新吗?"**

我的回答:**没有真正的算法/能力/问题创新。**

5 个贡献的真正分量:
- 2 个是 empirical finding(NEW SOTA + L2I head-to-head)→ **小**
- 1 个是 methodological(3 pitfalls)→ **更小**
- 1 个是 architectural insight(321 打败 32K)→ **negative result, 不构成正面**
- 1 个是 algorithmic(5-stage pipeline + ablation)→ **不构成算法创新**

**真实分量排名**:
1. L2I re-impl + 头对头(最大,但不公平)
2. Schedule validation(全领域没人做,但 reviewer 觉得 trivial)
3. MK13 SOTA(小,一个 instance 的 3%)
4. 3 pitfalls(基本工程规范)
5. 5-stage ablation(标准 ablation)

**如果用户接受"reproducibility 论文"的定位**:
- 投 Workshop 几乎必中
- 投 EAAI/ASOC 50-60% 中
- 投 JMS/EJOR 30% 中

**如果用户想要 A 档 method innovation**:
- 必须加 cross-instance training(1-2 周)
- 或者加 multi-objective(2-3 周)
- 或者加 adversarial FJSP(2-3 周)

**如果用户硬投 A 档,几乎必拒**。建议:
- (a) 接受现实,**先投 Workshop,加一篇 2 页附录给 reproducibility**,快速中
- (b) 同期做 cross-instance 训练,**2 周后**投 A 档

---

## 六、待用户决策

| 选项 | 目标 | 周期 | 接受率 | 我的建议 |
|---|---|---|---|---|
| A | Workshop (NeurIPS CO / ICML PRB / ReScience) | 1-2 周 | 70-80% | **最稳** |
| B | 加 cross-instance 训练 → A 档 | 2-3 周 | 30-50% | **最佳收益** |
| C | 改 abstract → EAAI/ASOC/COR | 1 周 | 50-60% | **较快** |

**我现在等你的指示**:
- (a) 改 framing 投 Workshop?
- (b) 加 cross-instance 训练,2 周后投 A 档?
- (c) 改 abstract 投 B+ 档?
- (d) 其他?
