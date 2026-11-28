# RiCL Experiment 文档地图（按论文故事线与实验阶段组织）

本索引按**实验推进顺序**区分**当前论文主线（Active Mainline）**与**历史探索归档（Archived）**，原有文档路径全部保持不变以保证历史 GitHub Issue / PR 链接不失效。

---

## 一、当前论文主线（Active Paper Mainline）

### Stage 2 · Track 2 Frozen-200 双模型确认实验与归因（最新成果）
- **[Track 2 Frozen-200 双模型实验报告](track2-frozen200-report.md)**：`200` 题冻结基准、`DeepSeek-V4.1-Flash` 与 `Qwen/Qwen3-14B` 全预算及 `Accuracy vs. Total Tokens` 曲线、分数据集/成对胜负矩阵与 `provider_error` 清零审计链路（对应 [`results/track2-frozen200/`](../results/track2-frozen200/) 与 `#35` / PR `#40`）。
- **[Track 2 Frozen-200 深度归因与机制诊断](track2-frozen200-attribution.md)**：200 题逐路径拆解、`RiCL_v0.2` 微负 `B1-SC` / `B2` 逐题败因、`prune_graph()` 首错后独立节点拦截实证、以及针对外部三项核心批评的整改清单。

### Stage 0 · Motivation Study 核心可用动机（仅 MS1 主力 + MS2-A 辅助）
- **[Motivation Study v1.0 阶段报告](motivation-study-report.md)**：包含 **MS1**（`#29`，ProcessBench 20 条含错轨迹：`61/103` 确认依赖传播、`33/131` 独立可复用结论，可进论文主动机图）与 **MS2-A**（`#32`，`5/32` 含错高分辅助动机）；同时如实记录了 MS2-B / MS3-A / MS3-B 的止损状态。
- **[MS1 依赖标注与指标契约](ms1-dependency.md)**：MS1 双盲标注导入、第三方仲裁、传播深度与可复用命题比率计算规则。

### 基础设施与数据治理规范
- **[实验仓架构与产物契约](architecture.md)**：模块职责、运行台账、CLI 与数据角色隔离边界。
- **[固定快照与防泄漏数据集清单](dataset-manifests.md)**：固定 revision 校验、分层抽样、`closed-answer-v1` 确定性评测与 `question_group_id` 隔离规则。

---

## 二、历史探索与止损归档（Archived · 不直接作为论文主结论）

以下文档记录了 Motivation Study v1.0 及更早 Pilot 阶段的探索尝试、旧端点 `max_tokens=2000` 思考耗尽截断诊断与跨代修复天花板发现。它们支撑了我们转向“同模型 On-Policy 竞赛难题 + `max_tokens >= 8192`”的方法论决策，但**不作为最终论文的主证据引用**：

- [Motivation Study Morning Brief](motivation-study-morning-brief.md)（历史一页决策摘要）
- [Motivation Study first batch](motivation-study-first-batch.md)（历史首批覆盖归档）
- [Motivation Study batch decision](motivation-study-batch-decision.md)（历史首批覆盖与预算门禁）
- [Motivation Study revised protocol](motivation-study-revised-protocol.md)（历史修订协议与授权边界）
- [MS2 adjustment validation](ms2-adjustment-validation.md) / [MS2 v2 协议](ms2-v2-protocol.md) / [MS2 v2 隔离验证报告](ms2-v2-validation-report.md) / [MS2 v3 校准决策](ms2-v3-calibration-decision.md)（历史前缀评分探索与空正文阻塞记录）
- [MS3 Pilot 验收](ms3-pilot-acceptance.md) / [MS3 修复盲审复算](ms3-pilot-blind-review.md) / [旧 Pilot 离线审计](pilot-offline-audit.md)（历史 8 题跨代修复满分天花板与盲审复算记录）

*(跨仓全局故事线与论文目标定位见 `RiCL-home/README.md` 与 `RiCL-home/docs/PAPER_STORY_ACL2027.md`。)*
