# MS2 v2 隔离复验结果

本轮使用 `https://code.aigcpro.org/v1` 与 `deepseek-v4.1-flash`，读取用户环境 `API_KEY`。这是供应商／模型变更，独立记录；旧官方 DeepSeek 端点的失败运行及两轮代理网络失败台账均保留，不改写历史 trace。

## 运行与预算

- 成功运行：`artifacts/ms2-v2-isolated-proposal-20261125/live/aigcpro/attempt-03/motivation/11a10d8ff4eb4d10aa6580f0aa977d65/`。
- 成功台账：85 次 HTTP 尝试、85 个逻辑结果；输入 23,056 Token、输出 86,046 Token，全部用量已知。按冻结费率估算 USD 0.110172。
- 旧端点失败 31 次、代理连接失败 35 次均无成功响应；台账对未知输出预留 66,896 Token。按每个未知尝试保守取 100 万输入 Token 的守卫口径，累计上界 USD 19.9904472，剩余 USD 90.0095528。该上界不是供应商账单；失败 401 与 ConnectError 的实际计费需以平台账单确认。`live/failure-summary.json` 是首次 30 次失败快照；`live/aigcpro/old-failure-reconciliation.json` 记录其后 1 次授权重派及当前 31／32,180 Token 台账对账。
- 全局输出预留按预授权自动提高；三次运行配置中的 `timeout_seconds` 均为 120，未观察到实际提高。每题 `max_tokens`、提示、题单、抽样与两次续推规则不变。

## 真实轨迹与盲审

- 10 题共 30 个 Checkpoint：13 个有文本，17 个为空。空输出对应 `reasoning_tokens` 用满预算而正文为空，不是 Runner 丢失响应。
- 13 个可用 Checkpoint 中，独立 Codex 盲审判定真实早答正确 0 个、无提前终答 13 个；另有 4 个续推终答裁决为正确。评阅来源明确为 Codex，不冒充人工签核。
- 26 次续推已派发：22 次可判定，4 次盲审后正确；34 次因空 Checkpoint 未派发。另有 4 次固定预算耗尽且无可辨识终答，记为执行上限内未完成，不计为判分未知或执行失败。
- 评分工件 13 个：Rater 有效 10 个、无效 3 个；证据抽取有效 3 个、无效 10 个；Verified Progress 有效分数 3 个。没有 Tool-Verified 裁决，Agent-Assessed 与 Tool-Verified、Derivation-Checked、Unknown 分开统计。

## 离线复算与门槛

- 派生 trace、盲审包、逐题形状、指标、Pilot gate、预算敏感性均位于 `artifacts/ms2-v2-isolated-proposal-20261125/live/aigcpro/`。运行时的核心代码 diff、文件哈希和工作树状态保存在 `runtime-code.diff`、`runtime-manifest.json` 与 `runtime-worktree.txt`；成功运行的 `metadata.json.experiment_commit_sha` 仅为启动时的 HEAD，不能单独标识含未提交修复的实际运行代码。
- 正确离线合同的复算摘要为 `pilot-metrics.json`，门控为 `pilot-gate-recomputed.json`；`offline_recomputation` 状态为 `consistent`。
- 5 题有至少两条路线完成两次续推，但每题至少缺一个 Rater 或 Evidence 分数，因此主分析完整题 0/10；独立证据完整题 0/10；有效排序题 0/10。
- 8／5／5 门槛分别为 0/8、0/5、0/5，均未达标。已观测续推结果中缺少成功／失败差异，亦无法形成有效排序对。
- 门控仍列出真实 API 冒烟、人工盲审和预算签收缺失。Codex 辅助评阅不能替代研究负责人的人工签核；本轮不关闭实验 Issue，不启动确认实验。

## 结论

本轮产生了真实数学轨迹和完整离线复算，但不足以检验 MS2 主假设。主要阻塞是提供方模型把大量输出预算用于内部推理产生空 Checkpoint，以及 JSON 证据抽取／审核的解析失败。后续需由研究负责人签收新的协议修订、供应商计价与范围，不能直接通过降低门槛、选择性删除失败样本或反复修改验收器来取得形式通过。
