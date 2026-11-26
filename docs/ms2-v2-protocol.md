# MS2 v2 隔离复验协议

本提案的配置为 `configs/models/deepseek-flash-ms2-v2.toml`。本地冻结清单和哈希见 `artifacts/ms2-v2-isolated-proposal-20261125/dispatch-proposal.json`。范围为一轮 10 题隔离复验。确认实验需要另行签收。

## 冻结设计

- 在配置中显式指定 `ms2.protocol_version = "ms2-v2"`。`phase` 只记录题单隔离阶段。旧版配置缺少该字段时，历史续跑仍按原规则校验。新启动的所有 MS2 阶段都要求显式版本。
- 使用固定种子 20261127，从两个锁定快照各抽 5 个未观测原题组。保留原 Pilot 的 MATH-500 学科和难度配额。此次 Omni-MATH 从计数、几何、多项式和组合题的五个固定层各抽一题。保留受限的排除列表、选中清单和参考答案哈希。
- 各题有三条固定槽位，起步策略依次为直接建立关系、寻找结构或不变量、转成另一种表示。Checkpoint 只建立一个可核查局部子目标，记录条件与推导并留下实质步骤。每题三个槽位同预算。MATH-500 Level 4 和 Omni-MATH 难度低于 5 使用 512，MATH-500 Level 5 和 Omni-MATH 难度 5 使用 1,500。
- 每个非空 Checkpoint 派发两次同提示续推，每次最多 3,000 输出 Token。先冻结 Rater 与 Evidence 评分，两步 Evidence 使用 JSON 对象模式，三个评分阶段各限 2,000 输出 Token。保持温度 0.7 和已实测的非推理模式。同题重复路线仍留在分母。
- 盲审所有可用 Checkpoint 的早答状态以及有歧义的终答。评阅人访问题面、参考材料和候选文本，隐藏策略、评分以及同题续推。只有明确错误早答或无明确早答路线可进入共同候选集。明确正确早答和无法判定的明确早答分别保留记录。待审核的 Checkpoint 会阻断整题主分析。
- 固定预算耗尽且无可辨识终答、判分未知、执行失败和未派发分别列报。主分析仍要求合格路线的两次独立续推都能判定。继续使用完整题、排序可用题和独立证据覆盖的 8／5／5 门槛。续推成本报告以台账为准，保留重试与未知预留。离线截点统计描述已观测成功的保留数，不估算改变上限后重新生成的效果。
- Agent-Assessed、Tool-Verified、Derivation-Checked 和 Unknown 分别统计。模型独立证据评议只产出 Agent-Assessed 或 Unknown。只有保存可复跑检查、适用条件和结果时，才计入 Tool-Verified。Codex 盲审记录明确标注 Codex 来源，由研究负责人决定签收。保存工具或推导证据的原始出处和裁决，单独解释各类证据支持的效果。

## 审核与复算

```powershell
python -m ricl_experiment ms2-answer-review export --run-dir <run> --output <new-bundle>
python -m ricl_experiment ms2-answer-review import --run-dir <run> --bundle <bundle> --reviews <review-json> --output <new-reviewed-trace>
python -m ricl_experiment ms2-analyze --trace <reviewed-trace> --scoring-dir <run>/scoring --output <new-metrics>
python -m ricl_experiment ms2-pilot-check --run-dir <run> --reviewed-trace <reviewed-trace> --output <new-gate>
python -m ricl_experiment ms2-budget --run-dir <run> --reviewed-trace <reviewed-trace> --cutoff 512 --cutoff 1000 --cutoff 1500 --cutoff 2000 --cutoff 3000 --output <new-budget-report>
```

审阅人只读取 bundle 内的 `blind/`。导入 JSON 包含 `reviewer_kind` 和 `reviews`。检查点裁决取 `correct`、`incorrect`、`unknown` 或 `no_final_answer`，续推取前三者。每条填写 `review_id`、`reviewer_id`、`verdict`、`rationale`。部分终答导入带待审 ID 和 `partial` 标记，验收使用完整审核件。两轮历史 trace 与敏感性分析继续保留。

## 预算与授权

最多 30 次 Checkpoint、60 次主续推、90 次评分。每次逻辑调用最多两次尝试，因此最多 360 次发送。输出预留为 780,360 Token，计入重试。按供应商公布的 100 万 Token 上下文、峰时未缓存输入 USD 0.30／百万 Token 和输出 USD 1.20／百万 Token，取每次发送都用满输入上下文的保守上界，合计 USD 108.936432。提请批准 **USD 110 硬上限**，该数字并非预计账单。先前一轮实际记录的峰时未缓存估算为 USD 0.209052，两轮预算、题目和模型别名版本都需要单独留档。若供应商模型、上下文或计价变化，重新核定后再申请。[DeepSeek 官方模型和价格](https://api-docs.deepseek.com/quick_start/pricing/)

派发前核对提案中题单、答案、配置及采样计划的 SHA-256。中断后只允许用原 run_id 续跑，受相同预算和尝试上限约束。研究负责人批准前，该配置用于离线预检和独立验收。

认证故障修复后，经研究负责人明确授权，可对原 run_id 使用 `ms2 resume --redispatch-failed`。该选项仅用于尚无成功响应的永久失败逻辑调用；它保持请求指纹和每个逻辑调用最多两次尝试的限制，新增审计事件和尝试记录，不删除失败响应。全部尝试共享原输出 Token 池，历史未知用量预留仍计入总预算。普通 `resume` 仍拒绝永久失败。

真实提供方变更为 `https://code.aigcpro.org/v1` 与 `deepseek-v4.1-flash` 时，使用新的独立运行目录，不续用旧官方 DeepSeek 端点失败运行。题单、答案哈希、起步策略、每题输出上限、评分提示和 8／5／5 门槛保持冻结；旧运行的 31 次失败尝试继续计入 USD 110 总上限。若运行因进程等待超时或全局输出预留耗尽，自动提高 `timeout_seconds` 或 `max_reserved_output_tokens` 后启动新的独立重试目录；每次重试保留原运行台账，不提高单题 `max_tokens`，不改写截断结果，且由累计费用守卫在 USD 110 硬上限处停止。该操作属于执行恢复，不改变主分析定义。
