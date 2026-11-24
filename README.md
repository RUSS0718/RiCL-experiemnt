# RiCL Experiment

RiCL Motivation Study 的独立实验仓。项目采用 Python 3.12 与 uv；实验能力按工单逐个纵向实现。当前实现单样例运行、OpenAI-compatible HTTPX 派发、SQLite 审计台账和显式续跑。

## 快速开始

```powershell
uv sync --frozen
uv run python -m ricl_experiment --help
uv run pytest
uv run ruff check .
```

模型配置示例位于 `configs/models/default.toml`，单样例清单格式见 `configs/examples/single-sample.json`。运行前按所选端点修改模型 ID、端点和静态参数能力，并在进程环境中设置配置所引用的 API key 环境变量（示例为 `RICL_API_KEY`）。不要将凭据写入配置或提交到 Git。

```powershell
uv run python -m ricl_experiment run `
  --study ms2 --phase pilot `
  --config configs/models/default.toml `
  --manifest configs/examples/single-sample.json
```

MS2 控制样例使用受控或真实 OpenAI-compatible endpoint，不把答案键传给生成请求：

```powershell
uv run python -m ricl_experiment ms2 run `
  --phase pilot `
  --config configs/models/default.toml `
  --manifest <prepared-ms2-manifest.json> `
  --answer-key <restricted/answer-key.json>

uv run python -m ricl_experiment ms2 resume `
  --run-id <run_id> `
  --config configs/models/default.toml `
  --manifest <prepared-ms2-manifest.json> `
  --answer-key <restricted/answer-key.json>
```

每题预登记三个候选槽位；非空 Checkpoint 固定原样前缀并独立续推两次。轨迹写入 `traces/ms2.json`，答案键保存在 `restricted/answer-key.json`，续推请求只接收题面、原样 Checkpoint 和中性模板。统一旧入口也支持 `run --study ms2 --answer-key ...` 与对应的 `resume --study ms2`。

成功或派发失败时都会显示 `run_id`。从原运行显式续跑，并传入原配置与清单：

```powershell
uv run python -m ricl_experiment resume `
  --run-id <run_id> `
  --config configs/models/default.toml `
  --manifest configs/examples/single-sample.json
```

恢复会校验配置、完整清单和提示指纹；有任何差异都必须创建新 Run。每个 Run 位于 `artifacts/motivation/<run_id>/`：`metadata.json` 保存脱敏配置/请求摘要与指纹，`run.sqlite3` 是逻辑调用、尝试、响应、用量/预算和唯一分析归属的权威台账，`responses/` 保存原始响应文件。未知用量保留输出 Token 预留；不确定尝试的重发受配置的尝试上限、累计预算和单调度进程锁限制。重复响应保留逐次审计记录，同一逻辑调用仅有一个分析归属。

`max_reserved_output_tokens` 表示所有尝试共享的输出 Token 预算；每次发送先按 `max_tokens` 预留，核实用量后计实际输出 Token，未知用量继续占用预留。不同 OpenAI-compatible 端点的可用参数可能不同，`model.capabilities` 必须按端点能力维护，不支持的配置参数会在 Run 创建和网络派发前拒绝。

依赖与锁文件由 `pyproject.toml` 和 `uv.lock` 管理。不要提交 API 凭据、固定数据副本或运行产物。

## 当前范围

旧 `run` 和 `resume` 仍接受一个带 `sample_id` 与 `prompt` 的版本化单样例清单；MS2 控制入口接受准备好的多样本清单和分离的受限答案键。Issue #15 增加本地固定快照准备 `prepare`、确认样本组排除导出 `exclude-groups` 和同阶段 MS2/MS3 重叠报告 `overlap`；操作说明见 [docs/dataset-manifests.md](docs/dataset-manifests.md)。Issue #16 实现 Checkpoint、独立续推和独立答案判分引用；Issue #19 增加 `ms3-diagnose` 三方法盲化诊断和 `ms3-analyze` 固定分母离线分析；Issue #20 增加离线可审计的 MS3 四组修复 `ms3-repair` 与分析 `analyze-ms3-repair`。MS3 分析只读取落盘工件，不调用模型。Issue #17/#18 的 MS2 证据评分和主分析，以及 MS1 编排、报告和绘图仍属后续工单。研究架构和 artifact 边界见 [docs/README.md](docs/README.md)。
