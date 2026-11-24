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

`run` 和 `resume` 目前只接受一个带 `sample_id` 与 `prompt` 的版本化 JSON 清单。Issue #15 增加本地固定快照准备 `prepare`、确认样本组排除导出 `exclude-groups` 和同阶段 MS2/MS3 重叠报告 `overlap`；操作说明见 [docs/dataset-manifests.md](docs/dataset-manifests.md)。准备好的多样本清单尚未接入批量 Study 执行。MS1/MS2/MS3 编排、分析与绘图 CLI 仍属后续工单。研究架构和 artifact 边界见 [docs/README.md](docs/README.md)。
