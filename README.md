# RiCL Experiment

RiCL Motivation Study 的独立实验仓。项目采用 Python 3.12 与 uv；当前仓库从最小 CLI 引导开始，研究流程会按工单逐个纵向实现。

## 快速开始

```powershell
uv sync --frozen
uv run python -m ricl_experiment --help
uv run pytest
uv run ruff check .
```

依赖与锁文件由 `pyproject.toml` 和 `uv.lock` 管理。不要提交 API 凭据、固定数据副本或运行产物。

## 当前状态

现阶段只提供包入口和帮助信息；`prepare`、`run`、`resume`、`analyze`、`plot` 是目标 CLI 接口，尚不可用于执行实验。研究架构见 [docs/README.md](docs/README.md)。
