# RiCL Experiment

**RiCL**（`RUSS0718/RiCL-experiemnt`）的独立基准构建、多基线同预算评测、冻结结果归档与归因分析仓库。

> 顶层研究故事线与三仓关系见总控仓 [`RiCL-home/README.md`](../RiCL-home/README.md)；完整文档分级索引见 [docs/README.md](docs/README.md)。

---

## 1. 仓库目录结构

```text
RiCL_experiment/
├── src/ricl_experiment/         # 核心实验包：基准准备、Track 1/2 编排、离线曲线评估、答案归一化与历史 MS1–MS3 工具
│   ├── track1.py                # 基准池加载、答案归一化、统一曲线/帕累托评测
│   ├── track1_phase_a.py        # Track 1 摸底筛选与难度分层
│   ├── track1_phase_c.py        # 同预算 5 基线（B1-SC / B2 / B3 / B4 / RiCL_v0.2）并发调度与绘图
│   ├── evaluation/              # 确定性 closed-answer-v1 答案核对与盲审分类
│   └── datasets/                # 固定 revision 快照校验与防泄漏分层抽样
├── scripts/
│   └── run_track1_exp35.py      # 当前主线 CLI（phase-c-b1sc / phase-c / 曲线重算入口）
├── results/
│   └── track2-frozen200/        # ✅ 已归档 Track 2 冻结 200 题基准 + 双模型最终结果、图表与审计快照
│       ├── dataset/             # 200 题冻结题面、答案键与 SHA-256 manifest
│       ├── deepseek-v41-flash/  # DeepSeek-V4.1-Flash 最终 summary/records/curves/PNG + pre_provider_retry 快照
│       └── qwen3-14b/           # Qwen3-14B 最终 summary/records/curves/PNG + pass1/pass2 快照
├── docs/                        # 按实验阶段组织的文档、报告与历史归档（见 docs/README.md）
├── tests/                       # 离线单元测试与评测回归（uv run pytest）
├── data/                        # (gitignored) 原始本地数据集快照与开发集
└── artifacts/                   # (gitignored) 原始逐题运行轨迹（runs/*/trace.jsonl 等大产物）
```

---

## 2. 实验推进顺序与各阶段作用

1. **Stage 0 · Motivation Study（前导实证诊断）**：
   - **MS1（主证据）**：ProcessBench `20` 条含错轨迹双盲依赖分析，证明错误传播（`61/103`）与独立可复用命题（`33/131`）并存。
   - **MS2-A（辅助动机）**：证明普通前缀打分器仍给 `5/32 (15.6%)` 含明确错误的轨迹打高分。
   - *历史探索止损归档*：`MS2-B / MS3-A / MS3-B`（早期 `max_tokens=2000` 空正文与跨代修复天花板）保留为工程与方法论教训，不直接进入论文正文。详情见 [docs/motivation-study-report.md](docs/motivation-study-report.md)。
2. **Stage 1 · Track 1 Dev-30（探索与缺陷收敛 · `#35`）**：
   - 切换至同模型 On-Policy 竞赛难题，完成 5 基线同总预算框架、双模型运行器与答案评估兼容修复（PR `#37–#39`）。
3. **Stage 2 · Track 2 Frozen-200（双模型冻结验证 · `#35` / PR `#40`）**：
   - 在隔离 Track 1 的 `200` 题冻结集上完成 `DeepSeek-V4.1-Flash` 与 `Qwen/Qwen3-14B` 全量 `B1-SC / B2 / B3 / B4 / RiCL_v0.2` 对比（`0` Phase C `provider_error`）。
   - 完整数据与结论：[docs/track2-frozen200-report.md](docs/track2-frozen200-report.md)
   - 逐题败因与 `prune_graph()` / `consensus_passed` 深度归因：[docs/track2-frozen200-attribution.md](docs/track2-frozen200-attribution.md)
4. **Stage 3 · 下一阶段（规划中）**：
   - 高难非饱和硬题主战集（BeyondAIME / AIME / Omni-MATH ≥ 6）+ 顶会正式收录 SOTA（`StepCo` / `REBASE` / `Thought Rollback` / `FoT`）与原版 `RSE` + 必跑消融 `RiCL w/o Dependency Graph` + 同检查点 CCL 离线因果归因。

---

## 3. 快速开始（当前主线 CLI）

```powershell
# 1. 环境同步与离线回归测试
uv sync --frozen
$env:OPENBLAS_NUM_THREADS = "1"; $env:OMP_NUM_THREADS = "1"
uv run pytest
uv run ruff check .

# 2. 查看 Track 1/2 主实验编排入口
uv run python scripts/run_track1_exp35.py --help
```

运行真实 Provider 前需在环境中设置 `API_KEY`（严禁写入仓库），并通过 `--base-url` 与 `--model` 显式指定端点与模型（示例参数与冻结 SHA 见 [docs/track2-frozen200-report.md](docs/track2-frozen200-report.md)）。

*(历史单样例 `ms2` / `ms3` / `prepare` CLI 仍完整保留以支持离线复现，操作说明见 [docs/dataset-manifests.md](docs/dataset-manifests.md) 与 [docs/README.md](docs/README.md)。)*
