# Track 2 Frozen-200 深度归因与机制诊断报告

本报告基于已冻结归档的 [`results/track2-frozen200/`](../results/track2-frozen200/)（`0` Phase C `provider_error`，`Challenge-Cup-2026` 修订 `fe5f219a97b3464e0d4e2da52c588df02d4df30c`，`RiCL_experiment` 执行修订 `43bbc28d40f57de337b5a4d9932972359ab33bd0`），对 **“为什么全预算 `49,152` tokens 下 `RiCL_v0.2` 仍微负 `B1-SC (N=3)` 及在 Qwen `Omni-MATH` 上微负 `B2`”** 进行逐题与代码级深度归因，并对外部机制审视意见做实证核验。

---

## 1. 200 题逐路径拆解：当前收益真正来自哪里？

统计全部 `200` 题在 `RiCL_v0.2` 中的最终执行路径与正确率：

### 1.1 `deepseek-ai/DeepSeek-V4.1-Flash`（总计 `175/200 = 87.5%`）

| 执行路径分类 | 正确 / 总数 | 准确率 | 占全部正确题比例 | 失败构成 |
| :--- | :---: | :---: | :---: | :--- |
| **`A · consensus_passed`**（双路答案一致早停） | **`165 / 170`** | `97.1%` | **`94.3%`** (`165/175`) | `5` `math_wrong`（假阳性 `2.9%`） |
| **`B · pruned_repair_only`**（仅触发剪枝修复，未切路） | `4 / 13` | `30.8%` | `2.3%` | `7` `math_wrong`, `1` `blind_review`, `1` `no_evaluable_output` |
| **`C · salvage_switch`**（触发过切路打捞） | `3 / 12` | `25.0%` | `1.7%` | `5` `math_wrong`, `2` `blind_review`, `2` `no_evaluable_output` |
| **`D · other (review_passed)`** | `3 / 5` | `60.0%` | `1.7%` | `1` `math_wrong`, `1` `blind_review` |

### 1.2 `Qwen/Qwen3-14B`（总计 `132/200 = 66.0%`，`32k` 峰值 `133/200 = 66.5%`）

| 执行路径分类 | 正确 / 总数 | 准确率 | 占全部正确题比例 | 失败构成 |
| :--- | :---: | :---: | :---: | :--- |
| **`A · consensus_passed`**（双路答案一致早停） | **`118 / 131`** | `90.1%` | **`89.4%`** (`118/132`) | `13` `math_wrong`（**假阳性 `9.9%`**） |
| **`B · pruned_repair_only`**（仅触发剪枝修复，未切路） | **`5 / 40`** | **`12.5%`** | `3.8%` | `19` `math_wrong`, `7` `technical_truncation`, `5` `blind_review`, `4` `no_evaluable_output` |
| **`C · salvage_switch`**（触发过切路打捞） | `7 / 25` | **`28.0%`** | `5.3%` | `12` `math_wrong`, `3` `technical_truncation`, `2` `no_evaluable_output`, `1` `blind_review` |
| **`D · other (timeout / token_limit)`** | `2 / 4` | `50.0%` | `1.5%` | `2` `math_wrong` |

**核心发现**：
1. 两个模型上 **`89%–94%` 的正确题来自 `consensus_passed`（`initial + scout_route` 双路答案一致早停）**，这正是 `RiCL_v0.2` 在 `2k/4k/8k` 低预算和 API 调用数上领跑的主因。
2. 一旦进入不共识的真难题（DeepSeek `30` 题、Qwen `69` 题），**`pruned_repair_only` 的转化率极低**——尤其在 `Qwen3-14B` 上仅 `5/40 (12.5%)`，显著低于 `salvage_switch` 的 `7/25 (28.0%)`。

---

## 2. 逐题败因剖析：为什么全预算（`49,152`）仍微负 `B1-SC (N=3)` 与 `B2`？

### 2.1 `DeepSeek-V4.1-Flash`（`RiCL` vs `B1-SC`：`8 胜 10 负`，净 `-2`）
`10` 场败例分布：
- **`request_limit` 耗尽（`6/10`）**：其中 `4` 题（`omnimath__test_01326`, `01400`, `01476`, `02522`）动作序列全部卡在 `['initial', 'scout_route', 'pruned_repair', 'pruned_repair']`，因每次求解后必须消耗 1 次独立 `review_and_extract` API 调用，2 次求解 + 2 次审核 + 2 次修复 + 2 次审核直接撞满 `request_limit=8`，**根本来不及触发 `salvage_switch`**。
- **`blind_review` 不可自动判分（`4/10`）**：`olympiadbench-580`, `omnimath__00398`, `00960`, `01451`（其中 2 例同时是 `request_limit`）。
- **`consensus_passed` 假阳性（`1/10`）**：`omnimath__test_01606`。
- **`protocol_failed`（`1/10`）**：`omnimath__test_00646`。

### 2.2 `Qwen/Qwen3-14B`（`RiCL` vs `B1-SC`：`8 胜 9 负`，净 `-1`；vs `B2`：`9 胜 10 负`，净 `-1`）
- **输给 `B1-SC` 的 `9` 题**：`5/9` 死在 `token_limit`（`32k–40k` tokens），其中 `math500__precalculus_902`、`omnimath__00527`、`01476`、`01971` 全卡在 `pruned_repair` 从未切路且 **`cited=0`**；另有 `2/9` 死于 `consensus_passed` 假阳性（`omnimath__00972`, `01749`）。
- **输给 `B2` 的 `10` 题**：`8/10` 卡在 `pruned_repair` 并撞到 `token_limit` 或 `request_limit`，其中 `7/10` 题 `cited=0`（没有成功复用任何引理）。
- **核心算力结构差**：`B1-SC (N=3)` 3 次调用 **100% 用于独立求解生成**（0 审核/抽图消耗），所以在 `49k` 全预算下 3 次完整探索占优；而 `RiCL_v0.2` 在不共识难题上将约 `40%–50%` 的请求配额和 Token 消耗在了 **L2 Critic 审核 + 低利用率抽图 + 低转化率原地 `pruned_repair`** 上。

---

## 3. 代码级真因核验：`prune_graph()` 锁死了首错后非连续引理打捞

核查运行时源码 `ricl_runtime/track1_verification.py`（`fe5f219`，第 `489–512` 行 `prune_graph()`）：

```python
if node.check == "pass" and (
    node.step <= reviewed_prefix or not node.premises or (first_error > 0 and node.step > first_error)
):
    evidence = L1_EVIDENCE
elif node.matches_source and node.step <= reviewed_prefix and (first_error == 0 or node.step < first_error):
    evidence = L2_EVIDENCE
else:
    continue
```

在 200 题 Track 2 实测轨迹中量化验证该约束的实际影响：

| 模型 | 首错后抽取节点总数 | 首错后**未被依赖污染**节点 | 因 `L2` 硬锁 `step < first_error` **被拦截拒入池** | 实际准入首错后节点（仅靠 `L1`） | 全池唯一准入引理 (`L1` / `L2`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`DeepSeek-V4.1-Flash`** | `148` | `53` | **`52 / 53` (`98.1%`)** | `1` | `101`（`2 L1` + `99 L2`） |
| **`Qwen/Qwen3-14B`** | `316` | `173` | **`168 / 173` (`97.1%`)** | `5` | `514`（`8 L1` + `506 L2`） |

**结论**：
1. 由于 `L1` 可执行验证覆盖率不足 `2%`（`2/101` 与 `8/514`），同时 `L2` 准入硬性要求 `node.step < first_error`，导致 **97%–98% 的首错后独立无污染节点根本无法进入共享池**。
2. 这意味着当前 `RiCL_v0.2` 在绝大多数难题上实际退化为“首错前前缀引理池”，最具方法差异化的**首错后非连续独立引理打捞**尚未真正发力。

---

## 4. 对三项核心机制批评的批判性接纳与后续改造清单

| 审视问题 | 200 题实证核验结论 | 批判性接纳与具体行动 |
| :--- | :--- | :--- |
| **问题 1：首错后独立引理几乎未准入** | ✅ **100% 坐实**（`52/53` 与 `168/173` 被 `prune_graph()` 拦截） | 不能直接放宽 `step > first_error`（否则引入未审查假引理）；正确解法是 **双管齐下**：① 升级脚本化 Python/SymPy `L1` 核查把覆盖率拉高到 `35%+`；② 对首错后无失效前提依赖的非计算命题做**隔离上下文单节点 L2 复核**（屏蔽错轨上下文，只看命题 + 有效前提）。 |
| **问题 2：收益可能主要来自双路共识而非依赖图** | ✅ **一针见血**（`consensus_passed` 贡献 `89%–94%` 正确题，且 `B2/B3/B4` 无 `scout_route`） | **必加核心消融 `RiCL w/o Dependency Graph`**（完整保留 `initial + scout_route + consensus_passed + 相同调度/验证器/预算`，仅移除依赖图与引理打捞）；同时必须修好 `pruned_repair` 烧光预算不切路的问题（限制原地修 1 次、提早触发 `salvage_switch`）并对 `consensus_passed` 加反伪共识核查。 |
| **问题 3：需对比原版 `RSE` 与顶会正式收录 SOTA** | ✅ **完全认同**（`WangXinglin/RSE` 未中 ICML 2026 [5/4/4/3]，不能仅靠自研 `RSE-style` 单扛） | **双层基线矩阵**：① 保留原版 `WangXinglin/RSE` + 统一 Harness `RSE + Same Verifier`；② 引入已获顶会正式收录的训练无关/测试时强 SOTA：**`StepCo` (ACL 2025)**、**`REBASE` (ICLR 2025)**、**`Thought Rollback` (ICML 2024)**、**`Forest-of-Thought` (AAAI 2025)**（及组件级依赖定位参考 `PARC` ICML 2025）。 |
