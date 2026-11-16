# 面向 BLE 众包静态定位的实验智能体

本项目是应用数学课程的中期研究原型，选取《HarmonyOS难题发布-第七期》难题 1 中的众包离线位置聚合子问题：从帮助设备报告位置和 BLE RSSI 估计一个二维静态目标的位置。实验模拟测距噪声、NLOS 正偏误差、帮助设备位置漂移、退化几何和校准偏差，比较 NLS、加权 NLS、Huber、Cauchy 和自适应可靠性加权。目标在每个试验中不运动，帮助设备不稳定度不是移动目标识别。

LangGraph 编排“读取验证结果 → 单一提案 → 白名单校验 → 确定性实验 → 规则接受/拒绝 → 保存日志”。模型负责提出配置改动，Python 工具负责求解和评分；模型理由只是待检验假设，不是机制证明。当前工作是既有鲁棒方法与受限实验闭环的组合，不宣称发明新定位算法，也不宣称 LLM 优于等预算网格或随机搜索。

## 已有实际中期运行

证据目录是 `artifacts/midterm/deepseek-midterm-20261116/`。其 `summary.json` 记录状态 `completed`、5 条迭代记账记录、4 轮真实 API 候选评估和 1 轮错误。第 5 轮连续 3 个无效提案被校验器阻断，没有执行候选实验；第 4 轮也有一次无效提案后重试。逐轮日志共保留 **8 次真实 DeepSeek 模型响应**，不能称作 8 轮实验，也不能把第 5 轮称作第五次有效评估。

最终选择为 **Huber + temporal weight**：`name=huber`、`loss_scale_m=8.0`、`reliability_threshold=0.2`、`max_irls_iterations=10`、`use_temporal_weight=true`。它来自第 1、4 轮被规则采纳的候选；描述性消融表不是另一条隐藏的选参路径。第 2 轮 scale=6 和第 3 轮 Cauchy 都因验证 CEP90 未改善而拒绝。

| 实际比较 | NLS | selected：Huber + temporal |
| --- | ---: | ---: |
| 验证试验数 | 288 | 288 |
| 验证 CEP90（m） | 23.6614 | 12.4578 |
| holdout 试验数 | 384 | 384 |
| holdout CEP90（m） | 19.4654 | 11.6754 |
| holdout 中位误差（m） | 5.8999 | 3.0728 |
| holdout >30m 失败率 | 5.2083% | 3.9063% |

holdout 的 selected−NLS CEP90 差为 −7.7900m，95% 配对百分位 bootstrap 区间为 [−12.6906, −5.0744]m。失败率差为 −0.013021（比例，即 −1.3021 个百分点），区间 [−0.031250, 0.005208] **跨零**，不能宣称失败率改善已被区间证据确认。bootstrap 使用 seed=91001、1000 次同索引配对重采样，仅描述固定合成场景混合内的不确定性，不覆盖新场景分布或真实工业环境。

已有 run 的场景、seed、trial_keys、原始误差和冻结后时间顺序可以事后核对，但原始 summary 并非完整的事前配置登记。以后为该 run 补齐 snapshot 或 hash 只能称为**事后核验**，不能追溯宣称事前登记；新 run 应在实验前保存完整配置 snapshot 和生成器 hash。具体边界见 [研究范围](docs/problem_scope.md) 与 [实验协议](docs/experiment_protocol.md)。2026-10-09 的旧实验和 `integration-setup` 保留为历史证据，不与本次成绩混用。

## 安装与环境

需要 **Python >=3.11**，推荐使用 `uv`。在仓库根目录执行以下命令；3.11 可替换为更高的兼容 Python 版本，不要求 3.14。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

已有 Python 环境也可使用标准 pip 的 editable 安装。

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

脚本使用正常的 `ble_agent` 包导入，需先安装项目，不通过临时修改 `sys.path` 绕过安装。依赖包括 NumPy、SciPy、LangGraph、模型 SDK、Matplotlib 和 ReportLab；`dev` 增加 pytest。安装可能需要网络，而**从已有产物重建报告不需要联网或模型凭据**。

## DeepSeek 配置与新 run

`api` 模式通过 OpenAI 兼容 SDK 访问 DeepSeek。需要时手工在项目根目录创建 `.env.local`，以下仅为模板，不能把真实 key 写入版本库、报告或提交包。

```dotenv
AGENT_PROVIDER=openai
AGENT_MODEL=deepseek-chat
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=替换为自己的密钥
AGENT_TIMEOUT_SECONDS=90
```

只有 API 提案路径显式加载项目的 `.env.local`，已有环境变量优先。报告命令不加载该文件。`diagnostic` 是结果驱动的本地规则，`scripted` 是固定计划，两者可以无凭据运行，但都不应标成真实 LLM 推理。即便 temperature=0，重新请求外部模型也不保证响应逐字相同；因此真实响应的 trace 是本次运行证据，不能以重新调用替代。

以下命令展示新运行入口，执行 `api` 会联网并消耗模型额度。必须使用新的 run ID，不能覆盖已经发布 holdout 的实际 run。

```bash
.venv/bin/python scripts/run_midterm.py --run-id my-new-midterm --policy api --iterations 5
.venv/bin/python scripts/run_midterm.py --run-id my-new-midterm --policy api --iterations 5 --finalize
```

`--iterations` 是该 run 的累计轮数预算，不是每次追加轮数；`--finalize` 表示冻结选择后评估 holdout。API 请求或有界重试后仍无法通过提案校验时，保存错误并停止，不暗中切换脚本冒充成功。被阻断的 run 应先核实源日志和运行器状态，不靠编辑 summary、删除日志或绕过错误获得“完成”状态。新运行器的配置身份核验、日志恢复和冻结后 missing-only 恢复要求见实验协议；冻结后不得继续调参。

## 从已有证据生成中文 PDF

下面命令只读取该 run 的 summary、迭代日志和实验 JSON，不请求模型、不重跑仿真，也不读取 `.env.local`。

```bash
.venv/bin/python scripts/build_midterm_report.py --run-id deepseek-midterm-20261116
```

输出为 `artifacts/midterm/deepseek-midterm-20261116/report/midterm_report_zh.pdf`。报告固定 8 页，覆盖范围、数学模型、Agent 协议、验证消融、逐轮记录、holdout/CI 和局限。中文优先嵌入可用 TrueType 字体；Linux 的 `DroidSansFallbackFull.ttf` 可用，CFF 类型的 Noto TTC 不应直接假定兼容 ReportLab `TTFont`。若回退为非嵌入 CID 字体，须在交付环境检查中文显示。缺失或矛盾证据应在报告中提示，不补造结果。

交付版本已经独立运行全套测试，结果为 153 passed；终端日志、源码摘要与环境版本保存于该 run 的 `verification/`。此外完成 3 轮本地 diagnostic、holdout 和冻结后重复进入不重跑的端到端检查，该检查不是 API 证据。生成 PDF 本身不等于测试通过。

```bash
.venv/bin/python -m pytest
```

中文演示稿见 [五分钟中期讲稿](docs/midterm_presentation.md) 和 [一分钟进展稿](docs/oral_progress_report.md)。讲稿中的数值绑定实际 run，换 run 时必须重新核对。

## 入口与证据结构

本次正式入口为 `scripts/run_midterm.py` 和 `scripts/build_midterm_report.py`。旧 `run_baseline.py`、`run_agent.py`、`run_final_test.py`、`build_report.py` **已弃用为本次中期流程入口**；它们及旧目录仅用于理解历史记录，不再作为复现推荐，更不能把旧 test 当新 holdout。

```text
src/ble_agent/                    # 仿真、求解、评估、Agent 图与报告生成
scripts/run_midterm.py            # 隔离 run、验证迭代、冻结与 holdout
scripts/build_midterm_report.py   # 仅从已有 run 产物生成 PDF
configs/midterm_*_scenarios.json  # 新 run 的源场景配置
artifacts/midterm/<run-id>/
  summary.json                   # 实际状态、最终配置、实验引用、CI
  iterations/agent_iterations.jsonl
  experiments/*.json             # 场景完整配置、指标、误差、trial_keys
  protocol/                      # 新运行协议快照；后补历史快照须标明核验性质
  report/midterm_report_zh.pdf
```

这是合成静态二维研究，不是真实工业 BLE 验收，也未研究移动目标、楼层地图、真实设备异构或目标轨迹。真实部署需在授权、脱敏观测上重新标定并按官方协议复测。保留历史与失败证据，不上传密钥和个人位置，不为展示成绩删除不利结果。
