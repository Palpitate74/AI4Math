# 面向 BLE 众包离线设备定位的鲁棒实验智能体

> 课程作业：开发一个智能体，并将其用于应用数学问题的研究与迭代实验。

本项目选择《HarmonyOS难题发布-第七期》难题 1 的一个可复现子问题：从有噪声、存在 NLOS 异常和帮助设备位置漂移的 BLE 众包观测中，估计二维离线设备位置。

## 核心思路

- **数学问题**：普通非线性最小二乘、多种鲁棒损失、迭代重加权最小二乘（IRLS）与可靠性加权。
- **智能体**：由 LangGraph 编排 `诊断 → 单一提案 → 白名单校验 → 确定性实验 → 接受/拒绝 → 日志保存`。
- **数据与评估**：固定随机种子的合成 BLE 场景，严格分离 train / validation / test，采用 CEP50、CEP90、失败率和移动帮助设备检测 F1。
- **诚实边界**：这是仿真研究原型，不宣称达到华为真实生产环境的 CEP90 或移动检测 KPI。

完整问题边界见 [`docs/problem_scope.md`](docs/problem_scope.md)，实验协议见 [`docs/experiment_protocol.md`](docs/experiment_protocol.md)。

## 项目结构

```text
src/ble_agent/
  simulator.py       # BLE 观测、NLOS、移动与位置漂移模拟
  solvers.py         # NLS、Huber/Cauchy、adaptive IRLS
  experiment.py      # 固定划分、指标和实验产物
  graph.py           # LangGraph 实验智能体状态图
  agent.py           # LLM/脚本策略、白名单验证
  report.py          # 从真实产物生成 PDF
scripts/
  run_baseline.py
  run_agent.py
  run_final_test.py
  build_report.py
artifacts/           # 运行后的指标、日志、图表和报告
```

## 安装

需要 Python 3.11+。以下命令使用 `uv` 创建环境：

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

## 复现实验

```bash
.venv/bin/python scripts/run_baseline.py
.venv/bin/python scripts/run_agent.py --policy scripted --reset
.venv/bin/python scripts/run_final_test.py
.venv/bin/python scripts/build_report.py
.venv/bin/python -m pytest
```

报告输出为：

```text
artifacts/report/robust_ble_localization_agent_report.pdf
```

## 智能体运行模式

- `--policy scripted`：默认且完全可复现。它仍通过同一 LangGraph 状态图执行，但使用透明、固定的受限试验策略；适合没有 API 凭据的老师或同学复现。
- `--policy api`：通过官方 Anthropic Python SDK 请求一次受限的 JSON 提案。模型仅可提出一个白名单内改动；程序不暴露 shell、任意代码执行或测试集读取权限。运行前需要配置 `ANTHROPIC_API_KEY` 或 `ANTHROPIC_AUTH_TOKEN`。

不要在提交中添加 API key、真实定位数据或个人隐私数据。

## 运行产物

- `artifacts/experiments/*.json`：原始指标、场景和误差序列；
- `artifacts/iterations/agent_iterations.jsonl`：每轮诊断、提案、实验、接受/拒绝理由；
- `artifacts/agent_state.json`：冻结的验证集方案；
- `artifacts/report/*.pdf`：自动生成的作业报告。

## 局限性

RSSI 到距离的映射、室内/室外传播环境、楼层与地图约束、设备异构性和延迟上报会显著影响真实部署。真实结果应在脱敏、授权的观测数据上，依据原题指定的验证协议复测。
