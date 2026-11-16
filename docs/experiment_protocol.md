# 中期实验协议与证据解释

本协议针对二维静态目标的合成 BLE 众包定位。目标在每个 trial 中固定，帮助设备漂移、重复测距变化和 NLOS 是观测污染。求解器只接收报告坐标、转换距离及允许的元数据，不接收目标真值和污染标签。`mobile_scores` 表示帮助设备距离不稳定度，旧字段 `mobile_detection` 不是移动目标检测指标。本实验不作为 Huawei 真实生产 KPI、移动目标能力或新算法的证据。

## 本次实际数据身份

当前演示绑定 `artifacts/midterm/deepseek-midterm-20261116/`，以 `summary.json`、`iterations/agent_iterations.jsonl` 与引用的 `experiments/*.json` 交叉核对。旧 train/validation/test 和 2026-10-09 的结果只保留作历史材料；新中期运行使用 `midterm_validation` 与 `midterm_holdout`，不能混入旧 test 成绩。

| 场景 | 验证 seed / trials | holdout seed / trials | 主要条件 |
| --- | --- | --- | --- |
| clean_15 | 41001 / 48 | 51001 / 64 | 15 个帮助设备，环绕，常规噪声，无污染 |
| nlos_10 | 41002 / 48 | 51002 / 64 | 10 个，环绕，outlier_rate=0.25 |
| helper_drift_15 | 41003 / 48 | 51003 / 64 | 15 个，环绕，outlier=0.15，helper drift=0.25 |
| clustered_20 | 41004 / 48 | 51004 / 64 | 20 个，聚簇，outlier=0.20，helper drift=0.15 |
| collinear_15 | 41005 / 48 | 51005 / 64 | 15 个，共线，outlier=0.20，helper drift=0.10 |
| calibration_15 | 41006 / 48 | 51006 / 64 | 15 个，环绕，outlier=0.10，helper drift=0.10，偏差 2dBm |

每个 split 各含 6 个场景；验证共 288 trials，holdout 共 384 trials。完整配置还包括噪声、正偏误差、漂移幅度、时间窗口、位置噪声和 `metadata_mode`，以实验 JSON 为准，不以上表摘要替代。当前两组 seed 不重叠，但 holdout 仍是相同六类合成场景的新随机实例，不是独立的新场景分布。每个方法在同 split 使用相同场景完整配置和 `trial_keys` 顺序，配对身份需同时比较完整数据条件，而非只比较 trial 数或字符串中的 seed。

`observable` 的精度、年龄代理来自与 outlier/mobile 标签独立的随机流，仍非真实质量观测。`oracle` 包含污染信息，不能用于 Agent 调参。`clean` 切片无异常、帮助设备移动和校准偏差，仍有普通测距与位置噪声；clean 场景按包括 seed、metadata、噪声在内的完整身份比较。

## 新 run 的登记、恢复与既有证据边界

新 run 应在任何实验前落盘完整展开的验证/holdout 场景配置 snapshot、数据摘要和生成器源码 hash，并将完整配置身份作为后续恢复的校验条件。只有 seed 和试验数不能识别噪声、元数据或生成逻辑的变化。改变数据条件、生成器或策略应开新 run，而不是沿用旧 run ID。

新运行器以不可变决策文件及 JSONL 镜像重建 summary，恢复写入落后的已采纳决策，并拒绝重复轮次和审计尾部丢失。模型响应在提案校验前独立保存；若发生硬中断，只留下响应而未形成完整决策，运行器会停止并要求人工核对或新建 run，不自动重复调用。JSONL 半行损坏同样明确阻断。冻结后恢复仅补缺失产物（missing-only）；已有实验不覆盖、不重跑，最终配置与冻结时间不改变，不再执行候选调参。执行或评估异常记录安全错误类型，不保存可能含密钥的异常原文。Linux/macOS 上同一 run 使用进程锁防止并行写入；Windows 入口尚未验证。自动测试覆盖上述正常恢复与受控故障，但不能据此认证任意旧 run 已具备全部保障。

本次实际 run 原始 summary 记录了 split 的 seed、样本数、选择规则与冻结时间，但没有完整的事前 snapshot + generator hash 登记链。实验 JSON 可以用于事后核验完整场景和配对身份。为既有 run 后补 snapshot 或 hash 必须标明“事后核验”，不能宣称该文件在实验前存在，也不能将当前源码 hash 自动认作历史运行时的 hash。summary 中关于预先冻结 seed 的文字不是完整事前登记的证明。

冻结时间为 `2026-11-16T04:33:07.518450+00:00`，本次七个 holdout 实验的记录时间均在其后。这个时间顺序与最终配置可以交叉核对，但文件自身的时间记录不是外部不可篡改认证。已公开的 holdout 不再用于本轮选择；如未来修改算法，须新建协议和未查看的 holdout。

## Agent 与选择规则

LangGraph 流程为读取验证结果、生成单一提案、严格 schema 和白名单校验、受限实验执行、确定性规则接受/拒绝、持久化日志。模型仅提出一个合法字段改动，不能执行 shell、任意代码或读取 holdout 真值；候选执行器只允许验证 split。无变化、重复完整配置、无效参数及不适用的方法参数均应阻断。loss_scale_m 同时控制鲁棒损失尺度和启用时的 temporal 权重尺度；平方损失且关闭 temporal 时该参数不生效。reliability_threshold 与 max_irls_iterations 只允许用于 adaptive。

接受规则是验证 CEP90 严格降低，同时总体中位误差不超过当前的 110%，>30m 失败率不增加，数值收敛率下降不超过 5 个百分点，clean 切片身份一致且 CEP90 恶化不超过 `max(当前 clean CEP90 的 10%, 0.25m)`。这是预设的工程选择门槛，不是显著性检验。Huber、Cauchy 等模型的理由是待测假设，规则采纳也不能证明机制解释或 LLM 搜索优势。

描述性消融为 NLS、weighted NLS、Huber、Cauchy、adaptive 和 adaptive+temporal。它们供报告比较，最终方案仅沿 Agent 已采纳候选更新，不在消融表或 holdout 中事后挑最优。最终 selected 为 Huber+temporal，而不是 adaptive IRLS；Huber/Cauchy 本身也不能笼统称为交替可靠性 IRLS。

| 轮次 | 实际候选 | 验证 CEP90 当前 → 候选（m） | 结果 / 模型响应数 |
| --- | --- | --- | --- |
| 1 | name=huber | 23.6614 → 12.5395 | 采纳 / 1 |
| 2 | loss_scale_m=6 | 12.5395 → 12.8103 | 拒绝 / 1 |
| 3 | name=cauchy | 12.5395 → 13.1363 | 拒绝 / 1 |
| 4 | use_temporal_weight=true | 12.5395 → 12.4578 | 先一次无效再合法候选，采纳 / 2 |
| 5 | 无有效 proposal；3 次 max_irls_iterations=14 尝试 | 无候选评估 | 3 个不适用于当前 Huber 的无效提案被阻断 / 3 |

因此，实际证据是 **4 轮真实候选评估、5 条迭代记账、8 次真实模型响应、1 条错误轮次**。第 5 轮没有候选实验 ID，不写成第五次性能实验。原始日志保留模型、response_id、prompt_version、usage、耗时、原始响应和校验尝试；报告仅节录必要字段，模型成功返回与提案合法是两回事。`api`、结果驱动本地 `diagnostic`、固定计划 `scripted` 不能混称为 LLM。请求失败或有界校验重试耗尽应记录并阻断，不暗中使用脚本回退。

## 指标、holdout 与不确定性

绝对误差是估计坐标与目标真值的欧氏距离。CEP50 是误差中位数，CEP90 是第 90 百分位，失败率为误差严格大于 30m 的比例。>30m 是研究阈值，不是官方验收。求解器 success 只表示数值收敛，不代表正确定位；`mean_runtime_ms` 只包含求解计时，不含 API、仿真、图编排和报告生成。模型耗时单独见 trace。启发式 `confidence_radius_m` 不是经过覆盖率校准的置信区域。

冻结后对七个预列方法做 holdout 比较。selected 和 NLS 的 CEP90 分别是 11.6754m 与 19.4654m，中位误差分别是 3.0728m 与 5.8999m，>30m 失败率分别是 3.9063%（15/384）与 5.2083%（20/384）。`holdout-huber` 是无 temporal 的描述性 Huber，CEP90=12.2526m，不可冒充 selected 的结果。

bootstrap 对同场景同 trial 的两方法误差使用相同重采样索引，共 384 对，seed=91001，repeats=1000，取差值分布的 2.5/97.5 百分位。比较方向为 **candidate−NLS**，负值表示候选更低。这是普通 paired percentile bootstrap，仅针对固定场景混合内的 trial 不确定性，不是按场景分层或层级 bootstrap，也不支持工业或新场景泛化。

| selected−NLS 统计量 | 点估计差 | 95% 百分位区间 | 解释 |
| --- | ---: | --- | --- |
| CEP90（m） | −7.7900 | [−12.6906, −5.0744] | 固定混合内区间在零以下 |
| 中位误差（m） | −2.8271 | [−3.6059, −2.3201] | 同上，不等于真实环境保证 |
| >30m 失败率（原始比例） | −0.013021 | [−0.031250, 0.005208] | 跨零，不能确认失败率收益 |

失败率差若转为百分点，应同时将点估计与区间乘 100；不能把 −0.013021 说成米或相对下降百分比。CI 不是对模型理由的证明；多轮验证集选择仍有过拟合风险。没有等预算对照，不宣称 LLM 优于其他搜索。

## 操作入口

安装需 Python >=3.11。推荐 `uv`，也可在已有环境执行 pip editable 安装；详情和 DeepSeek `.env.local` 模板见 README。模板不含真实密钥，报告不读取凭据。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
# 若已激活自己的 Python 环境：
python -m pip install -e '.[dev]'
```

以下是新 run 命令示例，`api` 会请求外部模型，本次文档与报告核验不执行它。使用新的 ID，累计轮数预算到 5；冻结不得绕过错误或继续调参。

```bash
.venv/bin/python scripts/run_midterm.py --run-id my-new-midterm --policy api --iterations 5
.venv/bin/python scripts/run_midterm.py --run-id my-new-midterm --policy api --iterations 5 --finalize
```

以下命令只从实际 run 重建 PDF。报告生成不重算或补造缺失实验，正式报告应在运行器 summary 已落盘后生成，并与引用的源 JSON 核对。单元测试是独立动作，不能由报告渲染推断已通过。

```bash
.venv/bin/python scripts/build_midterm_report.py --run-id deepseek-midterm-20261116
.venv/bin/python -m pytest
```

输出为 `artifacts/midterm/deepseek-midterm-20261116/report/midterm_report_zh.pdf`。旧 `run_baseline`、`run_agent`、`run_final_test`、`build_report` 入口已弃用为本次正式中期流程；保留历史脚本、实验证据和 `integration-setup`，不删除失败、不编辑原始实验证据、不推送密钥。
