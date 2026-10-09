# 实验协议

## 数据划分

`configs/train_scenarios.json`、`configs/validation_scenarios.json` 和 `configs/test_scenarios.json` 使用不重叠的随机种子。智能体只能查看和优化验证集；最终测试集在方案冻结后运行一次。

## 指标

- 绝对定位误差：估计坐标到真实坐标的欧氏距离；
- CEP50、CEP90：误差的第 50/90 百分位；
- 大于 30m 的失败率；
- 求解器收敛率与平均运行时间；
- 移动/不稳定帮助设备识别：precision、recall、F1。

## 智能体策略

LangGraph 状态图：

```text
inspect_results → propose_trial → validate_proposal
→ run_allowed_experiment → evaluate_decision → persist_iteration
```

单轮只允许改变一个白名单参数：鲁棒损失、尺度、可靠性阈值、时间稳定性权重或最大 IRLS 迭代数。候选方案只有在验证 CEP90 改善且中位误差没有超过预注册退化阈值时才被接受。

## 复现

```bash
.venv/bin/python scripts/run_baseline.py
.venv/bin/python scripts/run_agent.py --policy scripted --reset
.venv/bin/python scripts/run_final_test.py
.venv/bin/python scripts/build_report.py
```

`scripted` 是无外部凭据的确定性回退模式。`--policy api` 使用 Anthropic API；只有保留了真实 API 运行日志时，报告才应称该轮为 LLM 驱动。
