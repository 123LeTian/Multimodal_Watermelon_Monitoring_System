# V3 审核与训练结果

生成日期：2026-09-28

## 1. 健康等级复核

统一标准：

- `0`：未见明显异常
- `1`：局部轻微症状
- `2`：多个区域或明显病斑
- `3`：大面积病害、萎蔫或腐烂

对原先标为“中度异常”的 634 张图片进行 AI 辅助视觉复核：

| 复核结果 | 数量 |
| --- | ---: |
| 改为轻微异常 `1` | 131 |
| 保持中度异常 `2` | 77 |
| 改为严重异常 `3` | 273 |
| 严重程度无法可靠确认，`health_level_valid=0` | 153 |

共修改 404 个健康等级标签，153 个样本不再参与健康等级损失，但继续参与异常告警训练。

> 本次为基于图片的 AI 辅助复核，不等同于植物病理专家标注。后续正式研究结论应抽样进行人工专家复核。

## 2. 来源隔离划分

| Split | 数据来源 | 样本数 |
| --- | --- | ---: |
| Train | `watermelon_disease_recognition` | 1,381 |
| Train | `watermelon_ripe_semiripe_unripe` | 1,882 |
| Validation | `kurdistan_watermelon_disease` | 2,619 |
| Validation | `watermelon_ripe_unripe` | 1,718 |
| Test | `fruq_db` | 2,400 |

复核结果：10,000 个 `image_id` 唯一，无缺失来源，任何数据来源都没有跨 split 出现。图像和 24 小时环境窗口对齐覆盖率为 100%。

## 3. V3 训练

- 输入模式：多模态（图像 + 24 小时时序环境数据）
- 初始化权重：`checkpoints/enhanced_fused_v2/best_model.pth`
- 最多轮数：15
- 早停耐心：5
- 实际停止：第 6 轮
- 最佳轮次：第 1 轮
- 最佳综合验证分数：`0.780221`
- 最佳权重：`checkpoints/reviewed_v3/best_model.pth`
- SHA-256：`46ECD957FD0AA5D5DA9FFC960D248038223271F34C8AB5C8AB1643051C3EF684`

最佳轮次验证结果：

| 任务 | 指标 | 结果 |
| --- | --- | ---: |
| 健康等级 | Accuracy | 0.964108 |
| 健康等级 | Macro F1 | 0.723548 |
| 异常告警 | Macro F1 | 0.987200 |
| 异常告警 | 阳性召回率 | 0.994958 |
| 成熟度 | Macro F1 | 0.499316 |

验证集健康等级没有第 3 类，成熟度只有第 0、2 类，因此按全部四类计算的 Macro F1 分别受约 `0.75` 和 `0.50` 的上限约束。

## 4. 告警阈值与独立测试

在验证集上以异常阳性召回率不低于 `0.95` 为约束，校准得到告警阈值 `0.84`。

使用最佳 V3 权重和阈值 `0.84` 在完全独立的 `fruq_db` 来源上测试：

| 任务 | 指标 | 结果 |
| --- | --- | ---: |
| 健康等级 | Accuracy | 0.999167 |
| 健康等级 | Macro F1 | 0.749375 |
| 成熟度 | Accuracy | 0.999167 |
| 成熟度 | Macro F1 | 0.749375 |
| 异常告警 | Accuracy | 1.000000 |
| 异常告警 | Macro F1 | 1.000000 |
| 异常告警 | 阳性召回率 | 1.000000 |

测试集健康等级不含第 2 类，成熟度不含第 0 类，所以四分类 Macro F1 的理论上限约为 `0.75`。测试集来自通用水果质量域，类别视觉差异较明显；该结果只证明模型在当前构造数据集上的表现，不能替代真实西瓜大棚数据验收。

## 5. 主要产物

- 审核明细：`experiments/health_severity_review_v3/health_review_completed.csv`
- 审核决策：`experiments/health_severity_review_v3/review_decisions.yaml`
- 审核后标签：`dataset_fused/labels_reviewed_v3.csv`
- 来源隔离划分：`dataset_fused/split_source_isolated_v3.csv`
- 训练配置：`configs/train_reviewed_v3.yaml`
- 最佳模型：`checkpoints/reviewed_v3/best_model.pth`
- 训练日志：`experiments/reviewed_v3/train.log`
- 阈值报告：`outputs/reviewed_v3/alert_threshold.json`
- 校准后测试报告：`outputs/reviewed_v3/test_calibrated/evaluation_report.json`
- 校准后测试预测：`outputs/reviewed_v3/test_calibrated/predictions.csv`

## 6. 当前结论

本轮目标已完成：标签已复核、低置信严重度已屏蔽、来源已隔离、V3 已训练并早停、最佳权重已在独立来源上评估。下一阶段应使用真实大棚采集的西瓜图像和同步传感器序列进行盲测，并补齐各任务在验证集和测试集中的缺失类别。
