# 最终实验总结

## 当前交接结论

本项目已经完成机器学习模型工程搭建，并形成可交接的弱监督多模态基线模型。当前推荐交接版本为：

```text
final_weak_supervised_baseline
```

来源实验：

```text
exp007_masked_multitask_training
```

推荐模型文件：

```text
checkpoints/final_weak_supervised_baseline/best_model.pth
```

当前模型适合作为真实大棚数据到来后的微调起点，不应直接包装成最终生产模型。

## 模型搭建内容

模型是一个多模态、多任务分类模型：

- 图像输入：西瓜或叶片图片，经过 ResNet50 图像分支提取特征。
- 环境输入：拍摄前 24 小时内的环境序列，字段为 `temperature`、`soil_humidity`、`light`、`ph`，经过 Transformer Encoder 提取时序特征。
- 融合方式：图像特征和环境特征拼接后进入融合层。
- 输出任务：`growth_stage`、`health_level`、`maturity_level`、`abnormal_alert`。

后续新增了三种输入模式：

```text
multimodal
image_only
env_only
```

后续还新增了 per-sample task mask，使弱监督样本只在可信任务上参与 loss。

## 数据集来源

当前训练数据集是：

```text
dataset_fused
```

它由 `scripts/build_fused_watermelon_dataset.py` 生成，包含 10000 张采样图片。数据来源包括：

- `watermelon_disease_recognition`: 西瓜叶片健康/病害图片
- `kurdistan_watermelon_disease`: 西瓜叶片病害、虫害、缺素图片
- `watermelon_ripe_semiripe_unripe`: 西瓜成熟/未成熟图片
- `watermelon_ripe_unripe`: 西瓜成熟/未成熟图片
- `fruq_db`: 水果新鲜、轻微异常、腐烂质量图片
- `watermelon_environment`: 真实西瓜田天气和土壤水分数据

环境时间序列来自南意大利西瓜田数据，pH 因源数据缺失被固定为 6.8。

重要限制：`dataset_fused` 是融合弱监督数据集，不是单个大棚人工标注数据集。

## 数据文件说明

当前可交接数据集应包含：

```text
dataset_fused/images/
dataset_fused/environment.csv
dataset_fused/labels.csv
dataset_fused/labels_masked.csv
dataset_fused/split.csv
dataset_fused/split_source_aware.csv
dataset_fused/fusion_manifest.csv
dataset_fused/quality_report.json
dataset_fused/validation_report.json
dataset_fused/README.md
```

正式交接建议优先使用：

```text
labels_masked.csv
split_source_aware.csv
```

其中 `labels_masked.csv` 比 `labels.csv` 多了四个字段：

```text
growth_stage_valid
health_level_valid
maturity_level_valid
abnormal_alert_valid
```

字段含义：

```text
1 = 该样本该任务标签可信，参与 loss 或 masked evaluation
0 = 该样本该任务标签不可信，不参与 loss 或 masked evaluation
```

## 关键实验过程

### real_data_exp001

第一次完整训练，使用随机 train/val/test 划分。测试集 1500 条，结果较好：

| 任务 | Accuracy | Macro F1 |
| --- | ---: | ---: |
| 生长阶段 | 0.9780 | 0.9519 |
| 健康等级 | 0.9453 | 0.9384 |
| 成熟度 | 0.9973 | 0.9963 |
| 异常预警 | 0.9700 | 0.9696 |

结论：工程流程跑通，但随机划分可能偏乐观。

### 错误样本分析

对 `real_data_exp001` 做了错误分析：

- 测试样本数：1500
- 至少一个任务错误：112
- 样本级错误率：7.47%
- 错误最多任务：健康等级，82 次任务级错误
- 异常预警误报：25
- 异常预警漏报：20

### 消融实验 exp002-exp004

| 实验 | 输入模态 | 生长阶段 F1 | 健康等级 F1 | 成熟度 F1 | 异常预警 F1 | 错误样本 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `exp002_image_only` | 只用图片 | 0.5947 | 0.9421 | 0.9949 | 0.9615 | 390 |
| `exp003_env_only` | 只用环境 | 0.9533 | 0.4282 | 0.4375 | 0.7245 | 510 |
| `exp004_multimodal_retry` | 图片+环境 | 0.9451 | 0.9392 | 0.9963 | 0.9696 | 112 |

结论：多模态设计应该保留。环境数据对生长阶段帮助明显，图像数据对健康、成熟度和异常预警更关键。

### exp005_source_holdout

使用 source-aware split，将 `watermelon_disease_recognition` 整个来源作为测试集，检查跨来源泛化。

| 任务 | Accuracy | Macro F1 |
| --- | ---: | ---: |
| 生长阶段 | 0.8313 | 0.5574 |
| 健康等级 | 0.3773 | 0.2254 |
| 成熟度 | 1.0000 | 0.2500 |
| 异常预警 | 0.6705 | 0.5839 |

错误分析：

- 测试样本：1381
- 至少一个任务错误：924
- 样本级错误率：66.91%
- 健康等级错误：860
- 异常预警错误：455
- 异常预警误报：218
- 异常预警漏报：237

结论：随机划分结果偏乐观，模型存在数据来源风格依赖。

### exp006_label_audit

对弱监督标签做审计，结论是 `dataset_fused` 不能作为“四任务全强监督”数据集。

可信任务建议：

- 叶片病害/健康来源适合训练 `health_level` 和 `abnormal_alert`。
- 成熟/未成熟果实来源适合训练 `maturity_level`。
- `growth_stage` 当前主要来自规则、成熟度映射或环境先验，不建议作为强监督主任务。
- `fruq_db` 可作为果实质量异常补充，但建议降权或谨慎使用。

### exp007_masked_multitask_training

实现 per-sample task mask 后，在 source-aware split 上重新训练和评估。

held-out test 结果：

| 任务 | Evaluated Samples | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| 生长阶段 | 0 | 0.0000 | 0.0000 |
| 健康等级 | 1381 | 0.5373 | 0.2465 |
| 成熟度 | 0 | 0.0000 | 0.0000 |
| 异常预警 | 1381 | 0.7473 | 0.5472 |

与 exp005 对比：

- 健康等级 Accuracy: 0.3773 -> 0.5373
- 健康等级 Macro F1: 0.2254 -> 0.2465
- 异常预警 Accuracy: 0.6705 -> 0.7473
- 异常漏报: 237 -> 40
- 异常误报: 218 -> 309
- 异常预警 Macro F1: 0.5839 -> 0.5472

结论：task mask 有效，尤其降低了异常漏报并改善健康等级泛化。但异常误报增加，说明还需要阈值、class weight、source weight 或真实大棚微调进一步校准。

## 当前模型状态

当前模型已经完成：

- Dataset 读取
- ResNet50 + Transformer 多模态模型
- 多任务 loss
- masked multitask loss
- 训练脚本
- 评估脚本
- 单样本推理脚本
- 错误分析
- 消融实验
- source-aware 泛化验证
- 标签可信度审计
- 实验归档

当前仍存在：

- 训练数据不是最终真实大棚人工标注数据。
- source-aware 场景下泛化仍不够强。
- `growth_stage` 缺少真实强监督标签。
- 异常预警误报偏多。
- 最终上线前必须做真实大棚数据微调和独立测试。

## 最终推荐

交接时推荐把当前模型描述为：

```text
可训练、可评估、可推理的弱监督多模态基线模型。
```

不要描述为：

```text
已经可直接用于真实大棚生产环境的最终模型。
```

后续优先工作：

1. 收集真实大棚图片和对应环境数据。
2. 人工标注健康等级、异常预警、成熟度和生长阶段。
3. 从 `final_weak_supervised_baseline` 开始微调。
4. 在真实大棚独立测试集上验收。
5. 如有时间，再做类别权重/来源权重实验，控制异常误报。
