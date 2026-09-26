# final_weak_supervised_baseline 模型卡

## 模型来源

- 来源实验：`exp007_masked_multitask_training`
- 推荐权重：`checkpoints/final_weak_supervised_baseline/best_model.pth`
- 模型结构：ResNet50 图像分支 + Transformer 环境分支 + 多任务预测头
- 输入模式：`multimodal`
- 训练数据：`dataset_fused`
- 标签文件：`dataset_fused/labels_masked.csv`
- 划分文件：`dataset_fused/split_source_aware.csv`

## 任务

| 任务 | 类别数 | 当前状态 |
| --- | ---: | --- |
| `growth_stage` | 6 | 当前弱监督数据中缺少可靠强监督，已 mask |
| `health_level` | 4 | 可作为弱监督基线，仍需真实大棚微调 |
| `maturity_level` | 4 | 在成熟度来源上可训练，但本 held-out test 未评估 |
| `abnormal_alert` | 2 | 可作为弱监督基线，漏报下降但误报偏多 |

## held-out test 指标

测试来源：`watermelon_disease_recognition`

| 任务 | Evaluated Samples | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| `health_level` | 1381 | 0.5373 | 0.2465 |
| `abnormal_alert` | 1381 | 0.7473 | 0.5472 |

异常预警：

- Positive precision: 0.7593
- Positive recall: 0.9606
- Positive F1: 0.8482
- False positives: 309
- False negatives: 40

## 适用范围

适合：

- 作为项目机器学习模型工程交接版本
- 作为真实大棚数据微调的起点
- 作为弱监督多模态基线模型
- 用于演示训练、评估、推理流程

不适合：

- 直接作为真实大棚生产模型
- 作为论文中最终真实大棚准确率结论
- 在没有人工审核真实数据的情况下直接做农业决策

## 主要限制

- `dataset_fused` 是公开图片数据集 + 弱监督标签 + 真实西瓜田环境数据的融合数据集，不是单一大棚人工标注数据。
- source-aware 验证表明模型存在跨来源泛化问题。
- `growth_stage` 缺少真实强监督标签。
- `abnormal_alert` 漏报少，但误报偏多。

## 后续建议

1. 收集真实大棚图片和环境数据。
2. 人工审核四个任务标签。
3. 增加 `--pretrained-checkpoint` 功能，从当前模型开始微调。
4. 在真实大棚独立测试集上重新评估。
5. 根据业务需求调节异常预警误报/漏报平衡。
