# 大棚西瓜多模态生长监测模型

本项目是大棚西瓜生长监测的机器学习工程代码，已经完成从数据读取、模型搭建、训练、评估、错误分析到实验归档的基础闭环。

原模型与增强数据/训练机制的融合说明见
[`docs/ENHANCED_MULTIMODAL_FUSION.md`](docs/ENHANCED_MULTIMODAL_FUSION.md)。

当前推荐版本定位为：

```text
V3 Reviewed 多模态研究基线 / 真实大棚数据微调起点
```

它支持来源隔离训练、任务标签掩码、类别权重、告警阈值校准和单样本推理，但还不能直接称为真实大棚生产模型。最终应用仍需要真实同步采集的图片、环境数据和人工审核标签进行微调与验收。

## 当前推荐交接模型

推荐模型：

```text
checkpoints/reviewed_v3/best_model.pth
```

模型权重和包含 20 组样例的完整推理包通过 GitHub Release 发布：

[V3 Reviewed Release](https://github.com/123LeTian/Multimodal_Watermelon_Monitoring_System/releases/tag/v3-reviewed)

该版本使用：

```text
dataset_fused/labels_reviewed_v3.csv
dataset_fused/split_source_isolated_v3.csv
configs/train_reviewed_v3.yaml
```

634 张原“中度异常”图片经过 AI 辅助视觉复核，其中 131 张改为轻微、77 张保持中度、273 张改为严重，153 张因严重度无法可靠确认而设置 `health_level_valid=0`。本次复核不等同于植物病理专家标注。

最佳轮次验证结果：健康等级 Macro F1 `0.7235`，异常告警 Macro F1 `0.9872`，异常阳性召回率 `0.9950`。校准后的异常告警阈值为 `0.84`。

## 项目核心能力

- 图像分支：ResNet50
- 环境数据分支：Transformer Encoder
- 输入模态：`multimodal`、`image_only`、`env_only`
- 多任务输出：
  - `growth_stage`: 生长阶段 6 类
  - `health_level`: 健康等级 4 类
  - `maturity_level`: 成熟度 4 类
  - `abnormal_alert`: 异常预警 2 类
- 支持 task mask 的多任务损失
- 支持训练、评估、错误分析、消融实验和单样本推理

## 主要目录

```text
configs/        训练配置
src/            Dataset、模型、loss、metrics 等核心代码
scripts/        训练、评估、推理、数据生成和实验分析脚本
dataset_fused/  当前交接用融合数据集
dataset_sample/ 模拟数据，只用于流程检查
checkpoints/    模型权重
outputs/        评估报告、预测结果、混淆矩阵
experiments/    实验日志、命令、总结文档
docs/           训练、交接、真实数据微调说明
```

## 交接必读文档

- `docs/HANDOFF_GUIDE.md`: 队友接手训练和评估的主说明
- `docs/REAL_DATA_FINETUNE_GUIDE.md`: 真实大棚数据接入和微调说明
- `experiments/final_experiment_summary.md`: 当前所有关键实验结论
- `docs/DELIVERY_FILE_LIST.md`: 交接文件清单
- `docs/TRAINING_GUIDE.md`: 训练命令和数据格式详细说明
- `docs/V3_MODEL_USAGE_GUIDE.md`: V3 推理输入、运行方式和输出解释
- `docs/ENHANCED_MULTIMODAL_FUSION.md`: V4 数据与训练机制的融合实现
- `experiments/reviewed_v3/V3_RESULT_SUMMARY.md`: V3 审核、训练和测试结果

## 快速验证命令

```powershell
python -m py_compile scripts\train.py scripts\evaluate.py scripts\infer.py src\datasets\watermelon_dataset.py src\models\multimodal_model.py src\training\losses.py
```

评估 V3：

```powershell
python scripts\evaluate.py --dataset-dir dataset_fused --labels-file dataset_fused\labels_reviewed_v3.csv --split-file dataset_fused\split_source_isolated_v3.csv --checkpoint checkpoints\reviewed_v3\best_model.pth --output-dir outputs\reviewed_v3_check --split test --abnormal-alert-threshold 0.84
```

## 重要提醒

`dataset_fused` 不是单个大棚人工标注数据集。它由公开西瓜图片数据集、弱监督融合标签和真实西瓜田环境/土壤水分时间序列组成，适合做工程演示、预训练和基线实验。论文或答辩中不要把它描述成“真实大棚人工标注数据集”。

V3 的环境序列并非与每张公开图片在真实大棚中同步采集，pH 训练值固定为 6.8；生长阶段任务在 V3 中没有有效标签参与训练。这些限制必须在研究报告和演示中明确说明。
