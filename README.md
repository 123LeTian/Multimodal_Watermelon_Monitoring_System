# 大棚西瓜多模态生长监测模型

本项目是大棚西瓜生长监测的机器学习工程代码，已经完成从数据读取、模型搭建、训练、评估、错误分析到实验归档的基础闭环。

当前可交接版本定位为：

```text
弱监督多模态基线模型 / 真实大棚数据微调起点
```

它可以继续训练、评估和单样本推理，但还不能直接称为真实大棚生产模型。最终应用仍需要真实大棚图片和人工审核标签进行微调与验收。

## 当前推荐交接模型

推荐交接目录：

```text
checkpoints/final_weak_supervised_baseline/
outputs/final_weak_supervised_baseline/
experiments/final_weak_supervised_baseline/
```

该版本来自：

```text
exp007_masked_multitask_training
```

它使用 `dataset_fused`、`labels_masked.csv` 和 `split_source_aware.csv`，支持 per-sample task mask，让弱监督样本只在可信任务上参与训练。

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

## 快速验证命令

```powershell
python -m py_compile scripts\train.py scripts\evaluate.py scripts\infer.py src\datasets\watermelon_dataset.py src\models\multimodal_model.py src\training\losses.py
```

评估当前交接模型：

```powershell
python scripts\evaluate.py --dataset-dir dataset_fused --labels-file dataset_fused\labels_masked.csv --split-file dataset_fused\split_source_aware.csv --checkpoint checkpoints\final_weak_supervised_baseline\best_model.pth --output-dir outputs\final_weak_supervised_baseline_check
```

## 重要提醒

`dataset_fused` 不是单个大棚人工标注数据集。它由公开西瓜图片数据集、弱监督融合标签和真实西瓜田环境/土壤水分时间序列组成，适合做工程演示、预训练和基线实验。论文或答辩中不要把它描述成“真实大棚人工标注数据集”。
