# 项目目录结构说明

## 根目录

```text
watermelon_monitoring/
├─ configs/
├─ src/
├─ scripts/
├─ dataset_fused/
├─ dataset_sample/
├─ checkpoints/
├─ outputs/
├─ experiments/
├─ docs/
├─ README.md
└─ requirements.txt
```

## `configs/`

存放训练配置文件。当前主要文件：

```text
configs/train.yaml
```

`scripts/train.py` 支持 `--config configs/train.yaml`，命令行参数会覆盖 YAML 配置。

## `src/`

核心 Python 源代码。

```text
src/datasets/   数据读取、图像预处理、环境时间窗读取、task mask 返回
src/models/     ResNet50 + Transformer 多模态模型
src/training/   多任务 loss、masked loss、分类指标
src/utils/      YAML 配置读取和保存工具
```

## `scripts/`

命令行入口脚本。

常用脚本：

```text
train.py                          训练模型
evaluate.py                       评估模型
infer.py                          单样本推理
inspect_batch.py                  检查一个 batch 是否能读取
inspect_model.py                  检查模型前向传播
inspect_loss.py                   检查 loss 和反向传播
analyze_errors.py                 错误样本分析
summarize_ablation.py             汇总消融实验
create_source_aware_split.py      生成 source-aware 划分
audit_label_task_validity.py      标签可信度审计
create_masked_labels.py           生成 labels_masked.csv
build_fused_watermelon_dataset.py 生成 dataset_fused
validate_fused_dataset.py         验证 dataset_fused
```

## `dataset_fused/`

当前交接用融合数据集。

```text
images/                    10000 张采样图片
environment.csv            环境时间序列
labels.csv                 原始融合四任务标签
labels_masked.csv          带 task mask 的标签文件，当前推荐训练使用
split.csv                  随机划分
split_source_aware.csv     source-aware 划分，当前推荐验证使用
fusion_manifest.csv        每张图片的来源和融合规则信息
quality_report.json        数据质量统计
validation_report.json     数据验证报告
README.md                  数据集说明
```

## `dataset_sample/`

模拟数据，只用于检查工程流程。不能用于论文结论或真实模型效果说明。

## `checkpoints/`

模型权重目录。

当前推荐交接模型：

```text
checkpoints/final_weak_supervised_baseline/
```

历史实验权重按实验名保存，例如：

```text
checkpoints/real_data_exp001/
checkpoints/exp005_source_holdout/
checkpoints/exp007_masked_multitask_training/
```

## `outputs/`

模型评估输出。

每个正式实验通常包含：

```text
evaluation_report.json
predictions.csv
confusion_matrix.csv
```

当前推荐交接结果：

```text
outputs/final_weak_supervised_baseline/
```

## `experiments/`

实验记录、训练命令、训练日志和实验总结。

关键文件：

```text
experiments/final_experiment_summary.md
experiments/final_weak_supervised_baseline/
```

历史实验：

```text
real_data_exp001                  第一次随机划分完整训练
exp002_image_only                 图片单模态消融
exp003_env_only                   环境单模态消融
exp004_multimodal_retry           多模态复训消融
exp005_source_holdout             source-aware 泛化实验
exp006_label_audit                标签可信度审计
exp007_masked_multitask_training  masked multitask training
```

## `docs/`

文档目录。

交接重点文档：

```text
docs/HANDOFF_GUIDE.md
docs/REAL_DATA_FINETUNE_GUIDE.md
docs/DELIVERY_FILE_LIST.md
docs/TRAINING_GUIDE.md
docs/PROJECT_STRUCTURE.md
docs/数据交接规范表.md
docs/模型需求规格表.md
```

## 临时目录说明

以下目录是 smoke test 或过程检查结果，不建议作为正式实验汇报：

```text
checkpoints/smoke_masked/
checkpoints/smoke_weighted/
checkpoints/_smoke_image_only/
checkpoints/_smoke_env_only/
checkpoints/_smoke_multimodal/
outputs/smoke_masked/
outputs/smoke_weighted/
outputs/_smoke_image_only/
outputs/_smoke_env_only/
```

这些目录可以保留，但交接时需要说明它们不是正式结果。
