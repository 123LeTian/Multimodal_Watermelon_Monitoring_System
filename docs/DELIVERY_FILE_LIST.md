# 交接文件清单

## 推荐完整交接

直接复制整个项目目录：

```text
D:\Project\watermelon_monitoring
```

这样最稳妥，队友能看到代码、数据、模型、实验记录和文档。

## 最小必要交接清单

如果不能完整复制，至少交接以下内容。

### 代码

```text
src/
scripts/
configs/
requirements.txt
README.md
```

### 文档

```text
docs/HANDOFF_GUIDE.md
docs/REAL_DATA_FINETUNE_GUIDE.md
docs/TRAINING_GUIDE.md
docs/PROJECT_STRUCTURE.md
docs/数据交接规范表.md
experiments/final_experiment_summary.md
```

### 当前数据集

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
dataset_fused/split_source_aware_report.json
dataset_fused/README.md
```

### 当前推荐模型

```text
checkpoints/final_weak_supervised_baseline/best_model.pth
checkpoints/final_weak_supervised_baseline/last_model.pth
checkpoints/final_weak_supervised_baseline/effective_config.yaml
```

### 当前推荐模型评估结果

```text
outputs/final_weak_supervised_baseline/evaluation_report.json
outputs/final_weak_supervised_baseline/predictions.csv
outputs/final_weak_supervised_baseline/confusion_matrix.csv
```

### 当前推荐模型实验记录

```text
experiments/final_weak_supervised_baseline/experiment_summary.md
experiments/final_weak_supervised_baseline/train_command.txt
experiments/final_weak_supervised_baseline/train.log
```

## 可选交接内容

这些内容不是运行当前模型必须的，但有助于追溯实验过程：

```text
experiments/real_data_exp001/
experiments/exp002_image_only/
experiments/exp003_env_only/
experiments/exp004_multimodal_retry/
experiments/exp005_source_holdout/
experiments/exp006_label_audit/
experiments/exp007_masked_multitask_training/
outputs/real_data_exp001/
outputs/exp002_image_only/
outputs/exp003_env_only/
outputs/exp004_multimodal_retry/
outputs/exp005_source_holdout/
outputs/exp007_masked_multitask_training/
```

## 不建议作为正式结果交接

这些是临时 smoke test 目录，只用于检查流程能不能跑：

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

可以保留在本机，但不要在汇报中当正式实验结果。

## 交接前检查

交给队友前建议确认：

```powershell
python -m py_compile scripts\train.py scripts\evaluate.py scripts\infer.py src\datasets\watermelon_dataset.py src\models\multimodal_model.py src\training\losses.py
python scripts\evaluate.py --dataset-dir dataset_fused --labels-file dataset_fused\labels_masked.csv --split-file dataset_fused\split_source_aware.csv --checkpoint checkpoints\final_weak_supervised_baseline\best_model.pth --output-dir outputs\handoff_check
```

如果第二条命令能生成 `outputs/handoff_check/evaluation_report.json`，说明模型、数据和代码基本可用。
