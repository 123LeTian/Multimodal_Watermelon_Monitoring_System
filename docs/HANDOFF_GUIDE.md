# 模型交接说明

## 1. 交接目标

这份文档用于把当前大棚西瓜多模态机器学习模型交接给后续负责数据整理、训练和微调的同学。

当前交接内容包括：

- 可运行的 PyTorch 项目代码
- 已构建的多模态模型
- 已训练好的弱监督基线模型
- 当前融合数据集 `dataset_fused`
- 实验记录、评估结果和后续微调说明

当前模型不是最终生产模型，而是后续真实大棚数据微调的起点。

## 2. 应交接给队友的文件

建议直接交接整个项目目录：

```text
D:\Project\watermelon_monitoring
```

如果不能完整复制，至少交接这些目录和文件：

```text
configs/
src/
scripts/
docs/
dataset_fused/
checkpoints/final_weak_supervised_baseline/
outputs/final_weak_supervised_baseline/
experiments/final_weak_supervised_baseline/
experiments/final_experiment_summary.md
requirements.txt
README.md
```

`dataset_sample/` 是模拟数据，可以一起给，用于快速检查代码流程。

`data_external/` 是原始外部数据来源目录，如果体积太大，可以不交；但必须保留 `dataset_fused/`，因为当前模型训练和评估依赖它。

## 3. 当前推荐模型

当前推荐模型目录：

```text
checkpoints/final_weak_supervised_baseline/
```

关键文件：

```text
best_model.pth
last_model.pth
effective_config.yaml
```

使用建议：

- 做评估和推理时优先使用 `best_model.pth`。
- 继续训练或排查训练状态时可参考 `last_model.pth`。
- 复现实验必须参考 `effective_config.yaml`。

对应评估结果：

```text
outputs/final_weak_supervised_baseline/evaluation_report.json
outputs/final_weak_supervised_baseline/predictions.csv
outputs/final_weak_supervised_baseline/confusion_matrix.csv
```

对应实验说明：

```text
experiments/final_weak_supervised_baseline/experiment_summary.md
experiments/final_weak_supervised_baseline/train_command.txt
experiments/final_weak_supervised_baseline/train.log
```

## 4. 项目搭建了什么

模型结构：

```text
图片 -> ResNet50 图像分支
环境序列 -> Transformer Encoder 环境分支
图像特征 + 环境特征 -> 融合层 -> 四个任务预测头
```

输入数据：

- 图片：西瓜果实、叶片或相关样本图像
- 环境：拍摄前 24 小时内的时间序列

环境字段固定为：

```text
temperature
soil_humidity
light
ph
```

注意：当前项目使用的是土壤湿度 `soil_humidity`，不是空气湿度 `humidity`。

输出任务：

| 任务 | 类别数 | 说明 |
| --- | ---: | --- |
| `growth_stage` | 6 | 发芽期、伸蔓期、开花期、坐果期、膨大期、成熟期 |
| `health_level` | 4 | 正常、轻微异常、中度异常、严重异常 |
| `maturity_level` | 4 | 未成熟、接近成熟、成熟、过熟 |
| `abnormal_alert` | 2 | 无异常、有异常 |

## 5. 数据集怎么来的

当前训练数据集是：

```text
dataset_fused
```

它由脚本生成：

```text
scripts/build_fused_watermelon_dataset.py
```

数据集包含 10000 张采样图片。图片来自多个公开数据来源，环境数据来自真实西瓜田天气和土壤水分时间序列，pH 因源数据缺失被固定为 6.8。

重要说明：`dataset_fused` 是融合弱监督数据集。它可以用于工程演示、预训练和基线实验，但不能说成“单个大棚人工采集并人工标注的数据集”。

## 6. 数据格式要求

标准数据集目录应为：

```text
dataset_xxx/
├─ images/
├─ environment.csv
├─ labels.csv
├─ split.csv
└─ README.md
```

`environment.csv` 必须包含：

```text
timestamp,temperature,soil_humidity,light,ph
```

`labels.csv` 必须包含：

```text
image_id,image_path,capture_time,growth_stage,health_level,maturity_level,abnormal_alert
```

`split.csv` 必须包含：

```text
image_id,split
```

`split` 只能是 `train`、`val`、`test`。

当前弱监督训练建议使用：

```text
dataset_fused/labels_masked.csv
dataset_fused/split_source_aware.csv
```

## 7. task mask 是什么

因为 `dataset_fused` 是弱监督融合数据，不是每张图片都适合训练四个任务，所以当前使用 `labels_masked.csv`。

它比普通 `labels.csv` 多四列：

```text
growth_stage_valid
health_level_valid
maturity_level_valid
abnormal_alert_valid
```

含义：

```text
1 = 该任务标签可信，参与 loss
0 = 该任务标签不可信，不参与 loss
```

例如：

- 叶片病害数据主要训练 `health_level` 和 `abnormal_alert`。
- 成熟/未成熟果实数据主要训练 `maturity_level`。
- 当前 `growth_stage` 多来自规则推断，不作为强监督训练。

## 8. 环境安装

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

检查依赖：

```powershell
python -c "import torch, torchvision, numpy, PIL, yaml; print('dependencies=ok')"
```

检查 GPU：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## 9. 快速检查项目是否可运行

语法检查：

```powershell
python -m py_compile scripts\train.py scripts\evaluate.py scripts\infer.py src\datasets\watermelon_dataset.py src\models\multimodal_model.py src\training\losses.py
```

检查融合数据集：

```powershell
python scripts\validate_fused_dataset.py
```

检查一个 batch：

```powershell
python scripts\inspect_batch.py --dataset-dir dataset_fused --split train --batch-size 4
```

## 10. 如何评估当前交接模型

```powershell
python scripts\evaluate.py --dataset-dir dataset_fused --labels-file dataset_fused\labels_masked.csv --split-file dataset_fused\split_source_aware.csv --checkpoint checkpoints\final_weak_supervised_baseline\best_model.pth --output-dir outputs\final_weak_supervised_baseline_check
```

输出：

```text
outputs/final_weak_supervised_baseline_check/evaluation_report.json
outputs/final_weak_supervised_baseline_check/predictions.csv
outputs/final_weak_supervised_baseline_check/confusion_matrix.csv
```

## 11. 如何继续训练

如果继续在当前融合数据集上训练，建议新建实验编号，不要覆盖 final 目录。

示例：

```powershell
New-Item -ItemType Directory -Force experiments\exp008_next | Out-Null
python scripts\train.py --config configs\train.yaml --dataset-dir dataset_fused --labels-file dataset_fused\labels_masked.csv --split-file dataset_fused\split_source_aware.csv --input-mode multimodal --epochs 30 --batch-size 16 --lr 0.0001 --weight-decay 0.0001 --label-smoothing 0.05 --checkpoint-dir checkpoints\exp008_next --device auto 2>&1 | Tee-Object -FilePath experiments\exp008_next\train.log
```

评估：

```powershell
python scripts\evaluate.py --dataset-dir dataset_fused --labels-file dataset_fused\labels_masked.csv --split-file dataset_fused\split_source_aware.csv --checkpoint checkpoints\exp008_next\best_model.pth --output-dir outputs\exp008_next
```

每次新训练必须保存：

```text
checkpoints/实验名/best_model.pth
checkpoints/实验名/last_model.pth
checkpoints/实验名/effective_config.yaml
outputs/实验名/evaluation_report.json
outputs/实验名/predictions.csv
outputs/实验名/confusion_matrix.csv
experiments/实验名/train.log
experiments/实验名/train_command.txt
experiments/实验名/experiment_summary.md
```

## 12. 单样本推理

```powershell
python scripts\infer.py --checkpoint checkpoints\final_weak_supervised_baseline\best_model.pth --image dataset_fused\images\IMG_000001.png --environment dataset_fused\environment.csv --capture-time "2023-06-02 00:00:00" --output outputs\single_prediction_final.json --device auto
```

注意：`capture-time` 必须能在 `environment.csv` 中找到拍摄前 24 小时的环境数据。

## 13. 当前实验结论

简要结论：

- 随机划分下结果很好，但偏乐观。
- 消融实验表明多模态设计有价值。
- source-aware 实验暴露出跨来源泛化问题。
- 标签审计证明不能四任务全强监督。
- masked multitask training 改善了健康等级和异常漏报，但异常误报仍偏多。

详细结论见：

```text
experiments/final_experiment_summary.md
```

## 14. 后续最重要工作

下一位同学接手后，不建议继续盲目调参。优先做：

1. 收集真实大棚图片和环境数据。
2. 人工审核标签。
3. 用 `final_weak_supervised_baseline` 作为预训练模型进行微调。
4. 在真实大棚独立测试集上验收。
5. 根据误报/漏报要求调整异常预警阈值或 class weight。
