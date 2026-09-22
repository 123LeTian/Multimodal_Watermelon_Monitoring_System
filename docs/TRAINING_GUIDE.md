# 大棚西瓜多模态模型训练指南

## 1. 这份文档的作用

本文档用于指导从 GitHub 仓库开始，完成环境安装、模拟数据自检、真实数据训练、模型评估和单样本推理。

当前仓库已经包含：

- 数据读取代码
- ResNet50 + Transformer 多模态模型
- 多任务损失函数
- 训练脚本
- 评估脚本
- 单样本推理脚本
- 模拟数据集 `dataset_sample`
- 训练配置文件 `configs/train.yaml`

注意：`dataset_sample` 是模拟数据，只能用于验证工程流程，不能用于论文结论、模型效果汇报或真实种植决策。

## 2. 克隆项目

```powershell
git clone <GitHub仓库地址>
cd watermelon_monitoring
```

进入项目根目录后，应能看到：

```text
configs/
dataset_sample/
docs/
scripts/
src/
README.md
requirements.txt
```

## 3. 安装运行环境

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install -r requirements.txt
```

当前依赖包括：

```text
torch
torchvision
numpy
Pillow
PyYAML
```

检查依赖是否可用：

```powershell
python -c "import torch, torchvision, numpy, PIL, yaml; print('dependencies=ok')"
```

如果计划使用 NVIDIA GPU，检查 CUDA 是否可用：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 表示可以使用 `--device cuda`。如果输出 `False`，先使用 `--device cpu` 跑通流程。

## 4. 配置文件说明

训练配置文件是：

```text
configs/train.yaml
```

`scripts/train.py` 已经支持读取 YAML：

```powershell
python scripts\train.py --config configs\train.yaml
```

命令行参数可以覆盖 YAML 中的配置。优先级为：

```text
命令行参数 > configs/train.yaml > 程序内置默认值
```

例如，`configs/train.yaml` 中默认：

```yaml
training:
  epochs: 30
```

如果运行：

```powershell
python scripts\train.py --config configs\train.yaml --epochs 1
```

实际只训练 1 轮。

查看最终生效配置但不启动训练：

```powershell
python scripts\train.py --config configs\train.yaml --dry-run
```

常用命令行覆盖参数：

| 参数 | 作用 |
|---|---|
| `--dataset-dir` | 指定数据集目录 |
| `--epochs` | 指定训练轮数 |
| `--batch-size` | 指定 batch size |
| `--lr` | 指定学习率 |
| `--weight-decay` | 指定权重衰减 |
| `--seed` | 指定随机种子 |
| `--num-workers` | 指定 DataLoader 进程数 |
| `--device` | 指定 `cpu` 或 `cuda` |
| `--checkpoint-dir` | 指定模型权重保存目录 |
| `--window-hours` | 指定环境数据历史窗口 |
| `--image-size` | 指定图像尺寸 |
| `--label-smoothing` | 指定标签平滑系数 |

训练时，脚本会把最终生效配置保存为：

```text
checkpoints/某次实验目录/effective_config.yaml
```

这个文件是复现实验的重要依据。

## 5. 数据格式要求

真实数据必须整理为：

```text
dataset/
├─ images/
│  ├─ IMG_000001.jpg
│  ├─ IMG_000002.jpg
│  └─ ...
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

`split` 只能是：

```text
train
val
test
```

四个标签编码如下：

```text
growth_stage:   0 发芽期, 1 伸蔓期, 2 开花期, 3 坐果期, 4 膨大期, 5 成熟期
health_level:   0 正常, 1 轻微异常, 2 中度异常, 3 严重异常
maturity_level: 0 未成熟, 1 接近成熟, 2 成熟, 3 过熟
abnormal_alert: 0 无异常, 1 有异常
```

每张图片必须能通过 `capture_time` 匹配到拍摄前 24 小时内的环境数据。详细规范见：

```text
docs/数据交接规范表.md
```

重要提醒：当前代码默认环境字段是 `soil_humidity`。如果真实采集的是空气湿度而不是土壤湿度，需要先统一字段名称和模型输入配置，不要一部分叫 `humidity`，另一部分叫 `soil_humidity`。

## 6. 先用模拟数据检查流程

训练同学拿到仓库后，先不要直接训练真实数据。请先用 `dataset_sample` 检查代码流程。

检查模拟数据：

```powershell
python scripts\validate_sample_dataset.py
```

当前 `validate_sample_dataset.py` 固定检查 `dataset_sample`，不接收 `--dataset-dir` 参数。真实数据到达后，可按相同规则进行人工检查，或后续扩展该脚本。

检查数据 batch：

```powershell
python scripts\inspect_batch.py --dataset-dir dataset_sample --split train --batch-size 4
```

正常应看到类似：

```text
images_shape=(4, 3, 224, 224)
environment_shape=(4, 145, 4)
environment_mask_shape=(4, 145)
```

检查模型前向传播：

```powershell
python scripts\inspect_model.py --dataset-dir dataset_sample --split train --batch-size 4 --device cpu
```

正常应看到：

```text
growth_stage_logits=(4, 6)
health_level_logits=(4, 4)
maturity_level_logits=(4, 4)
abnormal_alert_logits=(4, 2)
```

检查损失函数和反向传播：

```powershell
python scripts\inspect_loss.py --dataset-dir dataset_sample --split train --batch-size 4 --device cpu
```

正常应看到：

```text
parameters_with_gradients=199/199
```

## 7. 用模拟数据跑通 1 轮训练

使用 YAML 配置，并临时覆盖训练轮数为 1：

```powershell
python scripts\train.py `
  --config configs\train.yaml `
  --dataset-dir dataset_sample `
  --epochs 1 `
  --device cpu `
  --checkpoint-dir checkpoints\sample_test
```

也可以写成一行：

```powershell
python scripts\train.py --config configs\train.yaml --dataset-dir dataset_sample --epochs 1 --device cpu --checkpoint-dir checkpoints\sample_test
```

成功后应生成：

```text
checkpoints/sample_test/
├─ best_model.pth
├─ last_model.pth
└─ effective_config.yaml
```

这一步只验证工程流程，不代表模型有真实预测能力。

## 8. 准备真实数据后的试运行

真实数据整理成 `dataset/` 后，先进行 1 轮试运行：

```powershell
python scripts\train.py `
  --config configs\train.yaml `
  --dataset-dir dataset `
  --epochs 1 `
  --batch-size 4 `
  --device cuda `
  --checkpoint-dir checkpoints\real_data_smoke_test
```

如果没有 GPU，使用 CPU：

```powershell
python scripts\train.py `
  --config configs\train.yaml `
  --dataset-dir dataset `
  --epochs 1 `
  --batch-size 2 `
  --device cpu `
  --checkpoint-dir checkpoints\real_data_smoke_test
```

如果试运行报错，优先检查：

```text
图片路径是否正确
environment.csv 字段名是否正确
labels.csv 是否包含四个标签
split.csv 是否包含 train/val/test
capture_time 是否能匹配前 24 小时环境数据
标签值是否超出规定范围
```

## 9. 正式训练

试运行无误后，使用真实数据正式训练：

```powershell
python scripts\train.py `
  --config configs\train.yaml `
  --dataset-dir dataset `
  --device cuda `
  --checkpoint-dir checkpoints\real_data_exp001
```

`configs/train.yaml` 中当前默认训练参数为：

```text
epochs = 30
batch_size = 4
learning_rate = 0.0001
weight_decay = 0.0001
```

如果要临时修改参数，可以命令行覆盖：

```powershell
python scripts\train.py `
  --config configs\train.yaml `
  --dataset-dir dataset `
  --epochs 50 `
  --batch-size 8 `
  --lr 0.00005 `
  --device cuda `
  --checkpoint-dir checkpoints\real_data_exp002
```

每次正式实验都建议使用独立目录，例如：

```text
checkpoints/real_data_exp001/
checkpoints/real_data_exp002/
checkpoints/real_data_exp003/
```

不要覆盖重要实验的权重文件。

## 10. 训练输出说明

每次训练会保存：

```text
best_model.pth
last_model.pth
effective_config.yaml
```

含义：

- `best_model.pth`：验证集 `total_loss` 最低时的模型。
- `last_model.pth`：最后一轮训练结束后的模型。
- `effective_config.yaml`：本次实验真正生效的完整配置。

训练日志中会输出：

```text
train_total_loss
val_total_loss
各任务 loss
saved_best
saved_last
```

## 11. 模型评估

训练完成后，用测试集评估最佳模型：

```powershell
python scripts\evaluate.py `
  --dataset-dir dataset `
  --checkpoint checkpoints\real_data_exp001\best_model.pth `
  --output-dir outputs\real_data_exp001 `
  --split test `
  --batch-size 4 `
  --device cuda
```

输出文件：

```text
outputs/real_data_exp001/
├─ evaluation_report.json
├─ predictions.csv
└─ confusion_matrix.csv
```

重点查看：

```text
Accuracy
Macro-F1
Macro-Precision
Macro-Recall
混淆矩阵
异常预警正类的 Precision、Recall、F1
```

测试集只用于最终评估，不要在反复调参时持续使用测试集。

## 12. 单样本推理

使用训练好的模型对单张图片进行预测：

```powershell
python scripts\infer.py `
  --checkpoint checkpoints\real_data_exp001\best_model.pth `
  --image dataset\images\IMG_000001.jpg `
  --environment dataset\environment.csv `
  --capture-time "2026-09-21 08:30:00" `
  --output outputs\real_data_exp001\single_prediction.json `
  --device cuda
```

输出 JSON 会包含：

```text
生长阶段
健康等级
成熟度等级
生长异常预警
每一类的概率
预测置信度
```

## 13. 当前脚本边界

当前 `scripts/train.py` 已经读取 `configs/train.yaml`。

当前 `scripts/evaluate.py` 和 `scripts/infer.py` 主要通过命令行参数运行，默认使用当前代码里的模型结构。请暂时不要随意修改 `configs/train.yaml` 中会改变模型结构的字段，例如：

```text
model.sensor_dim
model.time_feature_dim
model.output_tasks
model.image_pretrained
```

如果这些字段被改动，评估和推理脚本也需要同步适配。当前建议保持这些模型结构参数不变，只调整训练参数：

```text
epochs
batch_size
learning_rate
weight_decay
device
checkpoint_dir
dataset_dir
```

## 14. 每次实验需要记录什么

每次正式实验至少记录：

```text
实验编号
数据集版本
训练日期
训练集、验证集、测试集样本数量
GPU 型号
Python 和 PyTorch 版本
批次大小
学习率
权重衰减
训练轮数
随机种子
最佳 epoch
最佳验证集 loss
测试集各任务指标
错误样本分析
```

建议在 `experiments/real_data_exp001/` 中保存：

```text
effective_config.yaml
train.log
evaluation_report.json
predictions.csv
confusion_matrix.csv
```

## 15. 训练同学交付内容

训练同学每次正式训练后，建议提交：

```text
checkpoints/real_data_exp001/best_model.pth
checkpoints/real_data_exp001/last_model.pth
checkpoints/real_data_exp001/effective_config.yaml
outputs/real_data_exp001/evaluation_report.json
outputs/real_data_exp001/predictions.csv
outputs/real_data_exp001/confusion_matrix.csv
```

同时说明：

```text
使用的数据版本
训练命令
是否使用 GPU
训练轮数
batch size
学习率
验证集最佳 loss
测试集主要指标
目前最容易预测错误的样本类型
```

## 16. 模拟数据的边界

`dataset_sample` 只有 20 张模拟图片、208 条模拟环境记录，训练集、验证集和测试集分别为 14、3、3 条。

模拟数据可以验证：

```text
数据读取是否正常
模型前向传播是否正常
损失函数和反向传播是否正常
训练脚本是否正常
评估脚本是否正常
单样本推理是否正常
```

模拟数据不能验证：

```text
模型对真实西瓜的识别能力
模型在不同大棚和品种上的泛化能力
模型的实际预警可靠性
模型对产量或品质的提升效果
```

因此，在真实数据到达前，不要根据模拟数据报告模型准确率或论文结论。
