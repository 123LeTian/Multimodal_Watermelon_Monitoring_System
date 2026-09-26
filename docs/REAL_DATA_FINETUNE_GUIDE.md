# 真实大棚数据接入与微调说明

## 1. 目的

当前 `dataset_fused` 只能作为弱监督预训练/基线数据。要让模型真正服务于大棚西瓜监测，必须接入真实大棚数据，并用人工审核标签做微调和独立测试。

## 2. 真实数据目录建议

建议新建：

```text
dataset_real_v001/
├─ images/
├─ environment.csv
├─ labels.csv
├─ split.csv
└─ README.md
```

不要直接覆盖 `dataset_fused/`。

## 3. 图片要求

图片放在：

```text
dataset_real_v001/images/
```

建议命名：

```text
REAL_20260927_GREENHOUSE01_000001.jpg
REAL_20260927_GREENHOUSE01_000002.jpg
```

建议记录：

- 大棚编号
- 拍摄日期和时间
- 拍摄对象：叶片、果实、整株、病害区域
- 拍摄设备
- 是否补光
- 是否模糊、遮挡、过曝

## 4. 环境数据要求

`environment.csv` 必须包含：

```text
timestamp,temperature,soil_humidity,light,ph
```

说明：

- `timestamp`: 时间，建议格式 `YYYY-MM-DD HH:MM:SS`
- `temperature`: 温度
- `soil_humidity`: 土壤湿度
- `light`: 光照强度
- `ph`: 土壤 pH

当前项目没有使用空气湿度字段。不要把土壤湿度写成 `humidity`，也不要混用 `humidity` 和 `soil_humidity`。

每张图片的 `capture_time` 前 24 小时内必须能找到环境记录。

## 5. 标签文件要求

`labels.csv` 必须包含：

```text
image_id,image_path,capture_time,growth_stage,health_level,maturity_level,abnormal_alert
```

标签编码：

| 任务 | 编码 |
| --- | --- |
| `growth_stage` | 0 发芽期, 1 伸蔓期, 2 开花期, 3 坐果期, 4 膨大期, 5 成熟期 |
| `health_level` | 0 正常, 1 轻微异常, 2 中度异常, 3 严重异常 |
| `maturity_level` | 0 未成熟, 1 接近成熟, 2 成熟, 3 过熟 |
| `abnormal_alert` | 0 无异常, 1 有异常 |

真实数据中，如果某个任务无法判断，不要随便填默认值。建议先人工补标，或者后续使用 task mask。

## 6. 真实数据 task mask 建议

如果真实数据每张图四个任务都经过人工审核，可以不用 mask，直接使用 `labels.csv`。

如果某些图片只适合部分任务，建议生成 `labels_masked.csv`，新增：

```text
growth_stage_valid
health_level_valid
maturity_level_valid
abnormal_alert_valid
```

示例：

- 叶片病害图片：`health_level_valid=1`，`abnormal_alert_valid=1`，成熟度可设 0。
- 果实成熟度图片：`maturity_level_valid=1`，健康和异常如果没有人工判断可设 0。
- 整株生长期图片：`growth_stage_valid=1`。

## 7. 数据划分要求

`split.csv` 必须包含：

```text
image_id,split
```

建议比例：`train 70%`、`val 15%`、`test 15%`。

注意：同一株、同一天、同一连续拍摄序列，尽量不要同时出现在 train 和 test，避免数据泄漏。

## 8. 第一批真实数据数量建议

第一次微调不需要特别大，但要保证每类都有样本。

建议最低起步：

```text
总量 300-800 张
每个关键类别至少 30-50 张
异常样本尽量不少于 100 张
```

如果异常样本少，宁可先做二分类异常预警微调，也不要强行做完整四任务结论。

## 9. 从当前基线模型开始微调

当前 `scripts/train.py` 还没有实现从 checkpoint 恢复继续训练的参数。如果要严格从 `final_weak_supervised_baseline` 微调，需要先给训练脚本增加 `--resume-checkpoint` 或 `--pretrained-checkpoint` 功能。

临时方案：

- 如果真实数据量足够，可以直接用当前结构从头训练。
- 如果真实数据量较少，建议先实现 checkpoint 加载，再微调。

建议后续新增参数：

```text
--pretrained-checkpoint checkpoints\final_weak_supervised_baseline\best_model.pth
```

实现后训练命令示例：

```powershell
python scripts\train.py --config configs\train.yaml --dataset-dir dataset_real_v001 --labels-file dataset_real_v001\labels.csv --split-file dataset_real_v001\split.csv --input-mode multimodal --epochs 20 --batch-size 8 --lr 0.00005 --checkpoint-dir checkpoints\real_finetune_exp001 --device auto --pretrained-checkpoint checkpoints\final_weak_supervised_baseline\best_model.pth
```

## 10. 真实数据验收标准

真实大棚模型验收时，至少看：

- 测试集样本数是否足够
- 每个类别 support 是否足够
- `health_level` macro F1
- `abnormal_alert` positive precision、positive recall、positive F1
- 异常误报数量
- 异常漏报数量
- 错误样本是否符合农业常识

对于异常报警，不能只看 accuracy。必须同时看误报和漏报。

## 11. 真实数据实验归档

每次真实数据训练建议命名：

```text
real_finetune_exp001
real_finetune_exp002
real_finetune_exp003
```

每次必须保存：

```text
checkpoints/real_finetune_exp001/best_model.pth
checkpoints/real_finetune_exp001/last_model.pth
checkpoints/real_finetune_exp001/effective_config.yaml
outputs/real_finetune_exp001/evaluation_report.json
outputs/real_finetune_exp001/predictions.csv
outputs/real_finetune_exp001/confusion_matrix.csv
experiments/real_finetune_exp001/train.log
experiments/real_finetune_exp001/train_command.txt
experiments/real_finetune_exp001/experiment_summary.md
```

## 12. 当前不建议做的事

- 不要把 `dataset_fused` 的随机划分高指标当最终结论。
- 不要把没有人工判断的任务标签随便填 0。
- 不要继续覆盖 `final_weak_supervised_baseline`。
- 不要只看训练集或验证集结果。
- 不要只用 accuracy 判断异常报警好坏。
