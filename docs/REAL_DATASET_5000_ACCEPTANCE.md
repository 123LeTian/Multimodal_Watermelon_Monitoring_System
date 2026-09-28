# 5,000 张真实大棚西瓜数据集建设与验收规范

本规范用于把当前弱监督基线升级为可信的真实数据初版模型。`dataset_fused` 仍可作为预训练或工程演示数据，但不能混入本数据集充当人工真值。

## 目录

建议使用独立目录：

```text
dataset_real_v001/
├─ images/
├─ labels.csv
├─ environment.csv
├─ split.csv
└─ README.md
```

图片必须来自同一采集周期内的真实大棚，且每张图片、前 24 小时环境数据和人工标签真实对应。不要为了凑数量复制图片、平移时间、固定填 pH，或把公开图片伪装成同期大棚数据。

## labels.csv

训练必需字段：

```text
image_id,image_path,capture_time,growth_stage,health_level,maturity_level,abnormal_alert
```

强烈建议增加审计字段：

```text
plant_id,batch_id,device_id,annotator_count,label_status,review_notes
```

标签编码：

| 字段 | 编码 |
| --- | --- |
| `growth_stage` | 0 发芽期, 1 伸蔓期, 2 开花期, 3 坐果期, 4 膨大期, 5 成熟期 |
| `health_level` | 0 正常, 1 轻微异常, 2 中度异常, 3 严重异常 |
| `maturity_level` | 0 未成熟, 1 接近成熟, 2 成熟, 3 过熟 |
| `abnormal_alert` | 0 无异常, 1 有异常 |

## environment.csv

必需字段：

```text
timestamp,device_id,temperature,soil_humidity,light,ph
```

推荐每 10 分钟一条记录。每张图片拍摄前 24 小时窗口应至少有 90% 覆盖率，最大连续缺失不超过 30 分钟。没有连续 pH 传感器时，宁可留空并使用传感器掩码方案，也不要复制固定值。

## 分布目标

目标总量 5,000 张，允许约 10% 浮动，但不要伪造标签补齐。

| 生长阶段 | 正常 | 轻微异常 | 中度异常 | 严重异常 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 发芽期 | 250 | 70 | 50 | 30 | 400 |
| 伸蔓期 | 350 | 100 | 90 | 60 | 600 |
| 开花期 | 350 | 140 | 130 | 80 | 700 |
| 坐果期 | 400 | 160 | 140 | 100 | 800 |
| 膨大期 | 600 | 240 | 220 | 140 | 1,200 |
| 成熟期 | 550 | 290 | 270 | 190 | 1,300 |
| 合计 | 2,500 | 1,000 | 900 | 600 | 5,000 |

成熟度目标：

| 成熟度 | 数量 |
| --- | ---: |
| 未成熟 | 3,000 |
| 接近成熟 | 800 |
| 成熟 | 900 |
| 过熟 | 300 |

异常告警建议：无异常 2,500-2,800 张，有异常 2,200-2,500 张。严重异常最好不少于 400 张，过熟不少于 300 张。

## 划分

推荐：

```text
train 3500
val    750
test   750
```

同一植株同一天的相似照片必须放在同一个 split。测试集最好来自独立批次或后续时间段，确定后不要再参与调参。

## 审计命令

采集到数据后运行：

```powershell
python scripts\audit_real_dataset.py `
  --dataset-dir dataset_real_v001 `
  --verify-images
```

脚本会输出并写入：

```text
dataset_real_v001/real_dataset_audit_report.json
```

只有报告 `status` 为 `ok` 时，才建议进入训练。若只是先做 500 张流程验收，可以保留同一目录结构运行审计，报告中的数量问题会作为提醒保留。
