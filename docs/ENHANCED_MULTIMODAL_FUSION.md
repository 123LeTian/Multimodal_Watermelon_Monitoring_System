# 原模型与 V4 有效机制融合说明

## 保留内容

融合版本继续使用原项目的主体结构：

- ResNet50 图像编码器；
- Transformer 环境时序编码器；
- 图像与环境特征融合；
- 生长阶段、健康等级、成熟度、异常告警四个任务头；
- 正式模型使用 `multimodal`；
- `image_only` 和 `env_only` 继续用于消融实验。

已有检查点按旧配置构建，不启用传感器 mask，因此仍可加载。融合训练使用独立的
`checkpoints/enhanced_fused_multimodal`，不会覆盖原模型。

## 已吸收的机制

1. 图片只匹配拍摄前 24 小时内的环境记录。
2. 数据含 `device_id` 时按同设备匹配；真实多设备数据应开启
   `require_device_id: true`。
3. 每个传感器都有独立可用性 mask，缺失值不会被当成真实的零。
4. 环境均值和标准差由训练分割实际使用的环境记录自动计算，并写入检查点配置。
5. 训练集启用轻量图像增强；验证集和测试集不启用随机增强。
6. 图片等比例缩放并填充，不裁掉叶片、藤蔓、病斑或果实边缘。
7. 使用 ImageNet 预训练 ResNet50、余弦学习率、梯度裁剪、标签平滑和早停。
8. 训练前可独立检查时间范围、设备和 24 小时窗口覆盖率。

## 三项风险的处理

### 图片与环境时间不匹配

`scripts/validate_multimodal_alignment.py` 会输出图片和环境数据的时间范围，并在
覆盖率不足时返回失败。V4 中 2026 年的环境数据不能与当前 2023 年图片混合训练。

注意：检查通过只能证明 CSV 中的时间字段能够匹配，不能证明公开图片确实是在对应
大棚、对应时间拍摄。最终多模态结论仍需同设备、同期真实采集数据。

### 湿度类型不同

项目继续使用 `soil_humidity`。如果输入只有 `air_humidity` 而缺少
`soil_humidity`，数据加载和对齐检查会明确报错。空气湿度不能通过改列名冒充
土壤湿度；如未来需要同时使用，应增加新的传感器维度并重新训练。

### 弱标签

弱标签仍由 `growth_stage_valid`、`health_level_valid`、
`maturity_level_valid` 和 `abnormal_alert_valid` 控制。mask 为 0 的样本不参与
对应任务损失。融合配置要求四个 mask 字段全部存在，不会把弱标签强制改成有效标签。

当前 `dataset_fused/labels_masked.csv` 的生长阶段 mask 全部为 0，因此本数据集
不会训练生长阶段头。补充可信人工标注后，才应把相应样本的
`growth_stage_valid` 设置为 1。

## 使用方法

先检查当前融合数据：

```powershell
python scripts\validate_multimodal_alignment.py `
  --dataset-dir dataset_fused `
  --labels-file dataset_fused\labels_masked.csv `
  --environment-file dataset_fused\environment.csv `
  --require-task-masks `
  --minimum-coverage 1.0
```

开始融合版本训练：

```powershell
python scripts\train.py --config configs\train_enhanced_fused.yaml
```

真实多设备数据训练前，应把配置中的 `require_device_id` 改为 `true`，并确保
`labels.csv` 与 `environment.csv` 都包含 `device_id`。

如果真实数据确实缺少光照或 pH，可以把对应字段加入
`optional_environment_fields`。只有确认传感器没有采集该指标时才能这样设置，
不能用它掩盖错误列名或错误文件。
