# 西瓜多模态监测模型 V3 完整使用说明

版本：V3 Reviewed

模型文件：`runtime/checkpoints/best_model.pth`

模型 SHA-256：`46ECD957FD0AA5D5DA9FFC960D248038223271F34C8AB5C8AB1643051C3EF684`

异常告警阈值：`0.84`

## 1. 模型用途

模型对一张西瓜相关图片及该图片拍摄前 24 小时的环境时序数据进行联合推理，输出：

1. 健康等级：正常、轻微异常、中度异常、严重异常。
2. 成熟度：未成熟、接近成熟、成熟、过熟。
3. 异常告警：无异常、有异常。
4. 生长阶段：保留接口，但 V3 本轮没有有效生长阶段标签参与训练，不可作为正式结果。

这是研究阶段模型，不是经过生产认证的病害诊断设备。模型结果只能作为辅助判断，不能代替人工巡检和农艺专家诊断。

## 2. 运行条件

### 2.1 最低要求

- 操作系统：Windows 10 或 Windows 11。
- Python：建议 3.10 或 3.11。
- 内存：建议至少 8 GB。
- 可用磁盘：建议至少 2 GB。
- GPU：可选。没有 NVIDIA GPU 时可以使用 CPU 推理。
- 网络：安装 Python 依赖时需要；依赖安装完成后，本地推理不需要联网，也不消耗任何 AI 工具额度。

### 2.2 安装依赖

解压推理包后，在包的根目录打开 PowerShell，运行：

```powershell
.\setup.ps1
```

脚本会创建独立的 `.venv` 并安装所需依赖。后续运行推理脚本时会自动使用该环境，不需要手动激活。

如 PowerShell 禁止执行脚本，可仅对当前窗口临时放行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 3. 目录结构

```text
watermelon_v3_inference/
├─ input/
│  ├─ image.jpg                   待推理图片
│  ├─ environment.csv             对应环境时序数据
│  └─ environment_template.csv    CSV 格式模板
├─ output/
│  └─ prediction.json             默认输出结果
├─ examples_20/                   20 组图片、环境数据和结果
├─ runtime/                       模型、推理代码和依赖清单
├─ README.md                      本使用说明
├─ setup.ps1                      首次安装入口
├─ run_inference.ps1              正式推理入口
└─ run_20_examples.ps1            20 组样例批量入口
```

只进行推理时，不需要训练集、验证集、标签文件或训练软件。

## 4. 一次推理需要哪些输入

每次推理必须提供三项内容：

1. 一张待识别图片。
2. 图片拍摄前 24 小时的环境数据 CSV。
3. 图片准确拍摄时间 `CaptureTime`。

图片、拍摄时间和环境数据必须来自同一对象、同一地点和同一时间段。不能使用一张图片配另一座大棚或另一天的传感器数据。

## 5. 图片输入要求

### 5.1 文件要求

| 项目 | 要求 |
| --- | --- |
| 支持格式 | JPG、JPEG、PNG 等 Pillow 可读取的常见位图格式 |
| 色彩 | 彩色图片；程序会统一转换为 RGB |
| 建议分辨率 | 不低于 `512 × 512`，主体细节应清楚 |
| 文件大小 | 建议不超过 10 MB |
| 图片数量 | 单次命令处理一张 |
| EXIF 时间 | 程序不会自动读取，仍须显式传入 `CaptureTime` |

程序会保持图片长宽比，将图片缩放并用黑边补齐到 `224 × 224`，然后按 ImageNet 均值和标准差归一化。

### 5.2 拍摄质量要求

- 图片应对焦清楚，无明显运动模糊。
- 光照应足以看清颜色、纹理、病斑和腐烂区域。
- 西瓜果实或相关叶片应为画面主体，尽量减少人员、工具、文字、水印等干扰。
- 不要使用截图、拼图、重复图片或经过强滤镜处理的图片。
- 不要把其他水果、无法辨认的远景或严重遮挡图片作为正式输入。
- 用于健康等级判断时，应尽量展示完整病害范围；只拍到极小局部时，严重度可能被误判。
- 同一批采集尽量保持相近的相机、距离和拍摄角度。

模型训练图像同时包含叶片病害和果实质量图片，因此输入域变化较大。真实大棚使用前必须先做本地盲测。

### 5.3 图片放置位置

默认文件名：

```text
input/image.jpg
```

也可以保留原文件名，并在命令中通过 `-Image` 指定：

```powershell
-Image "input\watermelon_001.png"
```

## 6. 环境数据输入要求

### 6.1 文件位置与编码

默认位置：

```text
input/environment.csv
```

建议保存为 UTF-8 CSV，使用英文逗号分隔。小数必须使用点号，例如 `28.5`，不能写成 `28,5`。

### 6.2 必需字段

```csv
timestamp,temperature,soil_humidity,light,ph
2026-09-27 12:00:00,27.4,36.2,520.0,6.7
```

| 字段 | 必需 | 类型与单位 | 说明 |
| --- | --- | --- | --- |
| `timestamp` | 是 | `YYYY-MM-DD HH:MM:SS` | 传感器记录时间 |
| `temperature` | 是 | 摄氏度 `°C` | 空气温度，不是土壤温度 |
| `soil_humidity` | 是 | 百分比 `%` | 土壤含水率/土壤湿度，不是空气湿度 |
| `light` | 是 | PAR 口径 | 训练数据使用 PAR；不能把 lux 原值当作相同单位 |
| `ph` | 是 | pH，范围通常为 0–14 | 土壤 pH |

所有字段都必须为有效数值，不能包含 `%`、`°C`、`lux` 等单位文字，不能留空，也不能写 `NaN`。

### 6.3 当前模型的训练分布

以下范围仅用于识别明显的输入口径错误，不是农业合理范围，也不是硬性截断规则：

| 字段 | V3 训练数据范围 | 训练归一化中心 | 训练归一化尺度 |
| --- | ---: | ---: | ---: |
| `temperature` | 0.15–41.13 | 21.4761 | 6.0849 |
| `soil_humidity` | 28.90–41.00 | 34.9881 | 3.2051 |
| `light` | 0–1454 | 316.5772 | 428.1827 |
| `ph` | 固定 6.8 | 6.8 | 1.0 |

重要限制：

- 训练数据的 `light` 来自 PAR。PAR 常见单位为 `μmol·m⁻²·s⁻¹`。如果设备输出 lux，应先建立设备对应的转换或用真实数据重新标定，不能直接混用。
- V3 训练数据中的 pH 固定为 `6.8`，因此模型没有充分学习 pH 变化。接口仍要求 pH，但不能据此宣称模型已验证 pH 对预测的贡献。
- 不要为了让程序运行而伪造 pH、复制固定值或把空气湿度改名为土壤湿度。
- 输入明显超出训练分布时，程序仍可能给出结果，但可靠性会降低。

### 6.4 时间窗口要求

假设图片拍摄时间为：

```text
2026-09-28 12:00:00
```

环境数据必须覆盖：

```text
2026-09-27 12:00:00 至 2026-09-28 12:00:00
```

推荐要求：

- 与训练数据一致，先整理为每小时一条。
- 包含窗口起点和拍摄时刻时，共 25 条整点记录。
- 各字段按时间升序排列；程序也会重新排序。
- 图片时间和传感器时间必须使用同一时区。
- `CaptureTime` 必须与图片实际拍摄时间一致。
- 不应存在大段缺测、错误日期或时间漂移。
- 高频数据（例如每 10 分钟一条）建议先聚合为小时均值，以降低与训练分布的差异。

技术上，只要窗口中存在至少一条记录程序就可能运行，但这不满足可靠推理的数据质量要求。正式使用应尽量提供完整 24 小时窗口。

### 6.5 多传感器设备

如果 CSV 只有一个设备，可以不写 `device_id`，程序按 `default` 设备处理。

如果同一个 CSV 包含多个设备，必须增加 `device_id` 列：

```csv
timestamp,device_id,temperature,soil_humidity,light,ph
2026-09-27 12:00:00,greenhouse_01,27.4,36.2,520.0,6.7
```

运行时指定与图片对应的设备：

```powershell
-SensorDeviceId "greenhouse_01"
```

不得把多个设备的数据混成一条时间序列。

## 7. 拍摄时间要求

命令中的拍摄时间格式固定为：

```text
YYYY-MM-DD HH:MM:SS
```

正确示例：

```text
2026-09-28 12:00:00
```

错误示例：

```text
2026/09/28 12:00
2026-09-28T12:00:00
09-28-2026 12:00:00
```

## 8. 运行推理

### 8.1 使用默认输入路径

将图片放到 `input/image.jpg`，环境数据放到 `input/environment.csv`，然后运行：

```powershell
.\run_inference.ps1 -CaptureTime "2026-09-28 12:00:00"
```

如 PowerShell 禁止执行脚本，可仅对当前窗口临时放行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 8.2 指定输入与输出文件

```powershell
.\run_inference.ps1 `
  -Image "input\watermelon_001.png" `
  -Environment "input\environment_001.csv" `
  -CaptureTime "2026-09-28 12:00:00" `
  -Output "output\watermelon_001.json"
```

### 8.3 强制使用 CPU 或 GPU

CPU：

```powershell
.\run_inference.ps1 `
  -CaptureTime "2026-09-28 12:00:00" `
  -Device cpu
```

CUDA GPU：

```powershell
.\run_inference.ps1 `
  -CaptureTime "2026-09-28 12:00:00" `
  -Device cuda
```

默认 `-Device auto`，有可用 CUDA 时使用 GPU，否则使用 CPU。

### 8.4 多设备输入

```powershell
.\run_inference.ps1 `
  -CaptureTime "2026-09-28 12:00:00" `
  -SensorDeviceId "greenhouse_01"
```

### 8.5 多张图片

当前脚本一次处理一张图片。多张图片应逐张运行，并为每张图片提供准确时间、对应环境窗口和不同输出文件名：

```powershell
.\run_inference.ps1 `
  -Image "input\001.jpg" `
  -Environment "input\all_environment.csv" `
  -CaptureTime "2026-09-28 08:00:00" `
  -Output "output\001.json"

.\run_inference.ps1 `
  -Image "input\002.jpg" `
  -Environment "input\all_environment.csv" `
  -CaptureTime "2026-09-28 09:00:00" `
  -Output "output\002.json"
```

同一个环境 CSV 可以包含更长时间范围，程序会自动截取每张图片拍摄前 24 小时的数据。

## 9. 输出位置与字段解释

默认输出：

```text
output/prediction.json
```

主要结构：

```json
{
  "capture_time": "2026-09-28 12:00:00",
  "environment_rows": 25,
  "abnormal_alert_threshold": 0.84,
  "predictions": {
    "health_level": {
      "class_id": 2,
      "class_name": "中度异常",
      "confidence": 0.81,
      "probabilities": [0.03, 0.12, 0.81, 0.04]
    },
    "abnormal_alert": {
      "class_id": 1,
      "class_name": "有异常",
      "confidence": 0.93,
      "probabilities": [0.07, 0.93],
      "decision_threshold": 0.84
    }
  }
}
```

通用字段：

| 字段 | 含义 |
| --- | --- |
| `class_id` | 预测类别编号 |
| `class_name` | 预测类别中文名称 |
| `confidence` | 被输出类别对应的模型概率，不等同于农业诊断可信度 |
| `probabilities` | 按类别编号排列的全部类别概率 |
| `environment_rows` | 实际进入模型的环境记录数量 |
| `window_start` / `window_end` | 实际截取的 24 小时时间范围 |
| `sensor_device_id` | 本次使用的传感器设备 |

### 9.1 健康等级定义

| `class_id` | 名称 | 判定口径 |
| ---: | --- | --- |
| 0 | 正常 | 未见明显异常 |
| 1 | 轻微异常 | 局部轻微症状 |
| 2 | 中度异常 | 多个区域或明显病斑 |
| 3 | 严重异常 | 大面积病害、萎蔫或腐烂 |

### 9.2 成熟度定义

| `class_id` | 名称 |
| ---: | --- |
| 0 | 未成熟 |
| 1 | 接近成熟 |
| 2 | 成熟 |
| 3 | 过熟 |

### 9.3 异常告警规则

| `class_id` | 名称 |
| ---: | --- |
| 0 | 无异常 |
| 1 | 有异常 |

模型计算“有异常”的概率。当该概率大于或等于 `0.84` 时，输出“有异常”。不要随意修改阈值；修改阈值会改变漏报率和误报率。

### 9.4 生长阶段

输出中可能出现发芽期、伸蔓期、开花期、坐果期、膨大期或成熟期，但 V3 的生长阶段任务权重为 0，本轮没有有效标签训练或验收。该字段必须忽略。

## 10. 批量运行 20 组样例

推理包额外提供了健康等级四类各 5 张的 20 组西瓜叶片样例：

```powershell
.\run_20_examples.ps1
```

样例位置：

```text
examples_20/
```

图片总览见 `examples_20/contact_sheet.jpg`，类别和文件对应关系见 `examples_20/manifest.csv`，批量结果见 `examples_20/results_summary.csv`。这批样例经过筛选，只用于演示，不代表随机样本或真实大棚准确率。

## 11. 输入验收清单

正式推理前逐项确认：

- [ ] 图片文件可以正常打开。
- [ ] 图片清晰，西瓜果实或叶片是主要对象。
- [ ] 已记录图片准确拍摄时间。
- [ ] CSV 使用英文列名和英文逗号。
- [ ] CSV 包含 `timestamp,temperature,soil_humidity,light,ph`。
- [ ] 所有必需传感器字段均有有效数值。
- [ ] 温度单位为摄氏度。
- [ ] `soil_humidity` 是土壤湿度，不是空气湿度。
- [ ] `light` 与训练数据采用兼容的 PAR 口径，不是未经转换的 lux。
- [ ] 图片与传感器属于同一地点、设备和时间段。
- [ ] 环境数据覆盖拍摄前 24 小时，建议小时级 25 条。
- [ ] 图片时间与传感器时间使用同一时区。
- [ ] 多设备 CSV 已传入正确的 `SensorDeviceId`。
- [ ] 每张图片使用单独的输出文件，避免覆盖上一条结果。

## 12. 常见错误

### 12.1 找不到 Python

```text
python is not recognized
```

安装 Python 3.10 或 3.11，并在安装时勾选 Add Python to PATH。

### 12.2 PowerShell 不允许运行脚本

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

该设置只影响当前 PowerShell 窗口。

### 12.3 找不到图片或环境文件

检查 `-Image`、`-Environment` 路径和扩展名。相对路径以推理包根目录为基准。

### 12.4 环境窗口没有数据

```text
no environment data found
```

检查：

- `CaptureTime` 是否为图片真实时间。
- CSV 时间是否覆盖拍摄前 24 小时。
- 图片和 CSV 是否使用相同时区。
- 时间格式是否为 `YYYY-MM-DD HH:MM:SS`。
- 多设备时是否选中了正确 `SensorDeviceId`。

### 12.5 缺少 soil_humidity

空气湿度不能代替土壤湿度。不要简单把 `air_humidity` 改名为 `soil_humidity`。

### 12.6 CSV 包含多个设备

使用：

```powershell
-SensorDeviceId "设备编号"
```

### 12.7 CUDA 不可用

改用 CPU：

```powershell
-Device cpu
```

### 12.8 中文在控制台显示乱码

以保存的 UTF-8 JSON 为准，也可先运行：

```powershell
chcp 65001
```

## 13. 模型限制与结果使用原则

- V3 是来源隔离训练后的研究基线，不是生产认证模型。
- 训练图片来自多个公开数据源，与实际大棚相机存在域差异。
- 环境序列并非与每张公开图片真实同步采集，当前多模态效果仍需真实配对数据验证。
- pH 训练值固定为 6.8，pH 变化贡献未经验证。
- 验证集和测试集没有同时覆盖健康等级全部四类。
- 生长阶段没有在 V3 中有效训练。
- 低清晰度、遮挡、强反光、异常视角和训练分布外输入可能产生高置信度错误。
- 应保留原始图片、原始传感器数据、模型版本、阈值和 JSON 输出，便于追溯。
- 正式应用前应使用真实大棚数据进行盲测，重点检查异常召回率、误报率和四级健康混淆矩阵。

## 14. 最简交付说明

给使用者发送整个 `watermelon_v3_inference.zip`，不要只发送 `best_model.pth`。使用者解压、安装依赖，将图片与环境 CSV 放入 `input`，执行 `run_inference.ps1`，然后从 `output` 读取 JSON。模型推理不需要训练软件，也不消耗 AI 工具额度。
