# 项目目录结构说明

## `configs/`

用于存放训练参数、数据集路径、模型选项和实验设置等 YAML 或 JSON 配置文件。

## `src/datasets/`

用于存放数据集类和数据预处理逻辑。当前首先实现针对 `../dataset_sample`
的数据集读取器。

## `src/models/`

用于存放多模态模型模块：

- 图像分支：ResNet50
- 环境数据分支：Transformer Encoder
- 特征融合层
- 多任务预测头

## `src/training/`

用于存放损失函数、评估指标、训练循环、验证循环和模型权重保存逻辑。

## `src/utils/`

用于存放通用辅助功能，例如配置读取、随机种子设置、日志记录和路径处理。

## `scripts/`

用于存放可直接从命令行运行的入口脚本：

- `train.py`
- `evaluate.py`
- `infer.py`
- `inspect_batch.py`

## `checkpoints/`

用于存放训练得到的模型权重。不要手动修改其中的权重文件。

## `experiments/`

用于存放实验日志、评估指标和实验配置快照。

## `outputs/`

用于存放推理结果、导出的报告和其他自动生成的输出文件。
