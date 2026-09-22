# 大棚西瓜监测机器学习项目

本项目包含大棚西瓜生长监测项目所需的机器学习流程代码。

## 当前阶段

当前项目处于机器学习工程骨架搭建阶段。真实训练数据尚未采集完成，
因此现阶段应先使用 `dataset_sample` 开发和测试代码。

## 项目目录结构

```text
watermelon_monitoring/
├─ configs/       训练和实验配置文件
├─ src/           可复用的 Python 源代码
│  ├─ datasets/   数据集读取和预处理代码
│  ├─ models/     ResNet + Transformer 多模态模型代码
│  ├─ training/   损失函数、指标和训练辅助代码
│  └─ utils/      通用工具函数
├─ scripts/       训练、评估和推理入口脚本
├─ checkpoints/   保存训练得到的模型权重
├─ experiments/   实验日志和指标记录
├─ outputs/       推理结果、报告和导出文件
└─ docs/          模型和训练相关文档
```

## 计划中的处理流程

```text
模拟数据或真实数据集
-> 数据集读取器
-> ResNet 图像分支
-> Transformer 环境数据分支
-> 特征融合
-> 多任务预测头
-> 模型评估和推理输出
```

## 下一步

实现数据集读取器，使模拟数据集能够被读取为训练 batch。
