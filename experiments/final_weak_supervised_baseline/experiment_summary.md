# exp007_masked_multitask_training

## 本次实验目的

本次实验实现 per-sample task mask，让 `dataset_fused` 中的弱监督样本只在可信任务上参与 loss。目标不是继续调参，而是验证 exp006_label_audit 的结论是否能缓解 exp005_source_holdout 中暴露的数据来源泛化问题。

## mask 规则

| source_dataset | growth_stage | health_level | maturity_level | abnormal_alert | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| watermelon_disease_recognition | 0 | 1 | 0 | 1 | 叶片健康/病害识别，适合健康等级和异常预警 |
| kurdistan_watermelon_disease | 0 | 1 | 0 | 1 | 叶片健康、病害、虫害、缺素识别，适合健康等级和异常预警 |
| watermelon_ripe_semiripe_unripe | 0 | 0 | 1 | 0 | 果实成熟度数据，适合 maturity_level |
| watermelon_ripe_unripe | 0 | 0 | 1 | 0 | ripe/unripe 果实数据，适合 maturity_level |
| fruq_db | 0 | 1 | 1 | 1 | 中等可信补充来源，可提供果实质量异常/过熟信号，后续可考虑降权 |

## 训练命令

```powershell
python scripts\train.py --config configs\train.yaml --dataset-dir dataset_fused --labels-file dataset_fused\labels_masked.csv --split-file dataset_fused\split_source_aware.csv --input-mode multimodal --epochs 30 --batch-size 16 --lr 0.0001 --weight-decay 0.0001 --label-smoothing 0.05 --checkpoint-dir checkpoints\exp007_masked_multitask_training --device auto 2>&1 | Tee-Object -FilePath experiments\exp007_masked_multitask_training\train.log
```

## 评估命令

```powershell
python scripts\evaluate.py --dataset-dir dataset_fused --labels-file dataset_fused\labels_masked.csv --split-file dataset_fused\split_source_aware.csv --checkpoint checkpoints\exp007_masked_multitask_training\best_model.pth --output-dir outputs\exp007_masked_multitask_training
```

## 每个任务参与训练/评估的样本数

| split | samples | growth_stage_valid | health_level_valid | maturity_level_valid | abnormal_alert_valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 7326 | 0 | 4266 | 5100 | 4266 |
| val | 1293 | 0 | 753 | 900 | 753 |
| test | 1381 | 0 | 1381 | 0 | 1381 |
| all | 10000 | 0 | 6400 | 6000 | 6400 |

test split 是 held-out `watermelon_disease_recognition`，因此 masked evaluation 只评估 `health_level` 和 `abnormal_alert`。`growth_stage` 和 `maturity_level` 在 test 上有效样本数为 0，不再输出看似很高但无意义的成熟度指标。

## 与 exp005_source_holdout 的指标对比

| 任务 | exp005 evaluated | exp005 Accuracy | exp005 Macro F1 | exp007 evaluated | exp007 Accuracy | exp007 Macro F1 | 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| growth_stage | 1381 | 0.8313 | 0.5574 | 0 | 0.0000 | 0.0000 | exp007 mask 后不评估 |
| health_level | 1381 | 0.3773 | 0.2254 | 1381 | 0.5373 | 0.2465 | Accuracy +0.1600，Macro F1 +0.0211 |
| maturity_level | 1381 | 1.0000 | 0.2500 | 0 | 0.0000 | 0.0000 | exp007 mask 后不评估；exp005 的 1.0 来自全为 maturity=0 的占位标签 |
| abnormal_alert | 1381 | 0.6705 | 0.5839 | 1381 | 0.7473 | 0.5472 | Accuracy +0.0768，Macro F1 -0.0367 |

## health_level 和 abnormal_alert 是否改善

`health_level` 有明确改善：held-out source 上 Accuracy 从 0.3773 提升到 0.5373，Macro F1 从 0.2254 提升到 0.2465。说明把非健康任务来源的弱健康标签 mask 掉后，健康等级的跨来源泛化更稳。

`abnormal_alert` 的结果是部分改善：Accuracy 从 0.6705 提升到 0.7473，正类 F1 从 0.7737 提升到 0.8482，漏报显著减少。但 Macro F1 从 0.5839 降到 0.5472，原因是模型更偏向预测异常，负类识别变差。

## 误报和漏报变化

| 指标 | exp005 | exp007 | 变化 |
| --- | ---: | ---: | ---: |
| false positive | 218 | 309 | +91 |
| false negative | 237 | 40 | -197 |
| abnormal positive precision | 0.7811 | 0.7593 | -0.0218 |
| abnormal positive recall | 0.7665 | 0.9606 | +0.1941 |
| abnormal positive F1 | 0.7737 | 0.8482 | +0.0745 |

对监测系统来说，漏报下降是重要收益，因为异常漏报比误报更危险。但误报增加也需要后续通过阈值、类别权重或真实大棚验证来控制。

## 结论：task mask 是否有效

task mask 有效。它没有让所有指标同时变好，但修正了最关键的问题：不再把与原始数据集任务无关的弱标签作为强监督信号。exp007 在 held-out `watermelon_disease_recognition` 上改善了 `health_level`，并显著降低 `abnormal_alert` 漏报；同时也让 `maturity_level=0` 这类占位标签不再参与 test 指标，避免产生虚假的 100% 成熟度结论。

当前不足是异常预警更偏向正类，误报增加；`growth_stage` 完全没有强监督样本，暂时无法训练和评估为可靠任务。

## 是否需要真实大棚数据微调

需要。`dataset_fused` 适合作为弱监督预训练数据，但最终监测系统仍需要真实大棚、目标相机、目标品种和人工审核标签微调。尤其是 `growth_stage` 必须依赖真实生育期标注；`health_level` 和 `abnormal_alert` 也需要真实大棚样本来校准误报/漏报平衡。

## 归档文件

- `checkpoints\exp007_masked_multitask_training\best_model.pth`
- `checkpoints\exp007_masked_multitask_training\last_model.pth`
- `checkpoints\exp007_masked_multitask_training\effective_config.yaml`
- `outputs\exp007_masked_multitask_training\evaluation_report.json`
- `outputs\exp007_masked_multitask_training\predictions.csv`
- `outputs\exp007_masked_multitask_training\confusion_matrix.csv`
- `experiments\exp007_masked_multitask_training\train.log`
- `experiments\exp007_masked_multitask_training\train_command.txt`
- `experiments\exp007_masked_multitask_training\experiment_summary.md`
