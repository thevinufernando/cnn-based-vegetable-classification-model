| Model | Architecture Type | Total Parameters | Trainable Parameters | Disk Size (KB) | Disk Size (MB) | Avg Time/Epoch (s) | Test Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model A (Standard CNN) | Standard Conv2D + MaxPool | 260335 | 259631 | 3135.53 | 3.062 | 54.37 | 97.57 | 97.75 | 97.57 | 97.59 | 15.32 |
| Model B (Separable CNN) | Depthwise Separable Conv2D | 41391 | 40175 | 628.02 | 0.613 | 28.6 | 97.7 | 97.75 | 97.7 | 97.7 | 25.24 |
| MobileNetV2 (SOTA) | Inverted Residuals + Linear Bottlenecks | 2277199 | 2243087 | 27113.04 | 26.478 | 166.05 | 99.6 | 99.6 | 99.6 | 99.6 | 156.17 |
| EfficientNet-B0 (SOTA) | MBConv (Compound Scaling) | 4068786 | 4026763 | 48337.18 | 47.204 | 237.16 | 99.53 | 99.54 | 99.53 | 99.53 | 278.74 |
