| Model | Architecture | Total Parameters | Trainable Parameters | Model Size on Disk (KB) | Training Time / Epoch (s) | Test Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Inference Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model A (Standard CNN) | Standard Conv2D + MaxPool | 260335 | 259631 | 3135.53 | 54.37 | 97.57 | 97.75 | 97.57 | 17.67 |
| Model B (Separable CNN) | Depthwise Separable Conv2D | 41391 | 40175 | 628.02 | 28.6 | 97.7 | 97.75 | 97.7 | 31.82 |
