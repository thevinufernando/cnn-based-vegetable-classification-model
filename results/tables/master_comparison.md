| Model | Type | Total Parameters | Trainable Params | Disk Size (MB) | Test Accuracy (%) | Macro F1 (%) | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model A (Standard CNN) | From Scratch | 260335 | 259631 | 3.062 | 97.57 | 97.59 | 17.67 |
| Model B (Separable CNN) | From Scratch (<100k) | 41391 | 40175 | 0.613 | 97.7 | 97.7 | 31.82 |
| MobileNetV2 | Pre-trained SOTA | 2277199 | 2243087 | 26.478 | 99.6 | 99.6 | 175.36 |
| EfficientNet-B0 | Pre-trained SOTA | 4068786 | 4026763 | 47.204 | 99.53 | 99.53 | 354.14 |
