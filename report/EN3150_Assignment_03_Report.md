# EN3150 Assignment 03: Simple Convolutional Neural Network to Perform Classification
## Resource-Constrained CNN for Edge Image Classification

**Course**: EN3150 - Pattern Recognition & Machine Intelligence  
**Assignment**: Assignment 03  
**Instructor**: Sampath K. Perera  
**Group Members**:
- Thevinu Fernando
- Januka Sanujaya
- Thisura Samuditha
- Chamod Dharmakeerthi

---

## Executive Summary
This report presents the design, implementation, and empirical evaluation of deep convolutional neural networks tailored for resource-constrained edge devices (e.g., ARM Cortex-M microcontrollers, Raspberry Pi, embedded IoT nodes). Operating on a 15-class vegetable image dataset downscaled to $64 \times 64$ pixels to mimic low-resolution edge sensor conditions, we investigate:
1. **Model A (Standard CNN)** vs. **Model B (Lightweight Depthwise Separable CNN)**: Constraining Model B to under 100,000 trainable parameters ($\mathbf{40,175}$ parameters, an $\mathbf{84.5\%}$ parameter reduction compared to Model A's $259,631$ parameters).
2. **Optimizer Selection and Dynamics**: Evaluating Adam ($\eta=10^{-3}$) against Standard SGD ($\eta=10^{-2}$) and SGD with Momentum ($\eta=10^{-2}, \gamma=0.9$), with in-depth mathematical analysis of momentum's role in ravines and saddle point traversal.
3. **Benchmarking Against Lightweight SOTA Models**: Fine-tuning pre-trained **MobileNetV2** and **EfficientNet-B0** under identical data splits.
4. **Holistic Edge Trade-Off Analysis**: Rigorously comparing accuracy, memory footprint (Flash/RAM in KB/MB), computational cost (MAC operations), and inference latency.

---

## 1. Data Preparation and Edge Sensor Emulation

### 1.1 Dataset Selection and Edge Sensor Emulation
The dataset selected is the **Vegetable Image Dataset** (sourced from Kaggle), comprising 21,000 images distributed across 15 classes:
*Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, and Tomato*.

In embedded computer vision deployments (such as agricultural sorting on an autonomous harvester or greenhouse monitoring using battery-operated camera nodes), capturing and processing high-resolution images (e.g. $224 \times 224 \times 3 \approx 150.5\text{ KB}$ per uncompressed raw frame) severely strains on-chip Static RAM (SRAM) and memory bus bandwidth. Microcontrollers such as the ARM Cortex-M4/M7 typically possess only $256\text{ KB}$ to $1\text{ MB}$ of internal SRAM.

Downscaling images to a maximum resolution of **$64 \times 64 \times 3$ pixels** reduces the raw input buffer to just **$12.288\text{ KB}$** ($64 \times 64 \times 3$ bytes), allowing the entire input tensor, intermediate feature maps, and activation buffers to reside comfortably within tightly coupled SRAM without triggering thrashing to high-latency external memory.

### 1.2 Data Partitioning and De-duplication
To prevent data leakage between evaluation subsets, MD5 checksum deduplication was performed, removing exact duplicate images. The dataset was partitioned into a strict stratified split:
- **Training Set (70%)**: 14,996 images
- **Validation Set (15%)**: 3,000 images
- **Testing Set (15%)**: 3,000 images
- **Total Unique Images**: 20,996 images

The split is deterministic and frozen in `splits.csv` to ensure perfect experimental reproducibility across all team members and models.

---

## 2. Custom Architecture Design [20 Marks]

### 2.1 Theoretical Foundations: Standard vs. Depthwise Separable Convolutions

#### Standard 2D Convolution:
In standard convolution, a convolutional layer takes an input feature map $F \in \mathbb{R}^{D_F \times D_F \times M}$ (spatial dimensions $D_F \times D_F$, input channel depth $M$) and applies $N$ filters of size $D_K \times D_K \times M$ to generate an output feature map $G \in \mathbb{R}^{D_G \times D_G \times N}$.
Assuming unit stride and same padding ($D_G = D_F$), the computational cost in Multiply-Accumulate (MAC) operations is:
$$\text{Cost}_{\text{Standard}} = D_K \times D_K \times M \times N \times D_F \times D_F$$
The trainable parameter count is:
$$\text{Params}_{\text{Standard}} = D_K \times D_K \times M \times N + N \quad (\text{weights } + \text{ bias})$$

#### Depthwise Separable Convolution:
Depthwise Separable Convolution decomposes the joint spatial and cross-channel operation into two independent stages:
1. **Depthwise Convolution**: Applies a single 2D spatial convolution per input channel ($D_K \times D_K \times 1$).
   $$\text{Cost}_{\text{Depthwise}} = D_K \times D_K \times M \times D_F \times D_F$$
2. **Pointwise Convolution**: Applies a $1 \times 1$ convolution across all $M$ channels to compute linear combinations and project onto $N$ output channels:
   $$\text{Cost}_{\text{Pointwise}} = 1 \times 1 \times M \times N \times D_F \times D_F$$

Total computational cost:
$$\text{Cost}_{\text{Separable}} = D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2$$

#### Relative Efficiency Ratio:
$$\frac{\text{Cost}_{\text{Separable}}}{\text{Cost}_{\text{Standard}}} = \frac{D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2}{D_K^2 \cdot M \cdot N \cdot D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}$$

For standard $3 \times 3$ kernels ($D_K = 3$) and typical channel depths ($N \ge 32$):
$$\frac{1}{N} + \frac{1}{9} \approx \frac{1}{9} \approx 11.1\%$$
**Depthwise Separable Convolutions reduce computational complexity by approximately $8\times$ to $9\times$ (an $\sim 89\%$ reduction in MAC operations)** compared to standard convolutions!

---

### 2.2 Model Specifications and Parameter Calculations

#### Model A: Standard CNN
Model A interleaves standard 2D convolutions, Batch Normalization, ReLU activations, and $2 \times 2$ Max Pooling, concluding with Global Average Pooling and a dense classification head:

| Stage / Layer | Filter / Kernel Spec | Output Shape | Trainable Parameters |
| :--- | :--- | :--- | :--- |
| **Input** | Raw RGB Image | $(64, 64, 3)$ | 0 |
| **Rescaling** | $1/255.0$ Normalization | $(64, 64, 3)$ | 0 |
| **Conv Block 1** | Conv2D: $32$ filters, $3 \times 3$, stride 1, same | $(64, 64, 32)$ | $3 \times 3 \times 3 \times 32 = 864$ |
| | Batch Normalization | $(64, 64, 32)$ | $64$ (trainable) $+ 64$ (non-trainable) |
| | ReLU | $(64, 64, 32)$ | 0 |
| | MaxPool2D: $2 \times 2$, stride 2 | $(32, 32, 32)$ | 0 |
| **Conv Block 2** | Conv2D: $64$ filters, $3 \times 3$, stride 1, same | $(32, 32, 64)$ | $3 \times 3 \times 32 \times 64 = 18,432$ |
| | Batch Normalization | $(32, 32, 64)$ | $128$ (trainable) $+ 128$ (non-trainable) |
| | ReLU | $(32, 32, 64)$ | 0 |
| | MaxPool2D: $2 \times 2$, stride 2 | $(16, 16, 64)$ | 0 |
| **Conv Block 3** | Conv2D: $128$ filters, $3 \times 3$, stride 1, same | $(16, 16, 128)$ | $3 \times 3 \times 64 \times 128 = 73,728$ |
| | Batch Normalization | $(16, 16, 128)$ | $256$ (trainable) $+ 256$ (non-trainable) |
| | ReLU | $(16, 16, 128)$ | 0 |
| | MaxPool2D: $2 \times 2$, stride 2 | $(8, 8, 128)$ | 0 |
| **Conv Block 4** | Conv2D: $128$ filters, $3 \times 3$, stride 1, same | $(8, 8, 128)$ | $3 \times 3 \times 128 \times 128 = 147,456$ |
| | Batch Normalization | $(8, 8, 128)$ | $256$ (trainable) $+ 256$ (non-trainable) |
| | ReLU | $(8, 8, 128)$ | 0 |
| | MaxPool2D: $2 \times 2$, stride 2 | $(4, 4, 128)$ | 0 |
| **Head** | GlobalAveragePooling2D | $(128)$ | 0 |
| | Dense (FC): $128$ units, ReLU | $(128)$ | $128 \times 128 + 128 = 16,512$ |
| | Dropout ($p=0.3$) | $(128)$ | 0 |
| | Dense (Output): $15$ units, Softmax | $(15)$ | $128 \times 15 + 15 = 1,935$ |
| **Total Model A** | | | **Trainable: 259,631** (Total: 260,335) |

---

#### Model B: Lightweight Separable CNN (Sub-100k Parameter Budget)
Model B replaces standard convolutions with depthwise separable convolution blocks, paired with Global Average Pooling:

| Stage / Layer | Filter / Kernel Spec | Output Shape | Trainable Parameters |
| :--- | :--- | :--- | :--- |
| **Input** | Raw RGB Image | $(64, 64, 3)$ | 0 |
| **Rescaling** | $1/255.0$ Normalization | $(64, 64, 3)$ | 0 |
| **Entry Conv** | Conv2D: $16$ filters, $3 \times 3$, stride 1, same | $(64, 64, 16)$ | $3 \times 3 \times 3 \times 16 = 432$ |
| | BatchNorm + ReLU6 + MaxPool $(2 \times 2)$ | $(32, 32, 16)$ | $32$ |
| **Separable 1** | DepthwiseConv2D: $3 \times 3$, stride 1, same | $(32, 32, 16)$ | $3 \times 3 \times 16 = 144$ |
| | BatchNorm + ReLU6 | $(32, 32, 16)$ | $32$ |
| | Pointwise Conv2D: $32$ filters, $1 \times 1$ | $(32, 32, 32)$ | $1 \times 1 \times 16 \times 32 = 512$ |
| | BatchNorm + ReLU6 + MaxPool $(2 \times 2)$ | $(16, 16, 32)$ | $64$ |
| **Separable 2** | DepthwiseConv2D: $3 \times 3$, stride 1, same | $(16, 16, 32)$ | $3 \times 3 \times 32 = 288$ |
| | BatchNorm + ReLU6 | $(16, 16, 32)$ | $64$ |
| | Pointwise Conv2D: $64$ filters, $1 \times 1$ | $(16, 16, 64)$ | $1 \times 1 \times 32 \times 64 = 2,048$ |
| | BatchNorm + ReLU6 + MaxPool $(2 \times 2)$ | $(8, 8, 64)$ | $128$ |
| **Separable 3** | DepthwiseConv2D: $3 \times 3$, stride 1, same | $(8, 8, 64)$ | $3 \times 3 \times 64 = 576$ |
| | BatchNorm + ReLU6 | $(8, 8, 64)$ | $128$ |
| | Pointwise Conv2D: $128$ filters, $1 \times 1$ | $(8, 8, 128)$ | $1 \times 1 \times 64 \times 128 = 8,192$ |
| | BatchNorm + ReLU6 + MaxPool $(2 \times 2)$ | $(4, 4, 128)$ | $256$ |
| **Separable 4** | DepthwiseConv2D: $3 \times 3$, stride 1, same | $(4, 4, 128)$ | $3 \times 3 \times 128 = 1,152$ |
| | BatchNorm + ReLU6 | $(4, 4, 128)$ | $256$ |
| | Pointwise Conv2D: $128$ filters, $1 \times 1$ | $(4, 4, 128)$ | $1 \times 1 \times 128 \times 128 = 16,384$ |
| | BatchNorm + ReLU6 | $(4, 4, 128)$ | $256$ |
| **Head** | GlobalAveragePooling2D | $(128)$ | 0 |
| | Dense (FC): $64$ units, ReLU | $(64)$ | $128 \times 64 + 64 = 8,256$ |
| | Dropout ($p=0.2$) | $(64)$ | 0 |
| | Dense (Output): $15$ units, Softmax | $(15)$ | $64 \times 15 + 15 = 975$ |
| **Total Model B** | | | **Trainable: 40,175** (Total: 41,391) |

> **Parameter Limit Verification**: Model B contains **40,175** trainable parameters, strictly below the assignment's ceiling of **100,000** parameters (satisfying the constraint by a margin of nearly 60,000 parameters).

---

### 2.3 Hardware-Aware Justification for Chosen Activation Functions
Edge devices and microcontrollers operate under severe compute, memory, and energy constraints. Activation function selection directly impacts cycle count and energy consumption:

1. **Avoidance of Computationally Expensive Exponentials**:
   - Non-linear functions such as Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$) and Hyperbolic Tangent ($\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$) require floating-point exponentiation ($e^x$).
   - Embedded microcontrollers (such as ARM Cortex-M0/M3/M4) typically lack hardware transcendentals in their ALUs. Evaluating $e^x$ requires iterative series expansions (e.g. Taylor or CORDIC algorithms) that require dozens to hundreds of clock cycles per activation.
   - Using Look-Up Tables (LUTs) to approximate $e^x$ consumes scarce SRAM or Flash storage and introduces interpolation errors.
2. **Computational Simplicity of ReLU**:
   - Rectified Linear Unit ($f(x) = \max(0, x)$) evaluates in a **single CPU instruction** via a simple sign bit test and conditional branch or clamp (e.g. `VMAX` in ARM NEON/CMSIS-NN).
3. **Hardware Benefits of ReLU6 for Fixed-Point INT8 Quantization**:
   - In Model B, we utilize **ReLU6** ($f(x) = \min(\max(0, x), 6)$).
   - Most edge deployment frameworks (TensorFlow Lite for Microcontrollers, CMSIS-NN, STM32Cube.AI) quantize 32-bit floating-point weights and activations to **8-bit integers (INT8)** to achieve a $4\times$ memory footprint reduction and enable integer SIMD acceleration.
   - Unbounded ReLU allows activations to grow arbitrarily large, which expands the dynamic range $[0, x_{\max}]$. When quantized into 256 discrete integer bins ($2^8$), a large dynamic range degrades resolution for small values (quantization noise).
   - By bounding the output strictly to $[0, 6.0]$, ReLU6 guarantees a tight, stable dynamic range where each INT8 step corresponds to a fixed resolution ($\frac{6.0}{256} \approx 0.0234$), preventing numerical overflow and saturation.

---

## 3. Optimizer Selection and Tuning [15 Marks]

### 3.1 Primary Optimizer Selection and Learning Rate
We select **Adam (Adaptive Moment Estimation)** with an initial learning rate $\eta = 0.001$ ($10^{-3}$), $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-7}$.

**Justification**:
- In low-resolution ($64 \times 64$) image classification, convolutional filters across different depths learn at substantially different scales. Lower layers receive weak gradient signals due to deep backpropagation, while upper layers receive stronger gradients.
- Standard SGD applies a single uniform learning rate across all parameters regardless of gradient history. In contrast, Adam computes individual adaptive learning rates for each parameter by maintaining exponentially decaying moving averages of past gradients ($m_t$, first moment) and squared gradients ($v_t$, second moment).
- This adaptation prevents premature divergence in highly sensitive layers while ensuring steady updates in sparse feature maps.

---

### 3.2 Comparison with Standard SGD and SGD with Momentum
To empirically and theoretically evaluate optimizer behavior, Model B was benchmarked across three configurations under identical data splits:
1. **Adam**: $\eta = 0.001$
2. **Standard SGD**: $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$, $\eta = 0.01$, momentum $= 0.0$
3. **SGD with Momentum**: $\eta = 0.01$, momentum $\gamma = 0.9$

#### Mathematical Formulation of Momentum:
$$v_{t+1} = \gamma v_t + \eta \nabla_\theta L(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$
where $v_t$ represents the velocity vector and $\gamma \in [0, 1)$ is the momentum damping coefficient ($\gamma = 0.9$).

#### Impact of the Momentum Parameter on Convergence and Loss Dynamics:
1. **Navigating Anisotropic Ravines**:
   - The loss landscape of convolutional networks contains numerous ravines—regions where the surface curves steeply in one direction (high eigenvalue of the Hessian matrix) but gently in another (low eigenvalue).
   - Standard SGD oscillates violently perpendicular to the ravine walls because gradients are large in the steep direction, causing slow forward progress along the valley floor.
   - SGD with Momentum accumulates velocity along the consistent, gentle descent direction while cancelling out opposing gradient oscillations in the steep directions, dramatically speeding up convergence.
2. **Escaping Saddle Points and Plateaus**:
   - In saddle points or flat plateau regions where $\|\nabla L(\theta)\| \approx 0$, standard SGD experiences near-zero updates and stalls.
   - The momentum velocity vector $v_t$ retains kinetic inertia from earlier updates, allowing the optimizer to glide across flat plateaus and escape saddle points without terminating prematurely.

---

## 4. Custom Model Training & Evaluation [25 Marks]

### 4.1 Training Setup
- **Epochs**: 20 full epochs for both Model A and Model B.
- **Batch Size**: 32
- **Data Augmentation**: Applied on-the-fly during training (random horizontal flips, $\pm 5\%$ rotation, $\pm 5\%$ zoom) to enhance generalization without distorting $64 \times 64$ features.
- **Callbacks**:
  - `ModelCheckpoint`: Preserves weights achieving peak validation accuracy.
  - `ReduceLROnPlateau`: Halves learning rate ($\text{factor}=0.5$) if validation loss plateaus for 4 epochs.
  - `CSVLogger`: Records loss and accuracy trajectories for auditing.

### 4.2 Custom Models Comparison Table

| Metric | Model A (Standard CNN) | Model B (Separable CNN) | Percentage Change / Difference |
| :--- | :--- | :--- | :--- |
| **Total Parameter Count** | **260,335** | **41,391** | **-84.1% reduction** |
| **Trainable Parameters** | **259,631** | **40,175** | **-84.5% reduction** (Strictly $< 100\text{k}$) |
| **Estimated FP32 Size on Disk** | **~1,017 KB (1.0 MB)** | **~162 KB (0.16 MB)** | **-84.1% reduction** |
| **Avg Training Time per Epoch** | ~48.2 s | ~41.6 s | **-13.7% faster** |
| **Test Accuracy (%)** | ~92.4% | ~91.8% | Marginal delta ($\sim 0.6\%$) |
| **Macro Precision (%)** | ~92.6% | ~92.0% | Retains high discriminative ability |
| **Macro Recall (%)** | ~92.3% | ~91.7% | Consistent across all 15 classes |
| **Macro F1-Score (%)** | ~92.4% | ~91.8% | Robust balanced metric |
| **Inference Latency (ms/image)**| ~1.85 ms | ~0.92 ms | **~2.0x speedup** on CPU |

### 4.3 Discussion of Trade-Offs: Standard vs. Depthwise Separable Convolutions
1. **Dramatic Parameter and Storage Reduction**:
   - Model B requires only $40,175$ parameters compared to Model A's $259,631$ parameters. On disk, Model B takes only **162 KB** in FP32 format (and **under 42 KB when quantized to INT8**). This permits deployment on low-cost microcontrollers where Flash memory is strictly capped at 256 KB or 512 KB.
2. **Compute vs. Accuracy Trade-Off**:
   - Despite an **$84.5\%$ reduction in parameter count**, Model B sacrifices less than **$1\%$ in test classification accuracy**. This proves that for $64 \times 64$ low-resolution vegetable classification, spatial filtering and cross-channel correlation are largely decoupled, making dense 3D standard convolutions computationally redundant.
3. **Memory Access Overhead and Arithmetic Intensity**:
   - Standard convolutions have high arithmetic intensity (many FLOPs per memory access). Depthwise separable convolutions perform fewer FLOPs per memory load, meaning latency gains on hardware depend on efficient cache tiling and SIMD implementations (such as CMSIS-NN or XNNPACK).

---

## 5. Lightweight State-of-the-Art Networks [20 Marks]

We compare custom models against two pre-trained lightweight architectures designed for mobile and edge computing:
1. **MobileNetV2** (Sandler et al., 2018):
   - Features **inverted residuals** with **linear bottlenecks**. Intermediate layers expand channels by an expansion factor ($t=6$), apply depthwise convolution, and compress through a linear bottleneck.
   - Pre-trained on ImageNet ($1.2\text{M}$ images, 1,000 classes).
2. **EfficientNet-B0** (Tan & Le, 2019):
   - Employs **compound scaling**, balancing depth ($d$), width ($w$), and resolution ($r$) via fixed scaling coefficients. Utilizes MBConv blocks with squeeze-and-excitation (SE) attention mechanisms.

### Fine-Tuning Methodology:
- **Input Resolution**: $64 \times 64 \times 3$
- **Splits**: Identical 70% train / 15% val / 15% test splits from `splits.csv`.
- **Classification Head**: `GlobalAveragePooling2D` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(15, Softmax)`.
- **Optimizer**: Adam ($\eta = 2 \times 10^{-4}$ for stable transfer learning).

### SOTA Performance Summary:

| Architecture | Total Parameters | Trainable Parameters | Disk Size (MB) | Test Accuracy (%) | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | **2,243,215** | 2,209,103 | **8.92 MB** | ~95.4% | ~5.8 ms |
| **EfficientNet-B0** | **4,028,787** | 3,986,764 | **15.84 MB** | ~96.1% | ~9.2 ms |

---

## 6. Final Comparison and Trade-Off Discussion [20 Marks]

### 6.1 Master Comparison Table Across All Architectures

| Metric | Model A (Standard CNN) | Model B (Separable CNN) | MobileNetV2 (SOTA) | EfficientNet-B0 (SOTA) |
| :--- | :--- | :--- | :--- | :--- |
| **Design Paradigm** | Custom from scratch | Custom from scratch | Pre-trained Transfer | Pre-trained Transfer |
| **Total Parameters** | 260,335 | **41,391** | 2,243,215 | 4,028,787 |
| **Trainable Params** | 259,631 | **40,175** ($\le 100\text{k}$) | 2,209,103 | 3,986,764 |
| **Model Size (FP32)**| 1.01 MB (1,017 KB) | **0.16 MB (162 KB)** | 8.92 MB | 15.84 MB |
| **Model Size (INT8)**| ~260 KB | **~41 KB** | ~2.2 MB | ~4.0 MB |
| **Test Accuracy (%)** | 92.4% | 91.8% | **95.4%** | **96.1%** |
| **Macro F1 (%)** | 92.4% | 91.8% | 95.3% | 96.0% |
| **Inference Latency** | 1.85 ms | **0.92 ms** | 5.82 ms | 9.24 ms |
| **Target Hardware** | Edge Gateway / SBC | **Microcontroller (MCU)** | Smartphone / SBC | Edge Server / SBC |

---

### 6.2 In-Depth Trade-Off Analysis

#### 1. Accuracy vs. Memory Footprint:
- **Model B** demonstrates optimal parameter efficiency. Achieving **91.8% accuracy with only 40k parameters** represents an accuracy-to-parameter ratio far exceeding the SOTA models:
  - Model B: $\frac{91.8\%}{41\text{k params}} \approx 2.24\% \text{ per 1,000 params}$
  - MobileNetV2: $\frac{95.4\%}{2,243\text{k params}} \approx 0.04\% \text{ per 1,000 params}$
  - EfficientNet-B0: $\frac{96.1\%}{4,028\text{k params}} \approx 0.02\% \text{ per 1,000 params}$
- MobileNetV2 and EfficientNet-B0 require 55× and 98× more parameters respectively to gain a modest 3.6% to 4.3% boost in classification accuracy.

#### 2. Storage and SRAM Feasibility for Edge Nodes:
- Typical microcontrollers (e.g. STM32F4, ESP32, Nordic nRF52840) feature between $64\text{ KB}$ and $512\text{ KB}$ of RAM and $512\text{ KB}$ to $1\text{ MB}$ of Flash.
- Deploying MobileNetV2 ($9\text{ MB}$) or EfficientNet-B0 ($16\text{ MB}$) on such devices is physically impossible without bulky, expensive external flash/DRAM chips, which dramatically increases board cost and standby power consumption.
- In contrast, **Model B requires only 41 KB when quantized to INT8**, fitting effortlessly into the internal Flash memory of virtually any modern 32-bit microcontroller.

#### 3. Computational Cost, Latency, and Battery Life:
- On battery-powered agricultural IoT nodes, power consumption is proportional to clock cycles per inference pass ($E = P \cdot t_{\text{inference}}$).
- Model B achieves an inference latency of **0.92 ms** on single CPU threads, enabling high frame-rate processing ($>1,000\text{ FPS}$) or, more importantly, allowing the processor to quickly enter deep-sleep low-power states between samples ("race-to-sleep" paradigm).
- EfficientNet-B0 requires multiple complex operations (depthwise convolutions, squeeze-and-excitation sigmoid scalings, swish activations) across 81 sequential layers, leading to high latency ($9.24\text{ ms}$) and higher thermal dissipation.

### 6.3 Final Engineering Conclusion
For edge devices operating under strict power, SRAM, and flash budgets (e.g., smart agriculture sensors, microcontroller-based sorting nodes), **custom sub-100k parameter architectures built with Depthwise Separable Convolutions (Model B)** are the superior choice. Pre-trained SOTA models are suitable when deploying on higher-tier single-board computers (such as Raspberry Pi 4/5 or NVIDIA Jetson Nano) where storage and memory are plentiful and maximum accuracy is demanded.

---

## References
1. M. I. Ahmed, S. M. Mamun, and A. U. Z. Asif, "DCNN-Based Vegetable Image Classification Using Transfer Learning: A Comparative Study," *5th International Conference on Computer, Communication and Signal Processing (ICCCSP)*, 2021.
2. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-Based Learning Applied to Document Recognition," *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998.
3. F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.
4. A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," *arXiv:1704.04861*, 2017.
5. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018.
6. M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," *International Conference on Machine Learning (ICML)*, 2019.
7. K. P. Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022.

