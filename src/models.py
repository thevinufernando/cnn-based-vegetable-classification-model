
# Model architectures for Vegetable Classification on Edge Devices.

# Implements:
# 1. Model A: Standard CNN with interleaved Conv2D and MaxPooling2D.
# 2. Model B: Lightweight CNN using Depthwise Separable Convolutions.
# 3. MobileNetV2: Fine-tuned lightweight SOTA baseline.
# 4. EfficientNet-B0: Fine-tuned compound-scaled lightweight SOTA baseline.


from typing import Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model_a(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 15,
) -> keras.Model:

    # Model A: Standard CNN for Edge Classification.
    # Interleaves standard Conv2D layers with Batch Normalization, ReLU activations,
    # and MaxPooling2D layers.

    inputs = layers.Input(shape=input_shape, name="input_image")
    # Rescale pixel values from [0, 255] to [0.0, 1.0]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)  # 32x32

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)  # 16x16

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)  # 8x8

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)  # 4x4

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Model_A_Standard_CNN")
    return model


def build_model_b(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 15,
) -> keras.Model:
    
    # Model B: Ultra-lightweight CNN using Depthwise Separable Convolutions.
    # Strict constraint: Total trainable parameters MUST NOT EXCEED 100,000.

    # Depthwise Separable Convolution decomposes standard convolution into:
    # 1. Depthwise convolution: spatial filtering per channel (kernel_size * kernel_size * C_in)
    # 2. Pointwise convolution: 1x1 cross-channel combination (1 * 1 * C_in * C_out)
    # This drastically reduces multiply-accumulate (MAC) operations by ~8x-9x for 3x3 kernels.

    
  
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Initial standard Conv2D entry layer (extract low-level spatial edges)
    x = layers.Conv2D(16, (3, 3), padding="same", use_bias=False, name="entry_conv")(x)
    x = layers.BatchNormalization(name="entry_bn")(x)
    x = layers.ReLU(max_value=6.0, name="entry_relu6")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)  # 32x32

    # Separable Block 1: 16 -> 32 channels
    x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False, name="dw1")(x)
    x = layers.BatchNormalization(name="dw_bn1")(x)
    x = layers.ReLU(max_value=6.0, name="dw_relu1")(x)
    x = layers.Conv2D(32, (1, 1), padding="same", use_bias=False, name="pw1")(x)
    x = layers.BatchNormalization(name="pw_bn1")(x)
    x = layers.ReLU(max_value=6.0, name="pw_relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)  # 16x16

    # Separable Block 2: 32 -> 64 channels
    x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False, name="dw2")(x)
    x = layers.BatchNormalization(name="dw_bn2")(x)
    x = layers.ReLU(max_value=6.0, name="dw_relu2")(x)
    x = layers.Conv2D(64, (1, 1), padding="same", use_bias=False, name="pw2")(x)
    x = layers.BatchNormalization(name="pw_bn2")(x)
    x = layers.ReLU(max_value=6.0, name="pw_relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)  # 8x8

    # Separable Block 3: 64 -> 128 channels
    x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False, name="dw3")(x)
    x = layers.BatchNormalization(name="dw_bn3")(x)
    x = layers.ReLU(max_value=6.0, name="dw_relu3")(x)
    x = layers.Conv2D(128, (1, 1), padding="same", use_bias=False, name="pw3")(x)
    x = layers.BatchNormalization(name="pw_bn3")(x)
    x = layers.ReLU(max_value=6.0, name="pw_relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)  # 4x4

    # Separable Block 4: 128 -> 128 channels
    x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False, name="dw4")(x)
    x = layers.BatchNormalization(name="dw_bn4")(x)
    x = layers.ReLU(max_value=6.0, name="dw_relu4")(x)
    x = layers.Conv2D(128, (1, 1), padding="same", use_bias=False, name="pw4")(x)
    x = layers.BatchNormalization(name="pw_bn4")(x)
    x = layers.ReLU(max_value=6.0, name="pw_relu4")(x)

    # Classification Head (Global Average Pooling minimizes parameters)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Model_B_Separable_CNN")
    return model


def build_mobilenet_v2(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 15,
    freeze_base: bool = False,
) -> keras.Model:
    """
    Fine-tuned MobileNetV2 architecture for 64x64 edge classification.
    Uses pre-trained ImageNet weights.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    # MobileNetV2 expects inputs in [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_rescaling")(inputs)

    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2")
    return model


def build_efficientnet_b0(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 15,
    freeze_base: bool = False,
) -> keras.Model:
    """
    Fine-tuned EfficientNet-B0 architecture for 64x64 edge classification.
    Uses pre-trained ImageNet weights.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    # EfficientNet has built-in preprocessing expecting [0, 255]
    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNet_B0")
    return model


def get_model_stats(model: keras.Model) -> Dict[str, Any]:
    """Calculate parameter counts, memory footprint, and layer breakdown."""
    trainable_params = int(
        np.sum([np.prod(v.shape) for v in model.trainable_weights])
    )
    non_trainable_params = int(
        np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    )
    total_params = trainable_params + non_trainable_params

    # In FP32, each parameter occupies 4 bytes
    param_size_bytes = total_params * 4
    param_size_kb = param_size_bytes / 1024.0
    param_size_mb = param_size_bytes / (1024.0 * 1024.0)

    return {
        "name": model.name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "size_kb": round(param_size_kb, 2),
        "size_mb": round(param_size_mb, 2),
    }

