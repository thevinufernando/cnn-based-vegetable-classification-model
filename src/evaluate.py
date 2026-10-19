"""
Evaluation module for Vegetable Classification models.

Computes:
- Test Accuracy, Precision (Macro & Weighted), Recall (Macro & Weighted), F1-Score
- Confusion Matrix (raw counts and normalized)
- Per-class precision/recall classification report
- Model disk size (KB and MB)
- Edge inference latency benchmarks (ms per image on CPU)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import tensorflow as tf
from tensorflow import keras


def evaluate_model(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Run full test evaluation on test_ds.

    Returns:
        dict containing metrics, confusion matrix, predictions, and report.
    """
    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs_list.append(preds)
        # Handle categorical (one-hot) or sparse labels
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            y_true_list.append(np.argmax(labels.numpy(), axis=-1))
        else:
            y_true_list.append(labels.numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "model_name": model.name,
        "test_accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
        "classification_report": report_dict,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def measure_model_disk_size(model_path: Path | str) -> Dict[str, float]:
    """Measure exact file size on disk for saved model."""
    path = Path(model_path)
    if not path.exists():
        return {"size_kb": 0.0, "size_mb": 0.0}

    if path.is_file():
        size_bytes = os.path.getsize(path)
    else:
        size_bytes = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())

    return {
        "size_kb": round(size_bytes / 1024.0, 2),
        "size_mb": round(size_bytes / (1024.0 * 1024.0), 3),
    }


def benchmark_inference_latency(
    model: keras.Model,
    input_shape: tuple = (1, 64, 64, 3),
    num_runs: int = 100,
    warmup_runs: int = 20,
) -> Dict[str, float]:
    """
    Benchmark edge inference latency per image (in milliseconds) on CPU.
    """
    dummy_input = tf.random.normal(input_shape)

    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input, training=False)

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model(dummy_input, training=False)
        times.append(time.perf_counter() - t0)

    mean_latency_ms = float(np.mean(times) * 1000.0)
    std_latency_ms = float(np.std(times) * 1000.0)
    throughput_fps = float(1000.0 / mean_latency_ms) if mean_latency_ms > 0 else 0.0

    return {
        "latency_mean_ms": round(mean_latency_ms, 2),
        "latency_std_ms": round(std_latency_ms, 2),
        "throughput_fps": round(throughput_fps, 1),
    }

