"""
Training engine and optimizer tuning experiments for Vegetable Classification.

Supports:
- Standard 20+ epoch training with callbacks (checkpointing, LR scheduling, logging).
- Optimizer comparison: Adam vs standard SGD vs SGD with Momentum.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import time
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

from src.models import build_model_b


def get_optimizer(
    name: str = "adam",
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
) -> keras.optimizers.Optimizer:
    """
    Factory for optimizers.
    Supported:
    - 'adam': Adaptive Moment Estimation (lr default: 1e-3)
    - 'sgd': Standard Stochastic Gradient Descent (lr default: 1e-2, momentum=0.0)
    - 'sgd_momentum': SGD with Nesterov/Classical Momentum (lr default: 1e-2, momentum=0.9)
    """
    name_clean = name.lower()
    if name_clean == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif name_clean == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0)
    elif name_clean in ("sgd_momentum", "momentum"):
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer name: {name}")


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 20,
    optimizer: Optional[keras.optimizers.Optimizer] = None,
    loss: str = "categorical_crossentropy",
    model_checkpoint_path: Optional[Path | str] = None,
    csv_log_path: Optional[Path | str] = None,
    patience_es: int = 10,
    patience_lr: int = 4,
) -> Tuple[keras.callbacks.History, float]:
    """
    Train a Keras model with standard callbacks, logging, and epoch timing.

    Returns:
        (history, avg_time_per_epoch)
    """
    if optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )

    cb_list = []
    if model_checkpoint_path:
        checkpoint_dir = Path(model_checkpoint_path).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cb_list.append(
            callbacks.ModelCheckpoint(
                filepath=str(model_checkpoint_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            )
        )

    if csv_log_path:
        log_dir = Path(csv_log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        cb_list.append(callbacks.CSVLogger(str(csv_log_path)))

    cb_list.append(
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        )
    )

    cb_list.append(
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        )
    )

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
        verbose=1,
    )
    total_time = time.time() - start_time
    epochs_trained = len(history.history["loss"])
    avg_time_per_epoch = total_time / max(epochs_trained, 1)

    return history, avg_time_per_epoch


def run_optimizer_comparison(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 15,
    results_dir: Optional[Path | str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compare Adam vs Standard SGD vs SGD with Momentum using Model B.
    Saves and returns training histories.
    """
    configs = [
        {"name": "Adam", "opt": keras.optimizers.Adam(learning_rate=1e-3), "lr": 0.001},
        {"name": "SGD (Standard)", "opt": keras.optimizers.SGD(learning_rate=1e-2, momentum=0.0), "lr": 0.01},
        {"name": "SGD with Momentum", "opt": keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9), "lr": 0.01},
    ]

    histories = {}
    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        opt_name = cfg["name"]
        print(f"\n{'='*50}\nTraining Model B with {opt_name}\n{'='*50}")
        # Build fresh instance of Model B
        model = build_model_b()
        csv_path = results_dir / f"optimizer_{opt_name.replace(' ', '_').lower()}.csv" if results_dir else None
        
        hist, _ = train_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=epochs,
            optimizer=cfg["opt"],
            csv_log_path=csv_path,
        )
        df_hist = pd.DataFrame(hist.history)
        histories[opt_name] = df_hist

    return histories

