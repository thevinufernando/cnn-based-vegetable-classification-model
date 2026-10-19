"""
Dataset loading and preprocessing module for Vegetable Classification.

Handles data loading from data/processed/{train, val, test},
applies edge-appropriate data augmentations, and configures tf.data pipelines.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_CLASSES = 15

CLASS_NAMES = [
    "Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli",
    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
]


def get_data_augmentation() -> keras.Sequential:
    """
    Data augmentation layers for edge robustness.
    Mild augmentation to avoid distorting small 64x64 vegetable features.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
        ],
        name="data_augmentation"
    )


def load_datasets(
    data_dir: Path | str,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    augment_train: bool = True,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Load train, validation, and test datasets from directory structure:
        <data_dir>/train/<class_name>/<images>
        <data_dir>/val/<class_name>/<images>
        <data_dir>/test/<class_name>/<images>

    Returns:
        (train_ds, val_ds, test_ds, class_names)
    """
    data_path = Path(data_dir)
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"

    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found at {train_path}")

    # Load raw image datasets with categorical labels (one-hot)
    train_ds = keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    )
    class_names = train_ds.class_names

    val_ds = keras.utils.image_dataset_from_directory(
        val_path,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    test_ds = keras.utils.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    # Apply data augmentation to training set if requested
    if augment_train:
        augmentation = get_data_augmentation()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

