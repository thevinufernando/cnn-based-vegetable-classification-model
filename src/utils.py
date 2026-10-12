"""
Utility functions for plotting, table generation, and visualization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_training_history(
    history_dict: Dict[str, Any] | pd.DataFrame,
    title: str = "Training and Validation Curves",
    save_path: Optional[Path | str] = None,
) -> None:
    """
    Plot loss and accuracy curves side by side.
    """
    if isinstance(history_dict, pd.DataFrame):
        hist = history_dict.to_dict(orient="list")
    elif hasattr(history_dict, "history"):
        hist = history_dict.history
    else:
        hist = history_dict

    epochs = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, hist["loss"], "b-o", label="Training Loss", linewidth=2, markersize=4)
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], "r--s", label="Validation Loss", linewidth=2, markersize=4)
    axes[0].set_title(f"{title} - Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy plot
    acc_key = "accuracy" if "accuracy" in hist else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in hist else "val_acc"

    if acc_key in hist:
        axes[1].plot(epochs, hist[acc_key], "b-o", label="Training Accuracy", linewidth=2, markersize=4)
    if val_acc_key in hist:
        axes[1].plot(epochs, hist[val_acc_key], "r--s", label="Validation Accuracy", linewidth=2, markersize=4)
    axes[1].set_title(f"{title} - Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    plt.close()


def plot_optimizer_comparison(
    histories: Dict[str, pd.DataFrame | Dict[str, list]],
    save_path: Optional[Path | str] = None,
) -> None:
    """
    Plot comparison of loss and validation loss across multiple optimizers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = {"Adam": "#1f77b4", "SGD (Standard)": "#d62728", "SGD with Momentum": "#2ca02c"}

    for name, hist in histories.items():
        if isinstance(hist, pd.DataFrame):
            data = hist.to_dict(orient="list")
        elif hasattr(hist, "history"):
            data = hist.history
        else:
            data = hist

        epochs = range(1, len(data["loss"]) + 1)
        c = colors.get(name, None)
        axes[0].plot(epochs, data["loss"], label=f"{name} (Train)", color=c, linewidth=2)
        if "val_loss" in data:
            axes[0].plot(epochs, data["val_loss"], linestyle="--", label=f"{name} (Val)", color=c, alpha=0.8)

        acc_key = "accuracy" if "accuracy" in data else "acc"
        val_acc_key = "val_accuracy" if "val_accuracy" in data else "val_acc"
        if val_acc_key in data:
            axes[1].plot(epochs, data[val_acc_key], label=f"{name} (Val Acc)", color=c, linewidth=2)

    axes[0].set_title("Optimizer Comparison - Loss Convergence", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Categorical Cross-Entropy Loss", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].set_title("Optimizer Comparison - Validation Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved optimizer comparison figure: {save_path}")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalized: bool = False,
    save_path: Optional[Path | str] = None,
) -> None:
    """
    Plot and save heatmap confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    fmt = ".2f" if normalized else "d"
    cmap = "Blues"

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Ground Truth Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix: {save_path}")
    plt.close()


def save_comparison_table(
    table_data: List[Dict[str, Any]],
    output_csv_path: Path | str,
    output_md_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Save model comparison table to CSV and Markdown.
    """
    df = pd.DataFrame(table_data)
    csv_p = Path(output_csv_path)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_p, index=False)
    print(f"Saved comparison CSV: {csv_p}")

    if output_md_path:
        md_p = Path(output_md_path)
        md_p.parent.mkdir(parents=True, exist_ok=True)
        headers = [str(col) for col in df.columns]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = [
            "| " + " | ".join(str(val) for val in row) + " |"
            for row in df.values
        ]
        md_table = "\n".join([header_line, separator_line] + rows) + "\n"
        with open(md_p, "w", encoding="utf-8") as f:
            f.write(md_table)
        print(f"Saved comparison Markdown: {md_p}")

    return df

