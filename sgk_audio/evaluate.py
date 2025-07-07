from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def mse(y_true: np.ndarray, y_pred: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2, axis=axis)


def snr(y_true: np.ndarray, y_pred: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Signal to noise ratio in dB."""
    noise = y_true - y_pred
    power_signal = np.mean(y_true ** 2, axis=axis)
    power_noise = np.mean(noise ** 2, axis=axis) + 1e-12
    return 10 * np.log10(power_signal / power_noise)


def plot_dictionary_atoms(D: np.ndarray, n_cols: int = 10, figsize=(12, 6)):
    """Visualize dictionary atoms as images (heatmaps)."""
    n_atoms = D.shape[1]
    n_rows = int(np.ceil(n_atoms / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    for k in range(n_atoms):
        im = axes[k].imshow(D[:, k][:, None], aspect="auto", origin="lower", cmap="viridis")
        axes[k].set_title(f"Atom {k}")
        axes[k].axis("off")
    for k in range(n_atoms, len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    plt.show()


def plot_reconstruction_comparison(original: np.ndarray, reconstructed: np.ndarray, sample_idx: int = 0):
    """Plot original vs reconstructed MFCC vectors for a single frame."""
    plt.figure(figsize=(8, 4))
    plt.plot(original[:, sample_idx], label="Original")
    plt.plot(reconstructed[:, sample_idx], label="Reconstructed", linestyle="--")
    plt.title("MFCC Frame Reconstruction")
    plt.xlabel("Coefficient")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_histogram(errors: np.ndarray, title: str = "Reconstruction Error (MSE)"):
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("MSE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()