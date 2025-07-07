from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sgk_audio.audio_utils import create_dataset
from sgk_audio.sgk import SGK
from sgk_audio.evaluate import mse, snr, plot_dictionary_atoms, plot_reconstruction_comparison, plot_error_histogram


def parse_args():
    parser = argparse.ArgumentParser(description="Train SGK dictionary on chirp audio dataset and evaluate reconstruction.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing WAV files in subfolders (reference points)")
    parser.add_argument("--n_mfcc", type=int, default=20, help="Number of MFCC coefficients (feature dimension)")
    parser.add_argument("--n_atoms", type=int, default=40, help="Number of dictionary atoms (columns)")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of SGK iterations")
    parser.add_argument("--sparsity", type=int, default=8, help="Sparsity level (non-zero coefficients per sample)")
    parser.add_argument("--max_frames_per_file", type=int, default=500, help="Maximum MFCC frames to sample from each WAV file")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting (useful for headless environments)")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    print("Preparing dataset ...")
    Y_train, Y_test, _ = create_dataset(
        root_dir=data_dir,
        n_mfcc=args.n_mfcc,
        test_size=0.1,
        random_state=args.random_state,
        max_frames_per_file=args.max_frames_per_file,
        verbose=True,
    )
    print(f"Training samples: {Y_train.shape[1]} | Test samples: {Y_test.shape[1]}")

    print("Training SGK dictionary ...")
    sgk = SGK(n_atoms=args.n_atoms, n_iter=args.n_iter, sparsity=args.sparsity, random_state=args.random_state)
    sgk.fit(Y_train)

    D = sgk.dictionary_

    print("Encoding test set ...")
    X_test = sgk.transform(Y_test)
    Y_hat = sgk.reconstruct(X_test)

    # Metrics
    mse_values = mse(Y_test, Y_hat, axis=0)
    snr_values = snr(Y_test, Y_hat, axis=0)

    print(f"Average MSE: {np.mean(mse_values):.4e}")
    print(f"Median MSE: {np.median(mse_values):.4e}")
    print(f"Average SNR: {np.mean(snr_values):.2f} dB")

    if not args.no_plots:
        plot_dictionary_atoms(D)
        plot_error_histogram(mse_values)
        plot_reconstruction_comparison(Y_test, Y_hat, sample_idx=0)


if __name__ == "__main__":
    main()