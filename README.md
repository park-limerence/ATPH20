# SGK Dictionary Learning for Chirp Audio

This repository provides an end-to-end example of training an over-complete dictionary on smart-phone captured audio chirp signals using the Sequential Generalised K-means (SGK) algorithm.

## Pipeline

1. Collect `.wav` files in sub-folders, one sub-folder per reference point.
2. Convert audio to MFCC feature frames.
3. Split frames into training (90 %) and testing (10 %).
4. Train an SGK dictionary on the training frames.
5. Reconstruct the test frames via sparse coding (OMP).
6. Evaluate reconstruction quality (MSE, SNR) and visualise results.

## Quick Start

```bash
# 1. Install dependencies (ideally inside a virtual environment)
pip install -r requirements.txt

# 2. Run training & evaluation. Replace /path/to/data with your root data directory.
python -m sgk_audio.train_sgk --data_dir /path/to/data \
                               --n_mfcc 20 \
                               --n_atoms 40 \
                               --n_iter 20 \
                               --sparsity 8
```

Command-line arguments:

* `--data_dir` Root directory holding the WAV files.
* `--n_mfcc`  Number of MFCC coefficients (feature dimension).
* `--n_atoms` Number of atoms in the dictionary (over-completeness).
* `--n_iter`  SGK optimisation iterations.
* `--sparsity` Maximum non-zero coefficients per sample.
* `--max_frames_per_file` Limit frames per file (speed).
* `--no_plots` Disable matplotlib windows.

## Results

During execution the script prints average and median MSE as well as average SNR on the held-out test set. If plotting is enabled, it additionally shows:

* Heatmap grid of learned atoms.
* Histogram of reconstruction errors.
* Example frame reconstruction (original vs prediction).

Feel free to extend the visualisations in `sgk_audio/evaluate.py`.