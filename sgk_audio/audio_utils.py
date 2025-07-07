from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_wav_mfcc(file_path: os.PathLike | str, n_mfcc: int = 20, sr: int | None = None) -> np.ndarray:
    """Load a WAV file and extract MFCC features.

    Parameters
    ----------
    file_path : PathLike | str
        Path to the audio file.
    n_mfcc : int, default 20
        Number of MFCC coefficients to extract.
    sr : int | None, default None
        Target sampling rate. ``None`` means use the file's original sampling rate.

    Returns
    -------
    mfcc : ndarray of shape (n_mfcc, n_frames)
    """
    y, sr = librosa.load(file_path, sr=sr)
    # Pre-emphasis? Could be added; for now default
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def traverse_dataset(root_dir: os.PathLike | str, wav_ext: Tuple[str, ...] = (".wav", ".WAV")) -> List[Path]:
    """Return a list of WAV file paths beneath *root_dir*.

    The directory structure can be arbitrary depth. All files with given extensions are collected.
    """
    root = Path(root_dir)
    wav_files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in wav_ext:
            wav_files.append(path)
    return wav_files


def create_dataset(root_dir: os.PathLike | str, n_mfcc: int = 20,
                   test_size: float = 0.1, random_state: int | None = 42,
                   max_frames_per_file: int | None = None,
                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create training and test matrices of MFCC features from the WAV dataset.

    Parameters
    ----------
    root_dir : str or PathLike
        Root directory containing sub-folders with WAV files.
    n_mfcc : int, default 20
        Dimensionality of MFCC features (i.e., M in SGK notation).
    test_size : float, default 0.1
        Fraction of frames for the test split.
    random_state : int | None, default 42
        Random seed for reproducibility.
    max_frames_per_file : int | None, default None
        If specified, randomly sample at most this many frames from each file to reduce dataset size.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    Y_train : ndarray of shape (n_mfcc, n_train_samples)
    Y_test : ndarray of shape (n_mfcc, n_test_samples)
    file_labels : list of str
        The label (e.g., directory name) corresponding to each sample (column) aligning with the concatenated Y (train+test).
    """
    wav_files = traverse_dataset(root_dir)
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found under {root_dir}.")

    random.seed(random_state)
    np.random.seed(random_state)

    Y_list: List[np.ndarray] = []
    labels: List[str] = []

    iterator = wav_files
    if verbose:
        iterator = tqdm(iterator, desc="Extracting MFCCs", unit="file")

    for file_path in iterator:
        mfcc = load_wav_mfcc(file_path, n_mfcc=n_mfcc)
        if max_frames_per_file is not None and mfcc.shape[1] > max_frames_per_file:
            idx = np.random.choice(mfcc.shape[1], size=max_frames_per_file, replace=False)
            mfcc = mfcc[:, idx]

        Y_list.append(mfcc)
        # Label by immediate parent directory (reference point)
        labels.extend([file_path.parent.name] * mfcc.shape[1])

    Y = np.concatenate(Y_list, axis=1)  # Shape (n_mfcc, total_frames)

    # Train-test split across frames (randomly, ignoring sequential correlation)
    idx_all = np.arange(Y.shape[1])
    train_idx, test_idx = train_test_split(idx_all, test_size=test_size, random_state=random_state, shuffle=True)

    Y_train = Y[:, train_idx]
    Y_test = Y[:, test_idx]

    return Y_train, Y_test, labels