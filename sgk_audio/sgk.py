import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from tqdm import tqdm


class SGK:
    """Sequential Generalized K 均值算法 (SGK) for dictionary learning.

    This implementation follows the algorithm described in the provided image.

    Attributes
    ----------
    n_atoms : int
        Number of dictionary atoms (columns of the dictionary).
    n_iter : int, default 10
        Maximum number of optimization iterations.
    sparsity : int, default 10
        Number of non  zero coefficients allowed in sparse coding (p in the description).
    random_state : int | None, default None
        Random seed for reproducibility.
    verbose : bool, default True
        Whether to display a progress bar.
    """

    def __init__(self, n_atoms: int, n_iter: int = 10, sparsity: int = 10,
                 random_state: int | None = None, verbose: bool = True):
        self.n_atoms = n_atoms
        self.n_iter = n_iter
        self.sparsity = sparsity
        self.random_state = random_state
        self.verbose = verbose
        self.dictionary_: np.ndarray | None = None  # Shape (n_features, n_atoms)

    def _initialize_dictionary(self, n_features: int):
        rng = np.random.default_rng(self.random_state)
        D = rng.standard_normal(size=(n_features, self.n_atoms))
        D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
        return D

    def fit(self, Y: np.ndarray):
        """Learn an overcomplete dictionary from the observations.

        Parameters
        ----------
        Y : ndarray of shape (n_features, n_samples)
            Data matrix whose columns are the signals to be coded.
        """
        n_features, n_samples = Y.shape
        self.dictionary_ = self._initialize_dictionary(n_features)
        D = self.dictionary_

        # Preallocate coefficient matrix
        X = np.zeros((self.n_atoms, n_samples))

        progress = range(self.n_iter)
        if self.verbose:
            progress = tqdm(progress, desc="SGK training", unit="iter")

        for _ in progress:
            # 1. Sparse coding step using Orthogonal Matching Pursuit (ROMP alternative)
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.sparsity, fit_intercept=False)
            for j in range(n_samples):
                omp.fit(D, Y[:, j])
                X[:, j] = omp.coef_

            # 2. Dictionary update (for each atom independently)
            for k in range(self.n_atoms):
                idx = np.nonzero(X[k, :])[0]
                if len(idx) == 0:
                    # Atom unused  reinitialize
                    D[:, k] = np.random.randn(n_features)
                else:
                    D[:, k] = np.mean(Y[:, idx], axis=1)

                # Normalize
                norm = np.linalg.norm(D[:, k])
                if norm < 1e-12:
                    # Rare case of zero norm, reinitialise
                    D[:, k] = np.random.randn(n_features)
                    norm = np.linalg.norm(D[:, k])
                D[:, k] /= norm

        self.dictionary_ = D
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Compute sparse codes of the signals using the learned dictionary.

        Parameters
        ----------
        Y : ndarray of shape (n_features, n_samples)
            Signals to encode.

        Returns
        -------
        X : ndarray of shape (n_atoms, n_samples)
            Sparse codes.
        """
        if self.dictionary_ is None:
            raise RuntimeError("The dictionary has not been fitted yet. Call fit() first.")

        D = self.dictionary_
        n_samples = Y.shape[1]
        X = np.zeros((self.n_atoms, n_samples))
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.sparsity, fit_intercept=False)
        for j in range(n_samples):
            omp.fit(D, Y[:, j])
            X[:, j] = omp.coef_
        return X

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct signals from sparse codes."""
        if self.dictionary_ is None:
            raise RuntimeError("Dictionary not fitted yet.")
        return self.dictionary_ @ X