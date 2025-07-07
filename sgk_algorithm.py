import numpy as np
from scipy.linalg import lstsq
from sklearn.linear_model import OrthogonalMatchingPursuit
from tqdm import tqdm


class SGKDictionaryLearning:
    """
    SGK (Sequential Generalized K-means) Dictionary Learning Algorithm
    基于图片中的SGK算法实现
    """
    
    def __init__(self, n_components=64, sparsity=10, max_iter=50, tol=1e-6, random_state=42):
        """
        初始化SGK字典学习算法
        
        Parameters:
        -----------
        n_components : int, 字典大小M
        sparsity : int, 稀疏度S
        max_iter : int, 最大迭代次数P
        tol : float, 收敛阈值
        random_state : int, 随机种子
        """
        self.n_components = n_components
        self.sparsity = sparsity
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.dictionary_ = None
        self.sparse_codes_ = None
        self.reconstruction_error_ = []
        
    def _initialize_dictionary(self, Y):
        """初始化字典D"""
        np.random.seed(self.random_state)
        n_features, n_samples = Y.shape
        
        # 随机选择训练样本的一部分作为初始字典原子
        if n_samples >= self.n_components:
            indices = np.random.choice(n_samples, self.n_components, replace=False)
            self.dictionary_ = Y[:, indices].copy()
        else:
            # 如果样本数少于字典大小，使用随机初始化
            self.dictionary_ = np.random.randn(n_features, self.n_components)
        
        # 归一化字典原子
        self._normalize_dictionary()
        
    def _normalize_dictionary(self):
        """归一化字典原子"""
        for j in range(self.n_components):
            norm = np.linalg.norm(self.dictionary_[:, j])
            if norm > 1e-12:
                self.dictionary_[:, j] /= norm
                
    def _sparse_coding(self, Y):
        """稀疏编码：对每个信号yj ∈ Y，使用OMP算法求解"""
        n_features, n_samples = Y.shape
        X = np.zeros((self.n_components, n_samples))
        
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.sparsity)
        
        for j in tqdm(range(n_samples), desc="稀疏编码"):
            # 使用OMP求解稀疏系数
            omp.fit(self.dictionary_, Y[:, j])
            X[:, j] = omp.coef_
            
        return X
    
    def _atom_grouping(self, X):
        """原子分组：对每个原子dℓ，计算使用该原子的样本索引集合"""
        atom_groups = {}
        
        for ell in range(self.n_components):
            # 找到使用原子dℓ的样本索引
            nonzero_indices = np.where(np.abs(X[ell, :]) > 1e-12)[0]
            atom_groups[ell] = nonzero_indices
            
        return atom_groups
    
    def _update_dictionary(self, Y, X, atom_groups):
        """字典更新：更新原子d*ℓ"""
        updated_atoms = 0
        
        for ell in range(self.n_components):
            group_indices = atom_groups[ell]
            
            if len(group_indices) == 0:
                # 如果没有样本使用该原子，随机重新初始化
                self.dictionary_[:, ell] = np.random.randn(self.dictionary_.shape[0])
                self.dictionary_[:, ell] /= np.linalg.norm(self.dictionary_[:, ell])
                continue
                
            # 计算残差矩阵
            Y_group = Y[:, group_indices]
            X_group = X[:, group_indices]
            
            # 移除当前原子的贡献
            residual = Y_group - self.dictionary_ @ X_group + np.outer(self.dictionary_[:, ell], X_group[ell, :])
            
            if len(group_indices) == 1:
                # 如果只有一个样本使用该原子
                if np.abs(X_group[ell, 0]) > 1e-12:
                    self.dictionary_[:, ell] = residual[:, 0] / X_group[ell, 0]
                    self.dictionary_[:, ell] /= np.linalg.norm(self.dictionary_[:, ell])
            else:
                # 使用SVD更新原子
                try:
                    U, s, Vt = np.linalg.svd(residual, full_matrices=False)
                    if len(s) > 0:
                        self.dictionary_[:, ell] = U[:, 0]
                        # 更新对应的稀疏系数
                        X[ell, group_indices] = s[0] * Vt[0, :]
                        updated_atoms += 1
                except np.linalg.LinAlgError:
                    # SVD失败时保持原子不变
                    pass
                    
        return updated_atoms
    
    def _compute_reconstruction_error(self, Y, X):
        """计算重建误差"""
        reconstruction = self.dictionary_ @ X
        error = np.linalg.norm(Y - reconstruction, 'fro') ** 2
        return error
    
    def fit(self, Y):
        """
        训练字典
        
        Parameters:
        -----------
        Y : array-like, shape (n_features, n_samples)
            训练数据矩阵
        """
        print(f"开始SGK字典学习训练...")
        print(f"数据维度: {Y.shape}")
        print(f"字典大小: {self.n_components}")
        print(f"稀疏度: {self.sparsity}")
        
        # 步骤1: 初始化
        self._initialize_dictionary(Y)
        
        prev_error = float('inf')
        
        for iteration in range(self.max_iter):
            print(f"\n迭代 {iteration + 1}/{self.max_iter}")
            
            # 步骤2: 稀疏编码
            X = self._sparse_coding(Y)
            
            # 步骤3: 原子分组
            atom_groups = self._atom_grouping(X)
            
            # 步骤4: 字典更新
            updated_atoms = self._update_dictionary(Y, X, atom_groups)
            
            # 计算重建误差
            error = self._compute_reconstruction_error(Y, X)
            self.reconstruction_error_.append(error)
            
            print(f"重建误差: {error:.6f}")
            print(f"更新原子数: {updated_atoms}")
            
            # 检查收敛
            if abs(prev_error - error) < self.tol:
                print(f"算法在第 {iteration + 1} 次迭代后收敛")
                break
                
            prev_error = error
            
        self.sparse_codes_ = X
        print("SGK字典学习训练完成!")
        
        return self
    
    def transform(self, Y):
        """
        对新数据进行稀疏编码
        
        Parameters:
        -----------
        Y : array-like, shape (n_features, n_samples)
            待编码数据
            
        Returns:
        --------
        X : array-like, shape (n_components, n_samples)
            稀疏编码系数
        """
        if self.dictionary_ is None:
            raise ValueError("字典尚未训练，请先调用fit方法")
            
        n_features, n_samples = Y.shape
        X = np.zeros((self.n_components, n_samples))
        
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.sparsity)
        
        for j in range(n_samples):
            omp.fit(self.dictionary_, Y[:, j])
            X[:, j] = omp.coef_
            
        return X
    
    def reconstruct(self, Y):
        """
        重建信号
        
        Parameters:
        -----------
        Y : array-like, shape (n_features, n_samples)
            原始信号
            
        Returns:
        --------
        Y_reconstructed : array-like, shape (n_features, n_samples)
            重建信号
        reconstruction_errors : array-like, shape (n_samples,)
            每个样本的重建误差
        """
        X = self.transform(Y)
        Y_reconstructed = self.dictionary_ @ X
        
        # 计算每个样本的重建误差
        reconstruction_errors = np.sum((Y - Y_reconstructed) ** 2, axis=0)
        
        return Y_reconstructed, reconstruction_errors
    
    def get_dictionary(self):
        """获取学习到的字典"""
        if self.dictionary_ is None:
            raise ValueError("字典尚未训练，请先调用fit方法")
        return self.dictionary_.copy()
    
    def get_sparse_codes(self):
        """获取训练数据的稀疏编码"""
        if self.sparse_codes_ is None:
            raise ValueError("字典尚未训练，请先调用fit方法")
        return self.sparse_codes_.copy()