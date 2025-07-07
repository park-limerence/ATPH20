import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')


class SGKVisualizer:
    """
    SGK字典学习结果可视化分析器（简化版）
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        初始化可视化器
        
        Parameters:
        -----------
        figsize : tuple, 图形大小
        """
        self.figsize = figsize
        plt.style.use('default')  # 使用默认样式
        
    def plot_dictionary_atoms(self, dictionary, n_atoms_to_show=16, save_path=None):
        """
        可视化字典原子
        
        Parameters:
        -----------
        dictionary : array-like, 字典矩阵
        n_atoms_to_show : int, 显示的原子数量
        save_path : str, 保存路径（可选）
        """
        n_features, n_components = dictionary.shape
        n_atoms_to_show = min(n_atoms_to_show, n_components)
        
        # 计算子图布局
        n_cols = 4
        n_rows = (n_atoms_to_show + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_atoms_to_show):
            row = i // n_cols
            col = i % n_cols
            
            atom = dictionary[:, i]
            axes[row, col].plot(atom, linewidth=2, color='blue')
            axes[row, col].set_title(f'原子 {i+1}')
            axes[row, col].set_xlabel('特征维度')
            axes[row, col].set_ylabel('幅值')
            axes[row, col].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_atoms_to_show, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('字典原子可视化', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_convergence_analysis(self, reconstruction_errors, save_path=None):
        """
        绘制收敛性分析图
        
        Parameters:
        -----------
        reconstruction_errors : list, 重建误差历史
        save_path : str, 保存路径（可选）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 重建误差随迭代变化
        iterations = range(1, len(reconstruction_errors) + 1)
        ax1.plot(iterations, reconstruction_errors, 'bo-', linewidth=2, markersize=6, color='blue')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('重建误差')
        ax1.set_title('重建误差收敛曲线')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 误差下降率
        if len(reconstruction_errors) > 1:
            error_reduction = [reconstruction_errors[i-1] - reconstruction_errors[i] 
                             for i in range(1, len(reconstruction_errors))]
            ax2.plot(range(2, len(reconstruction_errors) + 1), error_reduction, 'ro-', 
                    linewidth=2, markersize=6, color='red')
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('误差下降量')
            ax2.set_title('每次迭代误差下降量')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('SGK算法收敛性分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_sparse_codes_analysis(self, sparse_codes, save_path=None):
        """
        分析稀疏编码结果
        
        Parameters:
        -----------
        sparse_codes : array-like, 稀疏编码矩阵
        save_path : str, 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 稀疏编码热图
        # 选择前50个样本和前50个原子进行可视化
        n_atoms_show = min(50, sparse_codes.shape[0])
        n_samples_show = min(50, sparse_codes.shape[1])
        
        im = axes[0, 0].imshow(sparse_codes[:n_atoms_show, :n_samples_show], 
                              aspect='auto', cmap='RdBu_r', interpolation='nearest')
        axes[0, 0].set_title('稀疏编码系数热图')
        axes[0, 0].set_xlabel('样本索引')
        axes[0, 0].set_ylabel('字典原子索引')
        plt.colorbar(im, ax=axes[0, 0])
        
        # 2. 稀疏度分布
        sparsity_per_sample = np.sum(np.abs(sparse_codes) > 1e-6, axis=0)
        axes[0, 1].hist(sparsity_per_sample, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0, 1].set_xlabel('每个样本的非零系数数量')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('稀疏度分布')
        axes[0, 1].axvline(np.mean(sparsity_per_sample), color='red', linestyle='--', 
                          label=f'平均值: {np.mean(sparsity_per_sample):.2f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 原子使用频率
        atom_usage = np.sum(np.abs(sparse_codes) > 1e-6, axis=1)
        axes[1, 0].bar(range(len(atom_usage)), atom_usage, alpha=0.7, color='lightgreen')
        axes[1, 0].set_xlabel('字典原子索引')
        axes[1, 0].set_ylabel('使用频次')
        axes[1, 0].set_title('字典原子使用频率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 系数幅值分布
        nonzero_coeffs = sparse_codes[np.abs(sparse_codes) > 1e-6]
        axes[1, 1].hist(nonzero_coeffs, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 1].set_xlabel('系数值')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('非零系数幅值分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('稀疏编码分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_reconstruction_comparison(self, original_signals, reconstructed_signals, 
                                     n_samples_to_show=5, save_path=None):
        """
        比较原始信号和重建信号
        
        Parameters:
        -----------
        original_signals : array-like, 原始信号
        reconstructed_signals : array-like, 重建信号
        n_samples_to_show : int, 显示的样本数量
        save_path : str, 保存路径（可选）
        """
        n_samples_to_show = min(n_samples_to_show, original_signals.shape[1])
        
        fig, axes = plt.subplots(n_samples_to_show, 1, figsize=(15, 3 * n_samples_to_show))
        if n_samples_to_show == 1:
            axes = [axes]
        
        for i in range(n_samples_to_show):
            original = original_signals[:, i]
            reconstructed = reconstructed_signals[:, i]
            
            axes[i].plot(original, label='原始信号', linewidth=2, alpha=0.8, color='blue')
            axes[i].plot(reconstructed, label='重建信号', linewidth=2, alpha=0.8, color='red')
            
            # 计算相关系数
            correlation, _ = pearsonr(original, reconstructed)
            mse = mean_squared_error(original, reconstructed)
            
            axes[i].set_title(f'样本 {i+1} - 相关系数: {correlation:.4f}, MSE: {mse:.6f}')
            axes[i].set_xlabel('特征维度')
            axes[i].set_ylabel('幅值')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('原始信号 vs 重建信号对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_analysis(self, reconstruction_errors_per_sample, save_path=None):
        """
        分析重建误差
        
        Parameters:
        -----------
        reconstruction_errors_per_sample : array-like, 每个样本的重建误差
        save_path : str, 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 误差分布直方图
        axes[0, 0].hist(reconstruction_errors_per_sample, bins=50, alpha=0.7, 
                       edgecolor='black', density=True, color='lightblue')
        axes[0, 0].set_xlabel('重建误差')
        axes[0, 0].set_ylabel('密度')
        axes[0, 0].set_title('重建误差分布')
        axes[0, 0].axvline(np.mean(reconstruction_errors_per_sample), color='red', 
                          linestyle='--', label=f'平均值: {np.mean(reconstruction_errors_per_sample):.6f}')
        axes[0, 0].axvline(np.median(reconstruction_errors_per_sample), color='green', 
                          linestyle='--', label=f'中位数: {np.median(reconstruction_errors_per_sample):.6f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 误差累积分布
        sorted_errors = np.sort(reconstruction_errors_per_sample)
        cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[0, 1].plot(sorted_errors, cumulative_prob, linewidth=2, color='purple')
        axes[0, 1].set_xlabel('重建误差')
        axes[0, 1].set_ylabel('累积概率')
        axes[0, 1].set_title('重建误差累积分布函数')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 误差箱线图
        box_plot = axes[1, 0].boxplot(reconstruction_errors_per_sample, vert=True, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        axes[1, 0].set_ylabel('重建误差')
        axes[1, 0].set_title('重建误差箱线图')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 误差随样本索引变化
        axes[1, 1].plot(reconstruction_errors_per_sample, 'o-', markersize=3, linewidth=1, color='orange')
        axes[1, 1].set_xlabel('样本索引')
        axes[1, 1].set_ylabel('重建误差')
        axes[1, 1].set_title('重建误差随样本变化')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('重建误差分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_performance_metrics(self, train_errors, test_errors, save_path=None):
        """
        绘制性能指标对比
        
        Parameters:
        -----------
        train_errors : array-like, 训练集误差
        test_errors : array-like, 测试集误差
        save_path : str, 保存路径（可选）
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 训练集vs测试集误差分布对比
        axes[0].hist(train_errors, bins=30, alpha=0.7, label='训练集', density=True, color='blue')
        axes[0].hist(test_errors, bins=30, alpha=0.7, label='测试集', density=True, color='red')
        axes[0].set_xlabel('重建误差')
        axes[0].set_ylabel('密度')
        axes[0].set_title('训练集 vs 测试集误差分布')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 误差统计对比
        metrics = ['均值', '中位数', '标准差', '最大值', '最小值']
        train_stats = [np.mean(train_errors), np.median(train_errors), 
                      np.std(train_errors), np.max(train_errors), np.min(train_errors)]
        test_stats = [np.mean(test_errors), np.median(test_errors), 
                     np.std(test_errors), np.max(test_errors), np.min(test_errors)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1].bar(x - width/2, train_stats, width, label='训练集', alpha=0.8, color='blue')
        axes[1].bar(x + width/2, test_stats, width, label='测试集', alpha=0.8, color='red')
        axes[1].set_xlabel('统计指标')
        axes[1].set_ylabel('误差值')
        axes[1].set_title('误差统计指标对比')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metrics, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 散点图：训练误差 vs 测试误差（如果样本数相同）
        if len(train_errors) == len(test_errors):
            axes[2].scatter(train_errors, test_errors, alpha=0.6, color='green')
            axes[2].plot([min(train_errors), max(train_errors)], 
                        [min(train_errors), max(train_errors)], 'r--', linewidth=2)
            axes[2].set_xlabel('训练集误差')
            axes[2].set_ylabel('测试集误差')
            axes[2].set_title('训练集 vs 测试集误差散点图')
            
            # 计算相关系数
            correlation, _ = pearsonr(train_errors, test_errors)
            axes[2].text(0.05, 0.95, f'相关系数: {correlation:.4f}', 
                        transform=axes[2].transAxes, 
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        else:
            axes[2].text(0.5, 0.5, '样本数不匹配\n无法绘制散点图', 
                        ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
        
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('性能指标综合分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, sgk_model, X_train, X_test, save_dir='./results'):
        """
        生成完整的分析报告
        
        Parameters:
        -----------
        sgk_model : SGKDictionaryLearning, 训练好的SGK模型
        X_train : array-like, 训练数据
        X_test : array-like, 测试数据
        save_dir : str, 保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print("生成SGK字典学习分析报告...")
        
        # 1. 字典原子可视化
        print("1. 生成字典原子可视化...")
        dictionary = sgk_model.get_dictionary()
        self.plot_dictionary_atoms(dictionary, 
                                  save_path=os.path.join(save_dir, 'dictionary_atoms.png'))
        
        # 2. 收敛性分析
        print("2. 生成收敛性分析...")
        self.plot_convergence_analysis(sgk_model.reconstruction_error_, 
                                      save_path=os.path.join(save_dir, 'convergence_analysis.png'))
        
        # 3. 稀疏编码分析
        print("3. 生成稀疏编码分析...")
        sparse_codes = sgk_model.get_sparse_codes()
        self.plot_sparse_codes_analysis(sparse_codes, 
                                       save_path=os.path.join(save_dir, 'sparse_codes_analysis.png'))
        
        # 4. 重建对比
        print("4. 生成重建信号对比...")
        X_train_reconstructed, train_errors = sgk_model.reconstruct(X_train)
        self.plot_reconstruction_comparison(X_train, X_train_reconstructed, 
                                          save_path=os.path.join(save_dir, 'reconstruction_comparison.png'))
        
        # 5. 误差分析
        print("5. 生成误差分析...")
        self.plot_error_analysis(train_errors, 
                                save_path=os.path.join(save_dir, 'error_analysis.png'))
        
        # 6. 性能对比
        print("6. 生成性能对比...")
        X_test_reconstructed, test_errors = sgk_model.reconstruct(X_test)
        self.plot_performance_metrics(train_errors, test_errors, 
                                     save_path=os.path.join(save_dir, 'performance_metrics.png'))
        
        # 7. 生成数值报告
        print("7. 生成数值报告...")
        self._generate_numerical_report(sgk_model, train_errors, test_errors, save_dir)
        
        print(f"分析报告已保存到: {save_dir}")
    
    def _generate_numerical_report(self, sgk_model, train_errors, test_errors, save_dir):
        """
        生成数值分析报告
        """
        report_data = {
            '指标': [
                '字典大小',
                '稀疏度',
                '训练迭代次数',
                '最终重建误差',
                '训练集平均重建误差',
                '训练集重建误差标准差',
                '测试集平均重建误差',
                '测试集重建误差标准差',
                '泛化误差比率'
            ],
            '数值': [
                sgk_model.n_components,
                sgk_model.sparsity,
                len(sgk_model.reconstruction_error_),
                sgk_model.reconstruction_error_[-1],
                np.mean(train_errors),
                np.std(train_errors),
                np.mean(test_errors),
                np.std(test_errors),
                np.mean(test_errors) / np.mean(train_errors)
            ]
        }
        
        df = pd.DataFrame(report_data)
        
        # 保存为CSV
        df.to_csv(os.path.join(save_dir, 'numerical_report.csv'), index=False, encoding='utf-8-sig')
        
        # 打印报告
        print("\n" + "="*50)
        print("SGK字典学习数值分析报告")
        print("="*50)
        for i, row in df.iterrows():
            print(f"{row['指标']}: {row['数值']}")
        print("="*50)