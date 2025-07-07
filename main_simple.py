#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGK字典学习主程序（简化版）
基于智能手机chirp信号的简化特征进行字典训练和信号重建

作者: Claude Assistant
日期: 2024
"""

import os
import numpy as np
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

from sgk_algorithm import SGKDictionaryLearning
from data_processing_simple import ChirpDataProcessor
from visualization_simple import SGKVisualizer


def main():
    """主函数"""
    print("="*60)
    print("基于SGK算法的Chirp信号字典学习与重建系统（简化版）")
    print("="*60)
    
    # 设置参数（使用较小的参数以便快速演示）
    dict_size = 32
    sparsity = 5
    max_iter = 15
    test_ratio = 0.1
    n_synthetic_samples = 200
    results_dir = './results'
    
    print(f"参数设置:")
    print(f"  字典大小: {dict_size}")
    print(f"  稀疏度: {sparsity}")
    print(f"  最大迭代次数: {max_iter}")
    print(f"  测试集比例: {test_ratio}")
    print(f"  结果保存目录: {results_dir}")
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 步骤1: 数据处理
    print("\n" + "="*40)
    print("步骤1: 数据加载与预处理")
    print("="*40)
    
    processor = ChirpDataProcessor(sr=22050, n_features=39)
    
    print("生成合成chirp信号数据...")
    features_list, labels = processor.generate_synthetic_chirp_data(
        n_samples=n_synthetic_samples, duration=1.0
    )
    
    # 准备训练数据
    print("准备训练数据矩阵...")
    training_matrix = processor.prepare_training_data(features_list, segment_length=30)
    
    # 分割训练测试集
    print("分割训练测试集...")
    X_train, X_test = processor.split_train_test(
        training_matrix, test_size=test_ratio, random_state=42
    )
    
    # 数据标准化
    print("数据标准化...")
    X_train_norm, X_test_norm = processor.normalize_data(X_train, X_test)
    
    print(f"数据处理完成!")
    print(f"  训练集: {X_train_norm.shape[1]} 个样本")
    print(f"  测试集: {X_test_norm.shape[1]} 个样本")
    print(f"  特征维度: {X_train_norm.shape[0]}")
    
    # 步骤2: SGK字典学习
    print("\n" + "="*40)
    print("步骤2: SGK字典学习训练")
    print("="*40)
    
    # 初始化SGK模型
    sgk_model = SGKDictionaryLearning(
        n_components=dict_size,
        sparsity=sparsity,
        max_iter=max_iter,
        tol=1e-6,
        random_state=42
    )
    
    # 训练字典
    start_time = time.time()
    sgk_model.fit(X_train_norm)
    training_time = time.time() - start_time
    
    print(f"字典训练完成! 用时: {training_time:.2f} 秒")
    
    # 步骤3: 信号重建与评估
    print("\n" + "="*40)
    print("步骤3: 信号重建与性能评估")
    print("="*40)
    
    # 重建训练集信号
    print("重建训练集信号...")
    X_train_reconstructed, train_errors = sgk_model.reconstruct(X_train_norm)
    
    # 重建测试集信号
    print("重建测试集信号...")
    X_test_reconstructed, test_errors = sgk_model.reconstruct(X_test_norm)
    
    # 计算重建性能指标
    train_mse = np.mean(train_errors)
    test_mse = np.mean(test_errors)
    generalization_ratio = test_mse / train_mse
    
    print(f"重建性能评估:")
    print(f"  训练集平均MSE: {train_mse:.6f}")
    print(f"  测试集平均MSE: {test_mse:.6f}")
    print(f"  泛化误差比率: {generalization_ratio:.4f}")
    
    # 步骤4: 可视化分析
    print("\n" + "="*40)
    print("步骤4: 可视化分析与报告生成")
    print("="*40)
    
    # 设置matplotlib后端为Agg（非交互式）以避免显示问题
    import matplotlib
    matplotlib.use('Agg')
    
    visualizer = SGKVisualizer()
    
    # 生成完整的分析报告
    visualizer.generate_report(sgk_model, X_train_norm, X_test_norm, results_dir)
    
    # 保存模型和数据
    print("\n保存模型和数据...")
    
    # 保存字典
    dictionary = sgk_model.get_dictionary()
    np.save(os.path.join(results_dir, 'learned_dictionary.npy'), dictionary)
    
    # 保存稀疏编码
    sparse_codes = sgk_model.get_sparse_codes()
    np.save(os.path.join(results_dir, 'sparse_codes.npy'), sparse_codes)
    
    # 保存重建结果
    reconstruction_data = {
        'X_train_original': X_train_norm,
        'X_train_reconstructed': X_train_reconstructed,
        'X_test_original': X_test_norm,
        'X_test_reconstructed': X_test_reconstructed,
        'train_errors': train_errors,
        'test_errors': test_errors,
        'reconstruction_error_history': sgk_model.reconstruction_error_
    }
    processor.save_processed_data(reconstruction_data, 
                                 os.path.join(results_dir, 'reconstruction_results.npz'))
    
    # 生成总结报告
    print("\n" + "="*60)
    print("SGK字典学习完整流程总结报告")
    print("="*60)
    print(f"算法参数:")
    print(f"  字典大小: {dict_size}")
    print(f"  稀疏度: {sparsity}")
    print(f"  训练迭代次数: {len(sgk_model.reconstruction_error_)}")
    print(f"  训练时间: {training_time:.2f} 秒")
    print(f"\n数据统计:")
    print(f"  训练样本数: {X_train_norm.shape[1]}")
    print(f"  测试样本数: {X_test_norm.shape[1]}")
    print(f"  特征维度: {X_train_norm.shape[0]}")
    print(f"\n性能指标:")
    print(f"  最终训练误差: {sgk_model.reconstruction_error_[-1]:.6f}")
    print(f"  训练集重建MSE: {train_mse:.6f}")
    print(f"  测试集重建MSE: {test_mse:.6f}")
    print(f"  泛化性能: {generalization_ratio:.4f}")
    
    # 字典质量分析
    dictionary = sgk_model.get_dictionary()
    dict_coherence = compute_dictionary_coherence(dictionary)
    print(f"  字典相干性: {dict_coherence:.4f}")
    
    # 稀疏编码统计
    sparse_codes = sgk_model.get_sparse_codes()
    avg_sparsity = np.mean(np.sum(np.abs(sparse_codes) > 1e-6, axis=0))
    print(f"  平均稀疏度: {avg_sparsity:.2f}")
    
    print(f"\n所有结果已保存到: {results_dir}")
    print("="*60)
    print("程序执行完成!")
    
    # 显示生成的文件
    print("\n生成的文件:")
    for file in os.listdir(results_dir):
        if file.endswith(('.png', '.csv', '.npy', '.npz')):
            print(f"  {file}")


def compute_dictionary_coherence(dictionary):
    """
    计算字典的相干性（原子间的最大相关系数）
    
    Parameters:
    -----------
    dictionary : array-like, 字典矩阵
    
    Returns:
    --------
    coherence : float, 字典相干性
    """
    n_features, n_atoms = dictionary.shape
    max_coherence = 0.0
    
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            atom_i = dictionary[:, i]
            atom_j = dictionary[:, j]
            
            # 计算归一化相关系数
            correlation = np.abs(np.dot(atom_i, atom_j))
            max_coherence = max(max_coherence, correlation)
    
    return max_coherence


def demo_sgk_algorithm():
    """
    SGK算法演示函数
    """
    print("运行SGK算法演示...")
    
    # 生成简单的测试数据
    np.random.seed(42)
    n_features = 50
    n_samples = 100
    n_components = 20
    sparsity = 5
    
    # 生成稀疏信号
    true_dict = np.random.randn(n_features, n_components)
    for j in range(n_components):
        true_dict[:, j] /= np.linalg.norm(true_dict[:, j])
    
    true_codes = np.zeros((n_components, n_samples))
    for i in range(n_samples):
        indices = np.random.choice(n_components, sparsity, replace=False)
        true_codes[indices, i] = np.random.randn(sparsity)
    
    # 生成观测信号
    Y = true_dict @ true_codes + 0.1 * np.random.randn(n_features, n_samples)
    
    # 训练SGK模型
    sgk = SGKDictionaryLearning(n_components=n_components, sparsity=sparsity, max_iter=10)
    sgk.fit(Y)
    
    # 重建信号
    Y_reconstructed, errors = sgk.reconstruct(Y)
    
    print(f"重建误差: {np.mean(errors):.6f}")
    print("SGK算法演示完成!")


if __name__ == "__main__":
    # 运行主程序
    main()
    
    # 可选：运行算法演示
    # demo_sgk_algorithm()