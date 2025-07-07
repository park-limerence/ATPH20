#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGK字典学习主程序
基于智能手机chirp信号的MFCC特征进行字典训练和信号重建

作者: Claude Assistant
日期: 2024
"""

import os
import numpy as np
import argparse
import time
from sgk_algorithm import SGKDictionaryLearning
from data_processing import ChirpDataProcessor
from visualization import SGKVisualizer


def main():
    """主函数"""
    print("="*60)
    print("基于SGK算法的Chirp信号字典学习与重建系统")
    print("="*60)
    
    # 设置参数
    parser = argparse.ArgumentParser(description='SGK字典学习参数设置')
    parser.add_argument('--data_path', type=str, default='./chirp_data', 
                       help='chirp信号数据路径')
    parser.add_argument('--dict_size', type=int, default=64, 
                       help='字典大小')
    parser.add_argument('--sparsity', type=int, default=8, 
                       help='稀疏度')
    parser.add_argument('--max_iter', type=int, default=30, 
                       help='最大迭代次数')
    parser.add_argument('--test_ratio', type=float, default=0.1, 
                       help='测试集比例')
    parser.add_argument('--use_synthetic', action='store_true', 
                       help='使用合成数据')
    parser.add_argument('--n_synthetic_samples', type=int, default=800, 
                       help='合成数据样本数')
    parser.add_argument('--results_dir', type=str, default='./results', 
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"参数设置:")
    print(f"  字典大小: {args.dict_size}")
    print(f"  稀疏度: {args.sparsity}")
    print(f"  最大迭代次数: {args.max_iter}")
    print(f"  测试集比例: {args.test_ratio}")
    print(f"  结果保存目录: {args.results_dir}")
    
    # 步骤1: 数据处理
    print("\n" + "="*40)
    print("步骤1: 数据加载与预处理")
    print("="*40)
    
    processor = ChirpDataProcessor(sr=22050, n_mfcc=13)
    
    if args.use_synthetic or not os.path.exists(args.data_path):
        print("使用合成chirp信号数据...")
        features_list, labels = processor.generate_synthetic_chirp_data(
            n_samples=args.n_synthetic_samples, duration=1.0
        )
    else:
        print(f"从 {args.data_path} 加载实际数据...")
        file_info = processor.load_wav_files(args.data_path)
        if len(file_info) == 0:
            print("未找到WAV文件，使用合成数据...")
            features_list, labels = processor.generate_synthetic_chirp_data(
                n_samples=args.n_synthetic_samples, duration=1.0
            )
        else:
            features_list, labels, file_paths = processor.process_all_files(file_info)
    
    # 准备训练数据
    print("准备训练数据矩阵...")
    training_matrix = processor.prepare_training_data(features_list, segment_length=50)
    
    # 分割训练测试集
    print("分割训练测试集...")
    X_train, X_test = processor.split_train_test(
        training_matrix, test_size=args.test_ratio, random_state=42
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
        n_components=args.dict_size,
        sparsity=args.sparsity,
        max_iter=args.max_iter,
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
    
    visualizer = SGKVisualizer()
    
    # 生成完整的分析报告
    visualizer.generate_report(sgk_model, X_train_norm, X_test_norm, args.results_dir)
    
    # 保存模型和数据
    print("\n保存模型和数据...")
    
    # 保存字典
    dictionary = sgk_model.get_dictionary()
    np.save(os.path.join(args.results_dir, 'learned_dictionary.npy'), dictionary)
    
    # 保存稀疏编码
    sparse_codes = sgk_model.get_sparse_codes()
    np.save(os.path.join(args.results_dir, 'sparse_codes.npy'), sparse_codes)
    
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
                                 os.path.join(args.results_dir, 'reconstruction_results.npz'))
    
    # 生成总结报告
    print("\n" + "="*60)
    print("SGK字典学习完整流程总结报告")
    print("="*60)
    print(f"算法参数:")
    print(f"  字典大小: {args.dict_size}")
    print(f"  稀疏度: {args.sparsity}")
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
    
    print(f"\n所有结果已保存到: {args.results_dir}")
    print("="*60)
    print("程序执行完成!")


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


def run_parameter_study():
    """
    运行参数研究，分析不同参数对性能的影响
    """
    print("运行SGK参数敏感性研究...")
    
    # 参数网格
    dict_sizes = [32, 64, 128]
    sparsity_levels = [5, 10, 15]
    
    results = []
    
    processor = ChirpDataProcessor()
    features_list, labels = processor.generate_synthetic_chirp_data(n_samples=500)
    training_matrix = processor.prepare_training_data(features_list)
    X_train, X_test = processor.split_train_test(training_matrix, test_size=0.1)
    X_train_norm, X_test_norm = processor.normalize_data(X_train, X_test)
    
    for dict_size in dict_sizes:
        for sparsity in sparsity_levels:
            print(f"测试参数: 字典大小={dict_size}, 稀疏度={sparsity}")
            
            sgk_model = SGKDictionaryLearning(
                n_components=dict_size,
                sparsity=sparsity,
                max_iter=20,
                random_state=42
            )
            
            start_time = time.time()
            sgk_model.fit(X_train_norm)
            training_time = time.time() - start_time
            
            _, train_errors = sgk_model.reconstruct(X_train_norm)
            _, test_errors = sgk_model.reconstruct(X_test_norm)
            
            results.append({
                'dict_size': dict_size,
                'sparsity': sparsity,
                'train_mse': np.mean(train_errors),
                'test_mse': np.mean(test_errors),
                'training_time': training_time,
                'final_error': sgk_model.reconstruction_error_[-1]
            })
    
    # 保存参数研究结果
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('./results/parameter_study.csv', index=False)
    print("参数研究完成，结果保存到 ./results/parameter_study.csv")


if __name__ == "__main__":
    # 运行主程序
    main()
    
    # 可选：运行参数研究
    # run_parameter_study()