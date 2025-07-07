#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGK字典学习主程序（修复版，适配实际WAV文件）
基于智能手机chirp信号的MFCC特征进行字典训练和信号重建

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
from data_processing_fixed import ChirpDataProcessor
from visualization_simple import SGKVisualizer


def main():
    """主函数"""
    print("="*60)
    print("基于SGK算法的Chirp信号字典学习与重建系统（修复版）")
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
    parser.add_argument('--segment_length', type=int, default=20, 
                       help='特征片段长度（帧数）')
    parser.add_argument('--min_frames', type=int, default=10, 
                       help='最小帧数要求')
    parser.add_argument('--results_dir', type=str, default='./results', 
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"参数设置:")
    print(f"  数据路径: {args.data_path}")
    print(f"  字典大小: {args.dict_size}")
    print(f"  稀疏度: {args.sparsity}")
    print(f"  最大迭代次数: {args.max_iter}")
    print(f"  测试集比例: {args.test_ratio}")
    print(f"  片段长度: {args.segment_length} 帧")
    print(f"  最小帧数: {args.min_frames}")
    print(f"  结果保存目录: {args.results_dir}")
    
    # 步骤1: 数据处理
    print("\n" + "="*40)
    print("步骤1: 数据加载与预处理")
    print("="*40)
    
    processor = ChirpDataProcessor(sr=22050, n_mfcc=13)
    
    # 检查数据路径是否存在
    if os.path.exists(args.data_path):
        print(f"从 {args.data_path} 加载实际数据...")
        file_info = processor.load_wav_files(args.data_path)
        
        if len(file_info) == 0:
            print("未找到WAV文件，程序终止")
            return
        
        # 处理所有文件
        features_list, labels, file_paths = processor.process_all_files(file_info)
        
        if len(features_list) == 0:
            print("没有成功处理的文件，程序终止")
            return
        
        # 分析特征统计信息
        processor.analyze_features(features_list)
        
    else:
        print(f"数据路径 {args.data_path} 不存在，程序终止")
        return
    
    # 准备训练数据
    print("\n准备训练数据矩阵...")
    try:
        training_matrix = processor.prepare_training_data(
            features_list, 
            segment_length=args.segment_length,
            min_frames=args.min_frames
        )
    except ValueError as e:
        print(f"准备训练数据失败: {e}")
        print("\n建议:")
        print(f"1. 减少segment_length (当前: {args.segment_length})")
        print(f"2. 减少min_frames (当前: {args.min_frames})")
        print("3. 检查音频文件是否正确")
        return
    
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
    try:
        sgk_model.fit(X_train_norm)
        training_time = time.time() - start_time
        print(f"字典训练完成! 用时: {training_time:.2f} 秒")
    except Exception as e:
        print(f"字典训练失败: {e}")
        return
    
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
    try:
        visualizer.generate_report(sgk_model, X_train_norm, X_test_norm, args.results_dir)
    except Exception as e:
        print(f"可视化报告生成失败: {e}")
        print("继续保存数据...")
    
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
        'reconstruction_error_history': sgk_model.reconstruction_error_,
        'file_paths': file_paths[:len(features_list)],  # 对应处理成功的文件
        'labels': labels
    }
    processor.save_processed_data(reconstruction_data, 
                                 os.path.join(args.results_dir, 'reconstruction_results.npz'))
    
    # 生成总结报告
    print("\n" + "="*60)
    print("SGK字典学习完整流程总结报告")
    print("="*60)
    print(f"数据统计:")
    print(f"  处理的WAV文件: {len(features_list)}")
    print(f"  训练样本数: {X_train_norm.shape[1]}")
    print(f"  测试样本数: {X_test_norm.shape[1]}")
    print(f"  特征维度: {X_train_norm.shape[0]}")
    print(f"\n算法参数:")
    print(f"  字典大小: {args.dict_size}")
    print(f"  稀疏度: {args.sparsity}")
    print(f"  训练迭代次数: {len(sgk_model.reconstruction_error_)}")
    print(f"  训练时间: {training_time:.2f} 秒")
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
    
    # 显示生成的文件
    print("\n生成的文件:")
    try:
        for file in os.listdir(args.results_dir):
            if file.endswith(('.png', '.csv', '.npy', '.npz')):
                file_path = os.path.join(args.results_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  {file} ({file_size:.1f} KB)")
    except Exception as e:
        print(f"  无法列出文件: {e}")


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


def quick_analysis(data_path):
    """
    快速分析数据集
    
    Parameters:
    -----------
    data_path : str, 数据路径
    """
    print("执行快速数据分析...")
    
    processor = ChirpDataProcessor()
    file_info = processor.load_wav_files(data_path)
    
    if len(file_info) == 0:
        print("未找到WAV文件")
        return
    
    # 分析前10个文件
    sample_files = file_info[:min(10, len(file_info))]
    features_list = []
    
    for info in sample_files:
        features = processor.process_file(info['file_path'])
        if features is not None:
            features_list.append(features)
    
    if features_list:
        processor.analyze_features(features_list)
    else:
        print("没有成功处理的文件")


if __name__ == "__main__":
    import sys
    
    # 如果提供了 --analyze 参数，进行快速分析
    if len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        if len(sys.argv) > 2:
            quick_analysis(sys.argv[2])
        else:
            print("使用方法: python main_fixed.py --analyze <data_path>")
    else:
        # 运行主程序
        main()