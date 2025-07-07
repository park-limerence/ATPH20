#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGK字典学习系统 - 简化运行脚本
自动检测数据特征并建议合适的参数

使用方法:
1. 快速分析: python run_analysis.py analyze
2. 自动运行: python run_analysis.py auto
3. 自定义运行: python run_analysis.py custom
"""

import os
import sys
import subprocess
import argparse

def run_quick_analysis(data_path):
    """运行快速数据分析"""
    print("="*50)
    print("执行快速数据分析...")
    print("="*50)
    
    try:
        cmd = [sys.executable, "main_fixed.py", "--analyze", data_path]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("分析失败:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"运行分析时出错: {e}")
        return False

def run_auto_mode(data_path):
    """自动模式：使用保守的参数运行"""
    print("="*50)
    print("执行自动模式（使用保守参数）...")
    print("="*50)
    
    # 保守的参数设置
    params = [
        "--data_path", data_path,
        "--segment_length", "15",
        "--min_frames", "5", 
        "--dict_size", "32",
        "--sparsity", "6",
        "--max_iter", "20",
        "--results_dir", "./results_auto"
    ]
    
    try:
        cmd = [sys.executable, "main_fixed.py"] + params
        print(f"运行命令: {' '.join(cmd)}")
        print("\n开始处理...")
        
        result = subprocess.run(cmd, text=True, encoding='utf-8')
        return result.returncode == 0
    except Exception as e:
        print(f"运行时出错: {e}")
        return False

def run_custom_mode():
    """自定义模式：让用户输入参数"""
    print("="*50)
    print("自定义参数模式")
    print("="*50)
    
    # 获取用户输入
    data_path = input("请输入数据路径 [./chirp_data]: ").strip() or "./chirp_data"
    
    print("\n建议参数范围:")
    print("  segment_length: 10-30 (音频特征片段长度)")
    print("  min_frames: 3-15 (最小帧数要求)")
    print("  dict_size: 32-128 (字典大小)")
    print("  sparsity: 5-15 (稀疏度)")
    print("  max_iter: 15-50 (最大迭代次数)")
    
    segment_length = input("\nsegment_length [15]: ").strip() or "15"
    min_frames = input("min_frames [5]: ").strip() or "5"
    dict_size = input("dict_size [64]: ").strip() or "64"
    sparsity = input("sparsity [8]: ").strip() or "8"
    max_iter = input("max_iter [30]: ").strip() or "30"
    results_dir = input("results_dir [./results_custom]: ").strip() or "./results_custom"
    
    params = [
        "--data_path", data_path,
        "--segment_length", segment_length,
        "--min_frames", min_frames,
        "--dict_size", dict_size,
        "--sparsity", sparsity,
        "--max_iter", max_iter,
        "--results_dir", results_dir
    ]
    
    try:
        cmd = [sys.executable, "main_fixed.py"] + params
        print(f"\n运行命令: {' '.join(cmd)}")
        print("\n开始处理...")
        
        result = subprocess.run(cmd, text=True, encoding='utf-8')
        return result.returncode == 0
    except Exception as e:
        print(f"运行时出错: {e}")
        return False

def check_dependencies():
    """检查依赖项"""
    required_files = [
        "main_fixed.py",
        "data_processing_fixed.py", 
        "sgk_algorithm.py",
        "visualization_simple.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("缺少以下必需文件:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SGK字典学习系统 - 简化运行脚本')
    parser.add_argument('mode', choices=['analyze', 'auto', 'custom'], 
                       help='运行模式: analyze(分析), auto(自动), custom(自定义)')
    parser.add_argument('--data_path', type=str, 
                       default=r"E:\data_process\filtered_normalized_chirps",
                       help='数据路径')
    
    args = parser.parse_args()
    
    print("SGK字典学习系统 - 简化运行脚本")
    print("="*50)
    
    # 检查依赖
    if not check_dependencies():
        print("请确保所有必需文件都存在")
        return False
    
    # 检查数据路径
    if not os.path.exists(args.data_path):
        print(f"警告: 数据路径 {args.data_path} 不存在")
        if args.mode != 'custom':
            return False
    
    success = False
    
    if args.mode == 'analyze':
        # 分析模式
        success = run_quick_analysis(args.data_path)
        
        if success:
            print("\n分析完成! 基于分析结果:")
            print("1. 如果音频很短，使用: python run_analysis.py auto")
            print("2. 如果要自定义参数，使用: python run_analysis.py custom")
        
    elif args.mode == 'auto':
        # 自动模式 - 先分析再运行
        print("步骤1: 快速分析数据...")
        if run_quick_analysis(args.data_path):
            print("\n步骤2: 自动运行...")
            success = run_auto_mode(args.data_path)
        else:
            print("分析失败，无法继续自动运行")
            
    elif args.mode == 'custom':
        # 自定义模式
        success = run_custom_mode()
    
    if success:
        print("\n" + "="*50)
        print("处理完成! 请查看结果目录中的输出文件")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("处理失败! 请检查错误信息并重试")
        print("="*50)
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
        sys.exit(1)
    except Exception as e:
        print(f"\n意外错误: {e}")
        sys.exit(1)