# SGK字典学习与信号重建系统

基于序列广义K均值法(SGK)的智能手机Chirp信号字典学习与重建系统。

## 功能特点

- **完整SGK算法实现**: 基于图片中的SGK算法，包含稀疏编码、原子分组、字典更新等步骤
- **MFCC特征提取**: 支持从WAV音频文件提取梅尔频率倒谱系数特征
- **数据处理管道**: 自动化的数据加载、预处理、训练测试分割
- **多角度可视化分析**: 字典原子、收敛性、稀疏编码、重建性能等全方位分析
- **性能评估**: 训练集/测试集重建误差对比，泛化性能评估
- **参数研究**: 支持不同参数组合的性能对比分析

## 系统架构

```
├── sgk_algorithm.py      # SGK字典学习核心算法
├── data_processing.py    # 数据处理与MFCC特征提取
├── visualization.py      # 可视化分析模块
├── main.py              # 主程序入口
├── requirements.txt     # 依赖包列表
└── README.md           # 使用说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

使用合成chirp信号数据进行演示：

```bash
python main.py --use_synthetic
```

### 使用实际数据

如果您有实际的chirp信号WAV文件，请按以下目录结构组织：

```
chirp_data/
├── reference_point_1/
│   ├── signal_001.wav
│   ├── signal_002.wav
│   └── ...
├── reference_point_2/
│   ├── signal_001.wav
│   ├── signal_002.wav
│   └── ...
└── ...
```

然后运行：

```bash
python main.py --data_path ./chirp_data
```

### 参数配置

```bash
python main.py \
    --dict_size 64 \          # 字典大小
    --sparsity 8 \            # 稀疏度
    --max_iter 30 \           # 最大迭代次数
    --test_ratio 0.1 \        # 测试集比例
    --n_synthetic_samples 800 \  # 合成数据样本数
    --results_dir ./results   # 结果保存目录
```

## SGK算法原理

本系统实现的SGK算法包含以下主要步骤：

### 1. 初始化
- 设置字典大小M、稀疏度S、最大迭代次数P
- 随机初始化字典D⁰或从训练样本中选择

### 2. 稀疏编码
对每个信号 yⱼ ∈ Y，使用OMP算法求解：
```
xⱼ = arg min ||yⱼ - Dⁿ⁻¹xⱼ||₂²  s.t. ||xⱼ||₀ ≤ S
```

### 3. 原子分组
对每个原子 dₗ，计算使用该原子的样本索引集合：
```
mₗⁿ = {i | xₗⁱ(i) ≠ 0}
```

### 4. 字典更新
更新原子 dₗ*：
```
dₗⁿ = Σᵢ∈mₗⁿ yᵢ / |mₗⁿ|
```

### 5. 重复步骤2-4直至收敛

## 输出结果

运行完成后，在结果目录中会生成：

### 可视化图表
- `dictionary_atoms.png` - 字典原子可视化
- `convergence_analysis.png` - 算法收敛性分析
- `sparse_codes_analysis.png` - 稀疏编码分析
- `reconstruction_comparison.png` - 原始信号vs重建信号对比
- `error_analysis.png` - 重建误差分析
- `performance_metrics.png` - 性能指标对比

### 数据文件
- `learned_dictionary.npy` - 学习到的字典
- `sparse_codes.npy` - 稀疏编码系数
- `reconstruction_results.npz` - 重建结果
- `numerical_report.csv` - 数值分析报告

## 性能指标

系统提供多维度性能评估：

1. **重建精度**
   - 训练集/测试集均方误差(MSE)
   - 信号相关系数
   - 重建误差分布

2. **算法效率**
   - 训练时间
   - 收敛迭代次数
   - 字典更新原子数

3. **字典质量**
   - 字典相干性
   - 原子使用频率
   - 稀疏度分布

4. **泛化能力**
   - 测试集性能
   - 泛化误差比率

## 参数建议

根据信号特性调整参数：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| dict_size | 64-128 | 字典大小，根据信号复杂度调整 |
| sparsity | 5-15 | 稀疏度，通常为字典大小的10-20% |
| max_iter | 20-50 | 最大迭代次数，观察收敛情况 |
| test_ratio | 0.1-0.2 | 测试集比例 |

## 扩展功能

### 参数敏感性研究

取消注释`main.py`末尾的代码来运行参数研究：

```python
# 可选：运行参数研究
run_parameter_study()
```

这将测试不同参数组合的性能，结果保存在`parameter_study.csv`中。

## 故障排除

1. **内存不足**: 减少样本数量或降低字典大小
2. **收敛慢**: 增加最大迭代次数或调整稀疏度
3. **重建质量差**: 增加字典大小或调整稀疏度

## 技术细节

- **音频处理**: 使用librosa库进行音频加载和MFCC特征提取
- **稀疏编码**: 采用正交匹配追踪(OMP)算法
- **字典更新**: 使用SVD分解进行原子更新
- **可视化**: 基于matplotlib和seaborn的专业图表

## 引用

如果您在研究中使用了本系统，请引用相关的SGK算法文献。

## 许可证

本项目仅供学术研究使用。