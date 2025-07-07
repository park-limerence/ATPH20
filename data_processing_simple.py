import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm


class ChirpDataProcessor:
    """
    Chirp信号数据处理器（简化版，不依赖librosa）
    负责生成合成数据、提取简化特征、数据预处理
    """
    
    def __init__(self, sr=22050, n_features=39):
        """
        初始化数据处理器
        
        Parameters:
        -----------
        sr : int, 采样率
        n_features : int, 特征数量（模拟MFCC特征）
        """
        self.sr = sr
        self.n_features = n_features
        self.scaler = StandardScaler()
        
    def extract_simple_features(self, audio_data):
        """
        提取简化特征（模拟MFCC特征）
        
        Parameters:
        -----------
        audio_data : array-like, 音频数据
        
        Returns:
        --------
        features : array-like, 特征矩阵
        """
        # 简化的特征提取：使用滑动窗口计算统计特征
        window_size = 512
        hop_length = 256
        n_frames = (len(audio_data) - window_size) // hop_length + 1
        
        features = np.zeros((self.n_features, n_frames))
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + window_size
            frame = audio_data[start:end]
            
            # 计算简化特征
            # 频域特征（模拟MFCC）
            fft = np.fft.fft(frame)
            magnitude = np.abs(fft[:window_size//2])
            
            # 将频谱分成mel频带（简化版）
            n_mel_bands = 13
            mel_features = np.zeros(n_mel_bands)
            band_size = len(magnitude) // n_mel_bands
            
            for j in range(n_mel_bands):
                start_bin = j * band_size
                end_bin = min((j + 1) * band_size, len(magnitude))
                mel_features[j] = np.mean(magnitude[start_bin:end_bin])
            
            # 对数变换
            mel_features = np.log(mel_features + 1e-10)
            
            # DCT变换（简化版MFCC）
            mfcc = np.zeros(n_mel_bands)
            for k in range(n_mel_bands):
                for n in range(n_mel_bands):
                    mfcc[k] += mel_features[n] * np.cos(np.pi * k * (2*n + 1) / (2 * n_mel_bands))
            
            # 计算一阶和二阶差分（简化版）
            if i > 0:
                delta = mfcc - features[:n_mel_bands, i-1]
            else:
                delta = np.zeros(n_mel_bands)
                
            if i > 1:
                delta2 = delta - (features[n_mel_bands:2*n_mel_bands, i-1] if i-1 < n_frames else np.zeros(n_mel_bands))
            else:
                delta2 = np.zeros(n_mel_bands)
            
            # 组合特征
            features[:n_mel_bands, i] = mfcc
            features[n_mel_bands:2*n_mel_bands, i] = delta
            features[2*n_mel_bands:3*n_mel_bands, i] = delta2
            
            # 添加一些额外的时域特征
            if self.n_features > 3*n_mel_bands:
                extra_features = [
                    np.mean(frame),
                    np.std(frame),
                    np.max(frame),
                    np.min(frame)
                ]
                n_extra = min(len(extra_features), self.n_features - 3*n_mel_bands)
                features[3*n_mel_bands:3*n_mel_bands+n_extra, i] = extra_features[:n_extra]
        
        return features
    
    def prepare_training_data(self, features_list, segment_length=50):
        """
        准备训练数据：将特征分割成固定长度的片段
        
        Parameters:
        -----------
        features_list : list, 特征列表
        segment_length : int, 片段长度
        
        Returns:
        --------
        training_matrix : array-like, 训练数据矩阵 (n_features, n_samples)
        """
        all_segments = []
        
        for features in features_list:
            n_features, n_frames = features.shape
            
            # 将特征分割成固定长度的片段
            n_segments = n_frames // segment_length
            
            for i in range(n_segments):
                start_idx = i * segment_length
                end_idx = start_idx + segment_length
                segment = features[:, start_idx:end_idx]
                
                # 将片段展平为一维向量
                flattened_segment = segment.flatten()
                all_segments.append(flattened_segment)
        
        if len(all_segments) == 0:
            raise ValueError("没有有效的数据片段")
            
        # 转换为矩阵形式 (n_features, n_samples)
        training_matrix = np.array(all_segments).T
        
        print(f"训练数据形状: {training_matrix.shape}")
        print(f"总片段数: {len(all_segments)}")
        
        return training_matrix
    
    def split_train_test(self, training_matrix, test_size=0.1, random_state=42):
        """
        分割训练集和测试集
        
        Parameters:
        -----------
        training_matrix : array-like, 训练数据矩阵
        test_size : float, 测试集比例
        random_state : int, 随机种子
        
        Returns:
        --------
        X_train, X_test : array-like, 训练集和测试集
        """
        # 转置以适应sklearn的格式 (n_samples, n_features)
        X = training_matrix.T
        
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        
        # 转置回 (n_features, n_samples) 格式
        X_train = X_train.T
        X_test = X_test.T
        
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        
        return X_train, X_test
    
    def normalize_data(self, X_train, X_test=None):
        """
        数据标准化
        
        Parameters:
        -----------
        X_train : array-like, 训练数据
        X_test : array-like, 测试数据（可选）
        
        Returns:
        --------
        X_train_norm, X_test_norm : array-like, 标准化后的数据
        """
        # 在特征维度上进行标准化
        X_train_norm = self.scaler.fit_transform(X_train.T).T
        
        if X_test is not None:
            X_test_norm = self.scaler.transform(X_test.T).T
            return X_train_norm, X_test_norm
        
        return X_train_norm
    
    def generate_synthetic_chirp_data(self, n_samples=1000, duration=1.0):
        """
        生成合成的chirp信号数据用于演示
        
        Parameters:
        -----------
        n_samples : int, 样本数量
        duration : float, 信号时长（秒）
        
        Returns:
        --------
        features_list : list, 特征列表
        labels : list, 标签列表
        """
        print("生成合成chirp信号数据...")
        
        features_list = []
        labels = []
        
        # 生成不同类型的chirp信号
        reference_points = ['ref_point_1', 'ref_point_2', 'ref_point_3', 'ref_point_4']
        n_samples_per_ref = n_samples // len(reference_points)
        
        for ref_idx, ref_point in enumerate(reference_points):
            for sample_idx in tqdm(range(n_samples_per_ref), desc=f"生成 {ref_point} 数据"):
                # 生成时间轴
                t = np.linspace(0, duration, int(self.sr * duration))
                
                # 生成chirp信号参数
                f0 = 100 + ref_idx * 50 + np.random.uniform(-20, 20)  # 起始频率
                f1 = 1000 + ref_idx * 200 + np.random.uniform(-100, 100)  # 结束频率
                
                # 生成chirp信号
                chirp_signal = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
                
                # 添加噪声
                noise_level = 0.1
                noise = np.random.normal(0, noise_level, len(chirp_signal))
                chirp_signal += noise
                
                # 提取特征
                features = self.extract_simple_features(chirp_signal)
                
                features_list.append(features)
                labels.append(ref_point)
        
        print(f"生成了 {len(features_list)} 个合成样本")
        return features_list, labels
    
    def save_processed_data(self, data_dict, filename):
        """
        保存处理后的数据
        
        Parameters:
        -----------
        data_dict : dict, 数据字典
        filename : str, 保存文件名
        """
        np.savez_compressed(filename, **data_dict)
        print(f"数据已保存到: {filename}")
    
    def load_processed_data(self, filename):
        """
        加载处理后的数据
        
        Parameters:
        -----------
        filename : str, 数据文件名
        
        Returns:
        --------
        data_dict : dict, 数据字典
        """
        data = np.load(filename)
        data_dict = {key: data[key] for key in data.files}
        print(f"数据已从 {filename} 加载")
        return data_dict