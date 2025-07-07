import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm


class ChirpDataProcessor:
    """
    Chirp信号数据处理器
    负责加载WAV文件、提取MFCC特征、数据预处理
    """
    
    def __init__(self, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        初始化数据处理器
        
        Parameters:
        -----------
        sr : int, 采样率
        n_mfcc : int, MFCC系数数量
        n_fft : int, FFT窗口大小
        hop_length : int, 跳跃长度
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.scaler = StandardScaler()
        
    def load_wav_files(self, data_path):
        """
        加载数据文件夹中的所有WAV文件
        
        Parameters:
        -----------
        data_path : str, 数据根目录路径
        
        Returns:
        --------
        file_info : list, 包含文件信息的列表
        """
        file_info = []
        
        if not os.path.exists(data_path):
            print(f"警告: 数据路径 {data_path} 不存在")
            return file_info
            
        # 遍历数据目录
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    file_path = os.path.join(root, file)
                    # 提取参考点信息（假设文件夹名称为参考点）
                    reference_point = os.path.basename(root)
                    file_info.append({
                        'file_path': file_path,
                        'file_name': file,
                        'reference_point': reference_point
                    })
                    
        print(f"找到 {len(file_info)} 个WAV文件")
        return file_info
    
    def extract_mfcc_features(self, audio_data, sr=None):
        """
        提取MFCC特征
        
        Parameters:
        -----------
        audio_data : array-like, 音频数据
        sr : int, 采样率
        
        Returns:
        --------
        mfcc_features : array-like, MFCC特征矩阵
        """
        if sr is None:
            sr = self.sr
            
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # 计算一阶和二阶差分
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 组合特征
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        return features
    
    def process_file(self, file_path):
        """
        处理单个音频文件
        
        Parameters:
        -----------
        file_path : str, 文件路径
        
        Returns:
        --------
        features : array-like, 提取的特征
        """
        try:
            # 加载音频文件
            audio_data, sr = librosa.load(file_path, sr=self.sr)
            
            # 提取MFCC特征
            features = self.extract_mfcc_features(audio_data, sr)
            
            return features
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return None
    
    def process_all_files(self, file_info):
        """
        处理所有音频文件
        
        Parameters:
        -----------
        file_info : list, 文件信息列表
        
        Returns:
        --------
        features_list : list, 特征列表
        labels : list, 标签列表（参考点）
        file_paths : list, 文件路径列表
        """
        features_list = []
        labels = []
        file_paths = []
        
        for info in tqdm(file_info, desc="处理音频文件"):
            features = self.process_file(info['file_path'])
            if features is not None:
                features_list.append(features)
                labels.append(info['reference_point'])
                file_paths.append(info['file_path'])
                
        print(f"成功处理 {len(features_list)} 个文件")
        return features_list, labels, file_paths
    
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
        features_list : list, MFCC特征列表
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
                
                # 提取MFCC特征
                features = self.extract_mfcc_features(chirp_signal)
                
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