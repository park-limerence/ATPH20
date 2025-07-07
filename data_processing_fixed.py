import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ChirpDataProcessor:
    """
    Chirp信号数据处理器（修复版，适配实际WAV文件）
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
        提取MFCC特征（增强版，处理实际音频数据）
        
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
            
        try:
            # 检查音频数据长度
            if len(audio_data) < self.n_fft:
                print(f"警告: 音频太短 ({len(audio_data)} samples), 进行零填充")
                audio_data = np.pad(audio_data, (0, self.n_fft - len(audio_data)), 'constant')
            
            # 预加重滤波
            audio_data = librosa.effects.preemphasis(audio_data)
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window='hann'
            )
            
            # 计算一阶和二阶差分
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 组合特征
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            
            # 检查特征是否有效
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                print("警告: 特征包含NaN或Inf值，进行修复")
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            print(f"MFCC特征提取失败: {e}")
            # 返回全零特征作为备用
            n_frames = max(1, (len(audio_data) - self.n_fft) // self.hop_length + 1)
            return np.zeros((self.n_mfcc * 3, n_frames))
    
    def process_file(self, file_path):
        """
        处理单个音频文件（增强版）
        
        Parameters:
        -----------
        file_path : str, 文件路径
        
        Returns:
        --------
        features : array-like, 提取的特征
        """
        try:
            # 加载音频文件，支持多种格式
            audio_data, sr = librosa.load(file_path, sr=self.sr, mono=True)
            
            # 检查音频数据有效性
            if len(audio_data) == 0:
                print(f"警告: 文件 {file_path} 为空")
                return None
                
            # 音频归一化
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # 提取MFCC特征
            features = self.extract_mfcc_features(audio_data, sr)
            
            # 检查特征维度
            if features.shape[1] == 0:
                print(f"警告: 文件 {file_path} 特征为空")
                return None
                
            return features
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return None
    
    def process_all_files(self, file_info):
        """
        处理所有音频文件（增强版）
        
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
        failed_files = []
        
        for info in tqdm(file_info, desc="处理音频文件"):
            features = self.process_file(info['file_path'])
            if features is not None and features.shape[1] > 0:
                features_list.append(features)
                labels.append(info['reference_point'])
                file_paths.append(info['file_path'])
            else:
                failed_files.append(info['file_path'])
                
        print(f"成功处理 {len(features_list)} 个文件")
        if failed_files:
            print(f"处理失败 {len(failed_files)} 个文件")
            
        return features_list, labels, file_paths
    
    def prepare_training_data(self, features_list, segment_length=20, min_frames=10):
        """
        准备训练数据：将特征分割成固定长度的片段（增强版）
        
        Parameters:
        -----------
        features_list : list, 特征列表
        segment_length : int, 片段长度（帧数）
        min_frames : int, 最小帧数要求
        
        Returns:
        --------
        training_matrix : array-like, 训练数据矩阵 (n_features, n_samples)
        """
        all_segments = []
        
        print(f"开始准备训练数据，目标片段长度: {segment_length} 帧")
        
        for i, features in enumerate(features_list):
            n_features, n_frames = features.shape
            
            print(f"处理第 {i+1}/{len(features_list)} 个特征，形状: {features.shape}")
            
            # 如果帧数太少，跳过
            if n_frames < min_frames:
                print(f"跳过: 帧数太少 ({n_frames} < {min_frames})")
                continue
            
            # 如果音频比片段长度短，使用整个音频
            if n_frames < segment_length:
                # 零填充到目标长度
                padded_features = np.zeros((n_features, segment_length))
                padded_features[:, :n_frames] = features
                flattened_segment = padded_features.flatten()
                all_segments.append(flattened_segment)
                print(f"零填充: {n_frames} -> {segment_length} 帧")
            else:
                # 将特征分割成固定长度的片段
                n_segments = n_frames // segment_length
                
                for j in range(n_segments):
                    start_idx = j * segment_length
                    end_idx = start_idx + segment_length
                    segment = features[:, start_idx:end_idx]
                    
                    # 将片段展平为一维向量
                    flattened_segment = segment.flatten()
                    all_segments.append(flattened_segment)
                
                # 处理剩余的帧（如果有的话）
                remaining_frames = n_frames % segment_length
                if remaining_frames >= min_frames:
                    # 从末尾取一个完整片段
                    start_idx = n_frames - segment_length
                    segment = features[:, start_idx:]
                    flattened_segment = segment.flatten()
                    all_segments.append(flattened_segment)
                
                print(f"分割成 {n_segments} 个片段")
        
        if len(all_segments) == 0:
            raise ValueError(f"没有有效的数据片段！请检查:\n"
                           f"1. 音频文件是否正确加载\n"
                           f"2. 特征提取是否成功\n"
                           f"3. segment_length({segment_length})是否过大\n"
                           f"4. min_frames({min_frames})是否过严格")
            
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
    
    def analyze_features(self, features_list):
        """
        分析特征统计信息
        
        Parameters:
        -----------
        features_list : list, 特征列表
        """
        if not features_list:
            print("没有特征可以分析")
            return
            
        frame_counts = []
        feature_dims = []
        
        for features in features_list:
            n_features, n_frames = features.shape
            frame_counts.append(n_frames)
            feature_dims.append(n_features)
        
        print(f"\n特征分析:")
        print(f"  文件数量: {len(features_list)}")
        print(f"  特征维度: {feature_dims[0]} (应该都相同)")
        print(f"  帧数统计:")
        print(f"    最小: {min(frame_counts)}")
        print(f"    最大: {max(frame_counts)}")
        print(f"    平均: {np.mean(frame_counts):.1f}")
        print(f"    中位数: {np.median(frame_counts):.1f}")
        print(f"  建议片段长度: {int(np.percentile(frame_counts, 25))}")
    
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