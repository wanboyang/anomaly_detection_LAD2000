import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
import options
import os
import pickle
import random
import torch

"""
Dataset class for Anomaly Event Detection (AED) / 异常事件检测数据集类
This module contains the dataset class for loading and processing video anomaly detection data.
此模块包含用于加载和处理视频异常检测数据的数据集类。
"""

class dataset(Dataset):
    def __init__(self, args, train=True, trainlist=None, testlist=None):
        """
        Dataset class for video anomaly detection / 视频异常检测数据集类
        
        Args:
            args: Command line arguments / 命令行参数
            train: Boolean indicating training or testing mode / 指示训练或测试模式的布尔值
            trainlist: Custom training list file / 自定义训练列表文件
            testlist: Custom testing list file / 自定义测试列表文件
        
        Attributes:
            dataset_path: Path to directory containing anomaly datasets / 包含异常数据集的目录路径
            dataset_name: Name of dataset being used / 正在使用的数据集名称
            feature_modal: Features from different input modalities (rgb, flow, combine) / 来自不同输入模态的特征
            feature_pretrain_model: Model name for feature extraction / 特征提取的模型名称
            feature_path: Directory containing all features for training and testing / 包含训练和测试所有特征的目录
            videoname: Video names in the dataset / 数据集中的视频名称
            trainlist: Video names for training / 训练用的视频名称
            testlist: Video names for testing / 测试用的视频名称
            train: Boolean type, if True the dataset returns training data / 布尔类型，如果为True则返回训练数据
            t_max: Maximum sequence length for sampling in training / 训练中采样的最大序列长度
        """
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.feature_modal = args.feature_modal
        self.feature_pretrain_model = args.feature_pretrain_model
        
        # Construct feature path based on model type / 根据模型类型构建特征路径
        if self.feature_pretrain_model == 'c3d' or self.feature_pretrain_model == 'c3d_ucf':
            self.feature_layer = args.feature_layer
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_layer, self.feature_modal)
        else:
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_modal)
        
        self.videoname = os.listdir(self.feature_path)
        # if self.args.larger_mem:
        #     self.data_dict = self.data_dict_creater()
        
        # Load train and test splits / 加载训练和测试分割
        if trainlist:
            self.trainlist = self.txt2list(trainlist)
            self.testlist = self.txt2list(testlist)
        else:
            self.trainlist = self.txt2list(
                txtpath=os.path.join(self.dataset_path, self.dataset_name, 'train_split.txt'))
            self.testlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'test_split.txt'))
        
        # Load ground truth labels / 加载真实标签
        self.video_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, self.dataset_name, 'GT', 'video_label_IET.pickle'))
        self.frame_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, self.dataset_name, 'GT', 'frame_label_IET.pickle'))
        
        # Split dataset into normal and anomaly videos / 将数据集分割为正常和异常视频
        if self.dataset_name == 'LV':
            self.normal_video_train, self.anomaly_video_train = self.p_n_split_dataset_LV(self.frame_label_dict, self.trainlist)
        else:
            self.normal_video_train, self.anomaly_video_train = self.p_n_split_dataset(self.frame_label_dict, self.trainlist)
        
        self.train = train
        self.t_max = args.max_seqlen



    def data_dict_creater(self):
        """
        Create dictionary mapping video names to features / 创建视频名到特征的字典映射
        
        Returns:
            Dictionary with video names as keys and features as values / 以视频名为键、特征为值的字典
        """
        data_dict = {}
        for _i in self.videoname:
            data_dict[_i] = np.load(
                file=os.path.join(self.feature_path, _i.replace('\n', '').replace('Ped', 'ped'), 'feature.npy'))
        return data_dict

    def txt2list(self, txtpath=''):
        """
        Generate list from text file / 从文本文件生成列表
        
        Args:
            txtpath: Path of text file / 文本文件路径
        
        Returns:
            List of lines from text file / 文本文件中的行列表
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def pickle_reader(self, file=''):
        """
        Read pickle file / 读取pickle文件
        
        Args:
            file: Path to pickle file / pickle文件路径
        
        Returns:
            Loaded pickle object / 加载的pickle对象
        """
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)
        return video_label_dict

    def p_n_split_dataset(self, frame_label_dict, trainlist):
        """
        Split dataset into positive (anomaly) and negative (normal) videos / 将数据集分割为正例（异常）和负例（正常）视频
        
        Args:
            frame_label_dict: Dictionary of frame-level labels / 帧级标签字典
            trainlist: List of training video names / 训练视频名称列表
        
        Returns:
            normal_video_train: List of normal video names / 正常视频名称列表
            anomaly_video_train: List of anomaly video names / 异常视频名称列表
        """
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            if frame_label_dict[t.replace('\n', '')].sum():  # If video contains anomalies / 如果视频包含异常
                anomaly_video_train.append(t.replace('\n', ''))
            else:  # Normal video / 正常视频
                normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))
        return normal_video_train, anomaly_video_train

    def p_n_split_dataset_LV(self, frame_label_dict, trainlist):
        """
        Special split function for LV dataset / LV数据集的特殊分割函数
        
        Args:
            frame_label_dict: Dictionary of frame-level labels / 帧级标签字典
            trainlist: List of training video names / 训练视频名称列表
        
        Returns:
            normal_video_train: List of normal video names / 正常视频名称列表
            anomaly_video_train: List of anomaly video names / 异常视频名称列表
        """
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            anomaly_video_train.append(t.replace('\n', ''))
            normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))

        return normal_video_train, anomaly_video_train
        # Alternative implementation using video labels / 使用视频标签的替代实现
        # for k, v in video_label_dict.items():
        #     if v[0] == 1.:
        #         anomaly_video_train.append(k)
        #     else:
        #         normal_video_train.append(k)
        # return normal_video_train, anomaly_video_train

    def __getitem__(self, index):
        """
        Get a single data sample / 获取单个数据样本
        
        Args:
            index: Index of the sample / 样本索引
        
        Returns:
            For training: [anomaly_features, normaly_features], [anomaly_frame_labels, normaly_frame_labels, anomaly_video_labels, normaly_video_labels], [train_video_name, start_index, len_index]
            For testing: features, data_video_name
        """
        if self.train:
            # Initialize lists for training data / 初始化训练数据列表
            anomaly_train_video_name = []
            normaly_train_video_name = []
            anomaly_start_index = []
            anomaly_len_index = []
            normaly_start_index = []
            normaly_len_index = []
            
            # Randomly sample anomaly and normal videos / 随机采样异常和正常视频
            anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size)
            normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)
            
            # Initialize tensors for features and labels / 初始化特征和标签张量
            anomaly_features = torch.zeros(0)
            normaly_features = torch.zeros(0)
            anomaly_frame_labels = torch.zeros(0, dtype=torch.long)
            normaly_frame_labels = torch.zeros(0, dtype=torch.long)
            anomaly_video_labels = torch.zeros(0, dtype=torch.long)
            normaly_video_labels = torch.zeros(0, dtype=torch.long)

            # Process each pair of anomaly and normal videos / 处理每对异常和正常视频
            for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                anomaly_data_video_name = a_i.replace('\n', '').replace('Ped', 'ped')
                normaly_data_video_name = n_i.replace('\n', '').replace('Ped', 'ped')
                anomaly_train_video_name.append(anomaly_data_video_name)
                normaly_train_video_name.append(normaly_data_video_name)
                
                # Load and process anomaly features / 加载和处理异常特征
                anomaly_feature = np.load(
                    file=os.path.join(self.feature_path, anomaly_data_video_name, 'feature.npy'))
                anomaly_len_index.append(anomaly_feature.shape[0])
                
                # Process anomaly features until we get a sequence with anomalies / 处理异常特征直到获得包含异常的序列
                tmp_num = 0
                while True:
                    anomaly_feature, r = utils.process_feat(anomaly_feature, self.t_max, step=1)
                    anomaly_frame_label = self.frame_label_dict[anomaly_data_video_name]
                    anomaly_frame_label = utils.process_label(anomaly_frame_label, r, self.t_max)
                    anomaly_video_label = np.asarray(self.video_label_dict[anomaly_data_video_name])
                    if anomaly_frame_label.sum():  # If sequence contains anomalies / 如果序列包含异常
                        break
                    tmp_num += 1
                    if tmp_num > 10:  # Maximum retry attempts / 最大重试次数
                        break
                anomaly_start_index.append(r)
                anomaly_feature = torch.from_numpy(anomaly_feature).unsqueeze(0)
                anomaly_frame_label = torch.from_numpy(anomaly_frame_label).unsqueeze(0).long()
                anomaly_video_label = torch.from_numpy(anomaly_video_label).unsqueeze(0).unsqueeze(0).long()
                
                # Load and process normal features / 加载和处理正常特征
                normaly_feature = np.load(
                    file=os.path.join(self.feature_path, normaly_data_video_name, 'feature.npy'))
                normaly_len_index.append(normaly_feature.shape[0])
                normaly_feature, r = utils.process_feat(normaly_feature, self.t_max, 1)
                normaly_frame_label = self.frame_label_dict[normaly_data_video_name]
                normaly_frame_label = utils.process_label(normaly_frame_label, r, self.t_max)
                normaly_video_label = np.asarray(self.video_label_dict[normaly_data_video_name])
                normaly_feature = torch.from_numpy(normaly_feature).unsqueeze(0)
                normaly_frame_label = torch.from_numpy(normaly_frame_label).unsqueeze(0).long()
                normaly_video_label = torch.from_numpy(normaly_video_label).unsqueeze(0).unsqueeze(0).long()
                normaly_start_index.append(r)
                
                # Concatenate features and labels / 连接特征和标签
                anomaly_features = torch.cat((anomaly_features, anomaly_feature), dim=0)  # combine anomaly_feature of different a_i
                normaly_features = torch.cat((normaly_features, normaly_feature), dim=0)  # combine normaly_feature of different n_i
                anomaly_frame_labels = torch.cat((anomaly_frame_labels, anomaly_frame_label), dim=0)
                normaly_frame_labels = torch.cat((normaly_frame_labels, normaly_frame_label), dim=0)
                anomaly_video_labels = torch.cat((anomaly_video_labels, anomaly_video_label), dim=0)
                normaly_video_labels = torch.cat((normaly_video_labels, normaly_video_label), dim=0)
            
            # Combine all information / 组合所有信息
            train_video_name = anomaly_train_video_name + normaly_train_video_name
            start_index = anomaly_start_index + normaly_start_index
            len_index = anomaly_len_index + normaly_len_index

            return [anomaly_features, normaly_features], [anomaly_frame_labels, normaly_frame_labels, anomaly_video_labels, normaly_video_labels], [train_video_name, start_index, len_index]
        else:
            # For testing, return features and video name / 对于测试，返回特征和视频名称
            data_video_name = self.testlist[index].replace('\n', '').replace('Ped', 'ped')
            self.feature = np.load(file=os.path.join(self.feature_path, data_video_name, 'feature.npy'))
            return self.feature, data_video_name

    def __len__(self):
        """
        Get the length of the dataset / 获取数据集的长度
        
        Returns:
            Length of training or testing list / 训练或测试列表的长度
        """
        if self.train:
            return len(self.trainlist)
        else:
            return len(self.testlist)


if __name__ == "__main__":
    args = options.parser.parse_args()
    train_dataset = dataset(args=args, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, pin_memory=True,
                              num_workers=5, shuffle=True)
    test_dataset = dataset(args=args, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=5, shuffle=False)
    for epoch in range(1):
        for i, data in enumerate(test_loader):
            features, _ = data
            print(features.shape)
    for epoch in range(2):
        for i, data in enumerate(train_loader):
            [anomaly_features, normaly_features], [anomaly_label, normaly_label] = data
            print(anomaly_features.squeeze(0).shape)
            print(normaly_label.squeeze(0).shape)
