import argparse

"""
Command line argument parser for Anomaly Event Detection (AED) / 异常事件检测命令行参数解析器
This file defines all the command line arguments for training and testing the AED model.
此文件定义了训练和测试AED模型的所有命令行参数。
"""

parser = argparse.ArgumentParser(description='AED - Anomaly Event Detection / 异常事件检测')

# Hardware and basic training parameters / 硬件和基础训练参数
parser.add_argument('--device', type=int, default=0, help='GPU ID / GPU设备ID')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001) / 学习率 (默认: 0.0001)')
parser.add_argument('--model_name', default='AED', help='Model name (AED or AED_T) / 模型名称 (AED 或 AED_T)')
parser.add_argument('--loss_type', default='DMIL_C', type=str, help='the type of n_pair loss, max_min_2, max_min, attention, attention_median, attention_H_L or max / 损失函数类型')
parser.add_argument('--pretrain', type=int, default=0, help='Whether to use pretrained model / 是否使用预训练模型')
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model / 预训练模型检查点路径')

# Testing parameters / 测试参数
parser.add_argument('--testing_path', type=str, default=None, help='time file for test model / 测试模型的时间文件')
parser.add_argument('--testing_model', type=str, default=None, help='iteration name for testing model / 测试模型的迭代名称')

# Feature and model parameters / 特征和模型参数
parser.add_argument('--feature_size', type=int, default=2048, help='size of feature (default: 2048) / 特征维度 (默认: 2048)')
parser.add_argument('--batch_size', type=int, default=1, help='number of samples in one iteration / 每次迭代的样本数')
parser.add_argument('--sample_size', type=int, default=30, help='number of samples in one iteration / 每次迭代的样本数')
parser.add_argument('--sample_step', type=int, default=1, help='Sample step for feature extraction / 特征提取的采样步长')

# Dataset parameters / 数据集参数
parser.add_argument('--dataset_name', type=str, default='LAD2000', help='Dataset name / 数据集名称')
parser.add_argument('--dataset_path', type=str, default='/home/tu-wan/windowswan/dataset', help='path to dir contains anomaly datasets / 包含异常数据集的目录路径')
parser.add_argument('--feature_modal', type=str, default='rgb', help='features from different input, options contain rgb, flow, combine / 特征模态 (rgb, flow, combine)')
parser.add_argument('--max_seqlen', type=int, default=5, help='sequence length(default: 5) / 序列长度 (默认: 5)')
parser.add_argument('--Lambda', type=str, default='1_1', help='Loss weights for classification and regression / 分类和回归损失的权重')

# Training parameters / 训练参数
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1) / 随机种子 (默认: 1)')
parser.add_argument('--max_epoch', type=int, default=100, help='maximum iteration to train (default: 50000) / 最大训练迭代次数 (默认: 50000)')

# Feature extraction parameters / 特征提取参数
parser.add_argument('--feature_pretrain_model', type=str, default='c3d', help='type of feature to be used I3D or C3D (default: I3D) / 特征提取模型类型 (I3D 或 C3D)')
parser.add_argument('--feature_layer', type=str, default='pool5', help='pool5, fc6 or fc7 / 特征提取层 (pool5, fc6, fc7)')
parser.add_argument('--k', type=int, default=4, help='value of k for KMXMILL loss / KMXMILL损失函数的k值')

# Visualization and logging / 可视化和日志记录
parser.add_argument('--plot', type=int, default=1, help='whether plot the video anomalous map on testing / 是否在测试时绘制视频异常图')
parser.add_argument('--snapshot', type=int, default=200, help='anomaly sample threshold / 异常样本阈值')

# Label and classification parameters / 标签和分类参数
parser.add_argument('--label_type', type=str, default='unary', help='Type of labels / 标签类型')
parser.add_argument('--ano_class', type=int, default=14, help='Number of anomaly classes / 异常类别数量')
