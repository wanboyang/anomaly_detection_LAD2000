import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import options
# from video_dataset_anomaly_balance_sample import dataset # For anomaly
# from torch.utils.data import DataLoader
# import math
# from utils import fill_context_mask, median

"""
Loss functions for Anomaly Event Detection (AED) / 异常事件检测损失函数
This module contains various loss functions used in training the AED model.
此模块包含训练AED模型使用的各种损失函数。
"""

# Predefined loss functions / 预定义的损失函数
mseloss = torch.nn.MSELoss(reduction='mean')  # Mean squared error loss / 均方误差损失
mseloss_vector = torch.nn.MSELoss(reduction='none')  # MSE loss without reduction / 无缩减的MSE损失
binary_CE_loss = torch.nn.BCELoss(reduction='mean')  # Binary cross entropy loss / 二元交叉熵损失
binary_CE_loss_vector = torch.nn.BCELoss(reduction='none')  # BCE loss without reduction / 无缩减的BCE损失




def cross_entropy(logits, target, size_average=True):
    """
    Custom cross entropy loss function / 自定义交叉熵损失函数
    
    Args:
        logits: Model output logits / 模型输出logits
        target: Target labels / 目标标签
        size_average: Whether to average the loss / 是否对损失求平均
    
    Returns:
        Cross entropy loss / 交叉熵损失
    """
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


def hinger_loss(anomaly_score, normal_score):
    """
    Hinge loss for anomaly detection / 异常检测的铰链损失
    
    Args:
        anomaly_score: Anomaly prediction scores / 异常预测得分
        normal_score: Normal prediction scores / 正常预测得分
    
    Returns:
        Hinge loss value / 铰链损失值
    """
    return F.relu((1 - anomaly_score + normal_score))


def normal_smooth(element_logits, labels, device):
    """
    Smoothness loss for normal videos / 正常视频的平滑性损失
    Encourages consistent predictions in normal videos / 鼓励正常视频中的一致预测
    
    Args:
        element_logits: Frame-level prediction logits / 帧级预测logits
        labels: Video labels (0 for normal, 1 for anomaly) / 视频标签 (0表示正常, 1表示异常)
        device: Computation device / 计算设备
    
    Returns:
        Smoothness loss for normal videos / 正常视频的平滑性损失
    """
    normal_smooth_loss = torch.zeros(0).to(device)
    real_size = int(element_logits.shape[0])
    # Process each sample in the batch / 处理批次中的每个样本
    for i in range(real_size):
        if labels[i] == 0:  # Only apply to normal videos / 仅应用于正常视频
            # Calculate variance of predictions for this video / 计算此视频预测的方差
            normal_smooth_loss = torch.cat((normal_smooth_loss, torch.var(element_logits[i]).unsqueeze(0)))
    normal_smooth_loss = torch.mean(normal_smooth_loss, dim=0)  # Average over batch / 在批次上求平均
    return normal_smooth_loss










def KMXMILL_individual(element_logits,
                       seq_len,
                       labels,
                       device,
                       loss_type='CE',
                       args=None):
    """
    K-Max Multiple Instance Learning (KMXMILL) loss / K-最大多示例学习损失
    Selects top-k predictions from each video and applies loss / 从每个视频中选择top-k预测并应用损失
    
    Args:
        element_logits: Frame-level prediction logits / 帧级预测logits
        seq_len: Sequence lengths for each video / 每个视频的序列长度
        labels: Video labels (0 for normal, 1 for anomaly) / 视频标签 (0表示正常, 1表示异常)
        device: Computation device / 计算设备
        loss_type: Type of loss ('CE' or 'MSE') / 损失类型 ('CE' 或 'MSE')
        args: Command line arguments / 命令行参数
    
    Returns:
        Multiple instance learning loss / 多示例学习损失
    """
    # [train_video_name, start_index, len_index] = stastics_data  # Statistics data / 统计数据
    
    # Calculate k value for each video / 计算每个视频的k值
    k = np.ceil(seq_len/args.k).astype('int32')
    
    # Initialize tensors for instance logits and labels / 初始化实例logits和标签的张量
    instance_logits = torch.zeros(0).to(device)
    real_label = torch.zeros(0).to(device)
    real_size = int(element_logits.shape[0])
    
    # Process each video in the batch / 处理批次中的每个视频
    for i in range(real_size):
        # Select top-k predictions for this video / 为此视频选择top-k预测
        tmp, tmp_index = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        
        # Note: The commented code below was for logging statistics / 注意：下面的注释代码用于记录统计信息
        # top_index = np.zeros(len_index[i].numpy())
        # top_predicts = np.zeros(len_index[i].numpy())
        # top_index[tmp_index.cpu().numpy() + start_index[i].numpy()] = 1
        # if train_video_name[i][0] in log_statics:
        #     log_statics[train_video_name[i][0]] = np.concatenate((log_statics[train_video_name[i][0]], np.expand_dims(top_index, axis=0)),axis=0)
        # else:
        #     log_statics[train_video_name[i][0]] = np.expand_dims(top_index, axis=0)
        
        # Collect top-k predictions / 收集top-k预测
        instance_logits = torch.cat((instance_logits, tmp), dim=0)
        
        # Create labels for top-k instances / 为top-k实例创建标签
        if labels[i] == 1:  # Anomaly video / 异常视频
            real_label = torch.cat((real_label, torch.ones((int(k[i]), 1)).to(device)), dim=0)
        else:  # Normal video / 正常视频
            real_label = torch.cat((real_label, torch.zeros((int(k[i]), 1)).to(device)), dim=0)
    
    # Apply appropriate loss function / 应用适当的损失函数
    if loss_type == 'CE':
        milloss = binary_CE_loss(input=instance_logits, target=real_label)
        return milloss
    elif loss_type == 'MSE':
        milloss = mseloss(input=instance_logits, target=real_label)
        return milloss
