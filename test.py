import torch
# import torch.nn.functional as F
# import utils
# import numpy as np
# from torch.autograd import Variable
# import scipy.io as sio

"""
Testing function for Anomaly Event Detection (AED) / 异常事件检测测试函数
This module contains the testing function for the AED model.
此模块包含AED模型的测试函数。
"""

def test(test_loader, model, device, args):
    """
    Test function for AED model / AED模型的测试函数
    
    Args:
        test_loader: Test data loader / 测试数据加载器
        model: AED model instance / AED模型实例
        device: Testing device (CPU/GPU) / 测试设备 (CPU/GPU)
        args: Command line arguments / 命令行参数
    
    Returns:
        List containing [ano_result, cls_result, extend_numbers] / 包含[异常结果, 分类结果, 扩展数量]的列表
        - ano_result: Dictionary mapping video names to anomaly scores / 视频名到异常得分的字典
        - cls_result: Dictionary mapping video names to predicted classes / 视频名到预测类别的字典
        - extend_numbers: Dictionary mapping video names to extension counts / 视频名到扩展数量的字典
    """
    ano_result = {}  # Store anomaly scores for each video / 存储每个视频的异常得分
    cls_result = {}  # Store classification results for each video / 存储每个视频的分类结果
    extend_numbers = {}  # Store number of extended frames for each video / 存储每个视频的扩展帧数
    # Process each video in the test set / 处理测试集中的每个视频
    for i, data in enumerate(test_loader):
        features, data_video_name = data
        features = features.squeeze()  # Remove batch dimension / 移除批次维度

        # Move features to device / 将特征移动到设备
        features = features.to(device)
        features_overlap = features.new_zeros(0)  # Initialize overlap features / 初始化重叠特征
        
        # Pad features if sequence length is not divisible by max_seqlen / 如果序列长度不能被max_seqlen整除，则填充特征
        if features.shape[0] % args.max_seqlen != 0:
            extend_number = args.max_seqlen - (features.shape[0] % args.max_seqlen)
            extend_tensor = features[-1].unsqueeze(0).repeat(extend_number, 1)  # Repeat last frame / 重复最后一帧
            features = torch.cat((features, extend_tensor), 0)  # Concatenate extended frames / 连接扩展帧
        else:
            extend_number = 0
        
        clip_num = len(features)  # Total number of clips / 总片段数
        
        # Create overlapping sequences for temporal modeling / 为时序建模创建重叠序列
        for i in range(0, len(features)-args.max_seqlen + 1):
            features_overlap = torch.cat((features_overlap, features[i: i+args.max_seqlen]), 0)
        
        # Reshape features for model input / 为模型输入重塑特征
        features = features.view(args.max_seqlen, features.shape[-2]//args.max_seqlen, features.shape[-1])
        features_overlap = features_overlap.view(args.max_seqlen, features_overlap.shape[-2]//args.max_seqlen, features_overlap.shape[-1])
        
        # Initialize output tensor for anomaly scores / 初始化异常得分输出张量
        ano_out = features_overlap.new_zeros((features_overlap.shape[-2], 16*clip_num))

        # Model inference with no gradient computation / 无梯度计算的模型推理
        with torch.no_grad():
            cls_scores, reg_scores = model(features_overlap)
        
        # Aggregate regression scores across overlapping windows / 跨重叠窗口聚合回归得分
        for i_index, reg_score in enumerate(reg_scores):
            j_index = 16 * i_index
            ano_out[i_index, j_index:j_index + (args.max_seqlen * 16)] = reg_score
        
        # Take maximum score across all predictions / 取所有预测中的最大得分
        ano_out = torch.max(ano_out, 0)[0]
        # Alternative: ano_out = torch.median(ano_out, 0)  # Could use median instead / 可以使用中位数替代

        # Get predicted class labels / 获取预测的类别标签
        _, predicted = torch.max(cls_scores.data, 1)
        predicted = predicted.cpu().data.numpy().reshape(-1)
        reg_scores = ano_out.cpu().data.numpy().reshape(-1)
        
        # Store results for this video / 存储此视频的结果
        ano_result[data_video_name[0]] = reg_scores
        cls_result[data_video_name[0]] = predicted
        extend_numbers[data_video_name[0]] = [extend_number]
    
    return [ano_result, cls_result, extend_numbers]
