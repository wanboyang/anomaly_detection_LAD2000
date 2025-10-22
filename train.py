import torch
import numpy as np
from test import test
from eval import eval_p
import os
import pickle
from losses import KMXMILL_individual, normal_smooth
import torch.nn as nn

"""
Training function for Anomaly Event Detection (AED) / 异常事件检测训练函数
This module contains the main training loop for the AED model.
此模块包含AED模型的主要训练循环。
"""

def train(epochs, train_loader, all_test_loader, args, model, optimizer, logger, device, save_path):
    """
    Main training function for AED model / AED模型的主要训练函数
    
    Args:
        epochs: Number of training epochs / 训练轮数
        train_loader: Training data loader / 训练数据加载器
        all_test_loader: List of test data loaders / 测试数据加载器列表
        args: Command line arguments / 命令行参数
        model: AED model instance / AED模型实例
        optimizer: Optimizer for training / 训练优化器
        logger: Tensorboard logger / Tensorboard日志记录器
        device: Training device (CPU/GPU) / 训练设备 (CPU/GPU)
        save_path: Path to save checkpoints and results / 保存检查点和结果的路径
    """
    
    # Initialize loss functions / 初始化损失函数
    cp_loss = nn.CrossEntropyLoss()  # Classification loss / 分类损失
    mse_loss = nn.SmoothL1Loss()     # Regression loss / 回归损失
    
    # Parse loss weights from Lambda argument / 从Lambda参数解析损失权重
    cls_loss_weight = float(args.Lambda.split('_')[0])  # Classification loss weight / 分类损失权重
    mse_loss_weight = float(args.Lambda.split('_')[1])  # Regression loss weight / 回归损失权重
    
    # Extract test loader from list / 从列表中提取测试加载器
    [test_loader] = all_test_loader
    
    # Initialize iteration counter / 初始化迭代计数器
    itr = 0
    
    # Create result directory / 创建结果目录
    if os.path.exists(os.path.join('./result', save_path)) == 0:
        os.makedirs(os.path.join('./result', save_path))
    
    # Save training configuration / 保存训练配置
    with open(file=os.path.join('./result', save_path, 'result.txt'), mode='w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
    
    # Statistics logging dictionary / 统计日志字典
    log_statics = {}
    
    # Load pretrained weights if available / 如果可用则加载预训练权重
    if args.pretrained_ckpt:
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint)
        print('Model loaded weights from {}'.format(args.pretrained_ckpt))
    else:
        print('Model is trained from scratch / 模型从头开始训练')
    
    # Main training loop / 主训练循环
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            itr += 1
            
            # Unpack data batch / 解包数据批次
            [anomaly_features, normaly_features], [anomaly_frame_labels, normaly_frame_labels, anomaly_video_labels, normaly_video_labels], stastics_data = data
            
            # Combine anomaly and normal features / 合并异常和正常特征
            features = torch.cat((anomaly_features.squeeze(0), normaly_features.squeeze(0)), dim=0)
            frame_labels = torch.cat((anomaly_frame_labels.squeeze(0), normaly_frame_labels.squeeze(0)), dim=0)
            video_labels = torch.cat((anomaly_video_labels.squeeze(0), normaly_video_labels.squeeze(0)), dim=0)

            # Move data to device and reshape / 将数据移动到设备并重塑
            features = features.to(device).permute(1, 0, 2)  # [seq_len, batch, feature_dim]
            frame_labels = frame_labels.float().to(device).squeeze()
            video_labels = video_labels.to(device).squeeze()
            
            # Forward pass / 前向传播
            cls_scores, reg_scores = model(features)
            
            # Calculate losses / 计算损失
            cls_loss = cp_loss(cls_scores, video_labels)  # Classification loss / 分类损失
            reg_loss = mse_loss(reg_scores, frame_labels)  # Regression loss / 回归损失
            
            # Total loss with weights / 带权重的总损失
            total_loss = cls_loss_weight * cls_loss + mse_loss_weight * reg_loss
            
            # Log losses to tensorboard / 将损失记录到tensorboard
            logger.log_value('cls_loss', cls_loss, itr)
            logger.log_value('reg_loss', reg_loss, itr)
            
            # Print training progress / 打印训练进度
            if itr % 20 == 0 and not itr == 0:
                print('Iteration:{}, Loss: {}'
                      .format(itr, total_loss.data.cpu().detach().numpy()))
            
            # Backward pass and optimization / 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Save checkpoint and evaluate model periodically / 定期保存检查点并评估模型
            if itr % args.snapshot == 0 and not itr == 0:
                # Save model checkpoint / 保存模型检查点
                torch.save(model.state_dict(), os.path.join('./ckpt/', save_path, 'iter_{}'.format(itr) + '.pkl'))
                
                # Test model on test set / 在测试集上测试模型
                test_result_dict = test(test_loader, model, device, args)
                
                # Evaluate performance / 评估性能
                eval_p(
                    itr=itr, 
                    dataset=args.dataset_name, 
                    predict_dict=test_result_dict, 
                    logger=logger, 
                    save_path=save_path,
                    plot=1, 
                    args=args
                )
            
            # Early stopping condition (for debugging) / 早停条件（用于调试）
            if itr > 12000:
                print("Reached maximum iterations, stopping training / 达到最大迭代次数，停止训练")
                exit()
