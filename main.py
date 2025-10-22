from __future__ import print_function
import os
import torch
from model import model_generater
from video_dataset_anomaly_balance_uni_sample import dataset  # For anomaly
from torch.utils.data import DataLoader
from train import train
from tensorboard_logger import Logger
import options
import torch.optim as optim
import datetime
import glob

"""
Main entry point for Anomaly Event Detection (AED) training / 异常事件检测训练主入口
This script sets up the training environment, loads data, and starts the training process.
此脚本设置训练环境，加载数据，并启动训练过程。
"""

if __name__ == '__main__':
    # Parse command line arguments / 解析命令行参数
    args = options.parser.parse_args()
    
    # Set up device and random seed / 设置设备和随机种子
    torch.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    torch.cuda.set_device(args.device)
    
    # Create save path with timestamp / 创建带时间戳的保存路径
    time = datetime.datetime.now()
    save_path = os.path.join(
        args.model_name, 
        args.feature_pretrain_model, 
        args.dataset_name, 
        'sql_{}'.format(args.max_seqlen), 
        '_Lambda_{}'.format(args.Lambda), 
        args.feature_modal, 
        '{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
            time.year, time.month, time.day, time.hour, time.minute, time.second
        )
    )

    # Initialize model and optimizer / 初始化模型和优化器
    model = model_generater(model_name=args.model_name, feature_size=args.feature_size, arg=args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    
    # Load pretrained weights if specified / 如果指定则加载预训练权重
    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt))
        print("Loaded pretrained weights from: {}".format(args.pretrained_ckpt))

    # Create datasets and data loaders / 创建数据集和数据加载器
    train_dataset = dataset(args=args, train=True)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        pin_memory=True,
        num_workers=1, 
        shuffle=True
    )
    
    test_dataset = dataset(args=args, train=False)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=1, 
        pin_memory=True,
        num_workers=2, 
        shuffle=False
    )
    all_test_loader = [test_loader]

    # Note: The commented code below shows support for cross-validation splits
    # 注意：下面的注释代码显示了交叉验证分割的支持
    # if args.dataset_name == 'Avenue':
    #     train_test_split_path = os.path.join(args.dataset_path, args.dataset_name, 'Avenuetxt')
    #     train_lists = glob.glob(os.path.join(train_test_split_path, '*train*.txt'))
    #     test_lists = glob.glob(os.path.join(train_test_split_path, '*test*.txt'))
    # elif args.dataset_name == 'UCSDPed2':
    #     train_test_split_path = os.path.join(args.dataset_path, args.dataset_name, 'Ped2txt')
    #     train_lists = glob.glob(os.path.join(train_test_split_path, '*train*.txt'))
    #     test_lists = glob.glob(os.path.join(train_test_split_path, '*test*.txt'))
    # else:
    #     train_lists = None
    #     test_lists = None
    # if train_lists:
    #     # Cross-validation training loop / 交叉验证训练循环
    #     trainiters = 0
    #     for train_list, test_list in zip(train_lists, test_lists):
    #         train_dataset = dataset(args=args, train=True, trainlist=train_list, testlist=test_list)
    #         train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
    #                                   num_workers=2, shuffle=True)
    #         train2test_dataset = dataset_train2test(args=args, trainlist=train_list)
    #         test_dataset = dataset(args=args, train=False, trainlist=train_list, testlist=test_list)
    #         train2test_loader = DataLoader(dataset=train2test_dataset, batch_size=1, pin_memory=True,
    #                                  num_workers=2, shuffle=False)
    #         test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
    #                                  num_workers=2, shuffle=False)
    #         all_test_loader = [train2test_loader, test_loader]
    #         save_path_t = save_path + '/split_{}/'.format(trainiters)
    #         if not os.path.exists('./ckpt/' +save_path_t):
    #             os.makedirs('./ckpt/' + save_path_t)
    #         if not os.path.exists('./logs/' + save_path_t):
    #             os.makedirs('./logs/' + save_path_t)
    #         logger = Logger('./logs/' + save_path_t)
    #         train(epochs=args.max_epoch, train_loader=train_loader, all_test_loader=all_test_loader, args=args, model=model,
    #               optimizer=optimizer, logger=logger, device=device, save_path=save_path_t)
    #         trainiters += 1
    #         model = model_generater(model_name=args.model_name, feature_size=args.feature_size).to(device)
    #         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    #
    # else:
    #     # Standard training without cross-validation / 无交叉验证的标准训练
    #     train_dataset = dataset(args=args, train=True)
    #     train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
    #                               num_workers=1, shuffle=True)
    #     test_dataset = dataset(args=args, train=False)
    #     train2test_dataset = dataset_train2test(args=args)
    #     test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
    #                              num_workers=2, shuffle=False)
    #     train2test_loader = DataLoader(dataset=train2test_dataset, batch_size=1, pin_memory=True,
    #                                    num_workers=2, shuffle=False)
    #     all_test_loader = [train2test_loader, test_loader]

    # Create directories for checkpoints and logs / 创建检查点和日志目录
    if not os.path.exists('./ckpt/' + save_path):
        os.makedirs('./ckpt/' + save_path)
    if not os.path.exists('./logs/' + save_path):
        os.makedirs('./logs/' + save_path)
    
    # Initialize tensorboard logger / 初始化tensorboard日志记录器
    logger = Logger('./logs/' + save_path)
    
    # Start training / 开始训练
    print("Starting training with configuration:")
    print("Model: {}, Dataset: {}, Feature: {}".format(args.model_name, args.dataset_name, args.feature_modal))
    print("Save path: {}".format(save_path))
    
    train(
        epochs=args.max_epoch, 
        train_loader=train_loader, 
        all_test_loader=all_test_loader, 
        args=args, 
        model=model, 
        optimizer=optimizer, 
        logger=logger, 
        device=device, 
        save_path=save_path
    )
