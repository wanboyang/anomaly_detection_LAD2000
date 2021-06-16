from __future__ import print_function
import os
import torch
from model import model_generater
from video_dataset_anomaly_balance_uni_sample import dataset  # For anomaly
from torch.utils.data import DataLoader
from tensorboard_logger import Logger
import options
import torch.optim as optim
import datetime
import torch
import numpy as np
from test import test
from eval import eval_p
from confusion_m import cm
import os
import pickle
import torch.nn as nn
import glob

if __name__ == '__main__':

    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    torch.cuda.set_device(args.device)
    model = model_generater(model_name=args.model_name, feature_size=args.feature_size, arg=args).to(device)
    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt))
    else:
        print('pretrained_ckpt is needed for testing')
        exit()
    tmp_data = args.pretrained_ckpt.split('/')
    save_path = os.path.join(tmp_data[-8], tmp_data[-7], tmp_data[-6], tmp_data[-5], tmp_data[-4], tmp_data[-3], tmp_data[-2])
    dataset_name = tmp_data[-6]
    if not os.path.exists('./logs/' + save_path):
        os.makedirs('./logs/' + save_path)
    logger = Logger('./logs/'+ save_path)
    itr = int(tmp_data[-1].split('_')[-1].split('.')[0])
    test_dataset = dataset(args=args, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=2, shuffle=False)
    test_result_dict = test(test_loader, model, device, args)
    # train_result_dict = test(train2test_loader, model, device, args)
    # eval_p(itr=itr, dataset=dataset_name, predict_dict=test_result_dict, logger=logger, save_path=save_path, plot=1, args=args)
    cm(itr=itr, dataset=dataset_name, predict_dict=test_result_dict, logger=logger, save_path=save_path, plot=1, args=args)
    # with open(file=os.path.join('./result', save_path, 'predict.pickle'), mode='wb') as f:
    #     pickle.dump(train_result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(file=os.path.join('./result', save_path, 'log_statics.pickle'), mode='wb') as f:
    #     pickle.dump(log_statics, f, protocol=pickle.HIGHEST_PROTOCOL)

