import torch
import numpy as np
from test import test
from eval import eval_p
import os
import pickle
from losses import KMXMILL_individual, normal_smooth
import torch.nn as nn

def train(epochs, train_loader, all_test_loader, args, model, optimizer, logger, device, save_path):
    # [train2test_loader, test_loader] = all_test_loader
    cp_loss = nn.CrossEntropyLoss()
    mse_loss = nn.SmoothL1Loss()
    cls_loss_weight = float(args.Lambda.split('_')[0])
    mse_loss_weight = float(args.Lambda.split('_')[1])
    [test_loader] = all_test_loader
    itr = 0
    if os.path.exists(os.path.join('./result', save_path)) == 0:
        os.makedirs(os.path.join('./result', save_path))
    with open(file=os.path.join('./result', save_path, 'result.txt'), mode='w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
    log_statics = {}
    if args.pretrained_ckpt:
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint)
        print('model load weights from {}'.format(args.pretrained_ckpt))
    else:
        print('model is trained from scratch')
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            itr += 1
            [anomaly_features, normaly_features], [anomaly_frame_labels, normaly_frame_labels, anomaly_video_labels, normaly_video_labels],  stastics_data = data
            features = torch.cat((anomaly_features.squeeze(0), normaly_features.squeeze(0)), dim=0)
            frame_labels = torch.cat((anomaly_frame_labels.squeeze(0), normaly_frame_labels.squeeze(0)), dim=0)
            video_labels = torch.cat((anomaly_video_labels.squeeze(0), normaly_video_labels.squeeze(0)), dim=0)

            features = features.to(device).permute(1, 0, 2)
            frame_labels = frame_labels.float().to(device).squeeze()
            video_labels = video_labels.to(device).squeeze()
            cls_scores, reg_scores = model(features)
            cls_loss = cp_loss(cls_scores, video_labels)
            reg_loss = mse_loss(reg_scores, frame_labels)


            total_loss = cls_loss_weight * cls_loss + mse_loss_weight * reg_loss
            logger.log_value('cls_loss', cls_loss, itr)
            logger.log_value('reg_loss', reg_loss, itr)
            if itr % 20 == 0 and not itr == 0:
                # print(final_features.shape)
                print('Iteration:{}, Loss: {}'
                      .format(itr, total_loss.data.cpu().detach().numpy()))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if itr % args.snapshot == 0 and not itr == 0:
                torch.save(model.state_dict(), os.path.join('./ckpt/', save_path, 'iter_{}'.format(itr) + '.pkl'))
                test_result_dict = test(test_loader, model, device, args)
                eval_p(itr=itr, dataset=args.dataset_name, predict_dict=test_result_dict, logger=logger, save_path=save_path,
                       plot=1, args=args)
            if itr > 12000:
                exit()

