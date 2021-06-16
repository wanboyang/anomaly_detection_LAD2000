import torch
# import torch.nn.functional as F
# import utils
# import numpy as np
# from torch.autograd import Variable
# import scipy.io as sio

def test(test_loader, model, device, args):
    ano_result = {}
    cls_result = {}
    extend_numbers = {}
    for i, data in enumerate(test_loader):
        features, data_video_name = data
        features = features.squeeze()

        features = features.to(device)
        features_overlap = features.new_zeros(0)
        if features.shape[0] % args.max_seqlen != 0:
            extend_number = args.max_seqlen - (features.shape[0] % args.max_seqlen)
            extend_tensor = features[-1].unsqueeze(0).repeat(extend_number, 1)
            features = torch.cat((features, extend_tensor), 0)
        else:
            extend_number = 0
        clip_num = len(features)
        for i in range(0, len(features)-args.max_seqlen + 1):
            features_overlap = torch.cat((features_overlap, features[i: i+args.max_seqlen]), 0)
        features = features.view(args.max_seqlen, features.shape[-2]//args.max_seqlen, features.shape[-1])
        features_overlap = features_overlap.view(args.max_seqlen, features_overlap.shape[-2]//args.max_seqlen, features_overlap.shape[-1])
        ano_out = features_overlap.new_zeros((features_overlap.shape[-2], 16*clip_num))


        with torch.no_grad():
            cls_scores, reg_scores = model(features_overlap)
        for i_index, reg_score in enumerate(reg_scores):
            j_index = 16 * i_index
            ano_out[i_index, j_index:j_index + (args.max_seqlen * 16)] = reg_score
        ano_out = torch.max(ano_out, 0)[0]
        # ano_out = torch.median(ano_out, 0)


        _, predicted = torch.max(cls_scores.data, 1)
        predicted = predicted.cpu().data.numpy().reshape(-1)
        reg_scores = ano_out.cpu().data.numpy().reshape(-1)
        ano_result[data_video_name[0]] = reg_scores
        cls_result[data_video_name[0]] = predicted
        extend_numbers[data_video_name[0]] = [extend_number]
    return [ano_result, cls_result, extend_numbers]




