import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import sys
from utils import scorebinary, anomap


def eval_p(itr, dataset, predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):
    [ano_result, cls_result, extend_numbers] = predict_dict
    global label_dict_path
    if manual:
        save_root = './manul_test_result'
    else:
        save_root = './result'
    if dataset == 'shanghaitech':
        label_dict_path = '{}/shanghaitech/GT'.format(args.dataset_path)
    elif dataset == 'LAD2000':
        label_dict_path = '{}/LAD2000/GT'.format(args.dataset_path)
    elif dataset == 'UCF_Crime':
        label_dict_path = '{}/UCF_Crime/GT'.format(args.dataset_path)
    elif dataset == 'Avenue':
        label_dict_path = '{}/Avenue/GT'.format(args.dataset_path)
    elif dataset == 'UCSDPed2':
        label_dict_path = '{}/UCSDPed2/GT'.format(args.dataset_path)
    elif dataset == 'LV':
        label_dict_path = '{}/LV/GT'.format(args.dataset_path)
    else:
        raise ValueError


    with open(file=os.path.join(label_dict_path, 'frame_label_IET.pickle'), mode='rb') as f:
        frame_label_dict = pickle.load(f)
    with open(file=os.path.join(label_dict_path, 'video_label_IET.pickle'), mode='rb') as f:
        video_label_dict = pickle.load(f)

    all_ano_predict_np = np.zeros(0)
    all_ano_label_np = np.zeros(0)
    normal_ano_predict_np = np.zeros(0)
    normal_ano_label_np = np.zeros(0)
    abnormal_ano_predict_np = np.zeros(0)
    abnormal_ano_label_np = np.zeros(0)
    all_cls_predict_np = np.zeros(0)
    all_cls_label_np = np.zeros(0)
    for video_names, ano_scores in ano_result.items():
        cls_scores = cls_result[video_names]
        extend_number = extend_numbers[video_names]
        frame_label = frame_label_dict[video_names]
        video_label = np.asarray(video_label_dict[video_names])
        clip_number = len(ano_scores) // (16 * args.max_seqlen)
        if extend_number[0]:
            ano_index = (clip_number - 1) * (16 * args.max_seqlen)
        else:
            ano_index = clip_number * (16 * args.max_seqlen)
        if ano_scores[:ano_index].shape != frame_label[0, :ano_index].shape:
            print(video_names,ano_scores[:ano_index].shape,frame_label[0, :ano_index].shape)
        if frame_label.sum():
            all_ano_predict_np = np.concatenate((all_ano_predict_np, ano_scores[:ano_index]))
            all_ano_label_np = np.concatenate((all_ano_label_np, frame_label[0, :ano_index]))
            abnormal_ano_predict_np = np.concatenate((abnormal_ano_predict_np, ano_scores[:ano_index]))
            abnormal_ano_label_np = np.concatenate((abnormal_ano_label_np, frame_label[0, :ano_index]))
        else:
            all_ano_predict_np = np.concatenate((all_ano_predict_np, ano_scores[:ano_index]))
            all_ano_label_np = np.concatenate((all_ano_label_np, frame_label[0, :ano_index]))

            normal_ano_predict_np = np.concatenate((normal_ano_predict_np, ano_scores[:ano_index]))
            normal_ano_label_np = np.concatenate((normal_ano_label_np, frame_label[0, :ano_index]))
        all_cls_predict_np = np.concatenate((all_cls_predict_np, cls_scores))
        all_cls_label_np = np.concatenate((all_cls_label_np, video_label.repeat(len(cls_scores))))
    ano_category_acc = accuracy_score(y_true=all_cls_label_np, y_pred=all_cls_predict_np)
    all_auc_score = roc_auc_score(y_true=all_ano_label_np, y_score=all_ano_predict_np)
    binary_all_predict_np = scorebinary(all_ano_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=all_ano_label_np, y_pred=binary_all_predict_np).ravel()
    all_ano_false_alarm = fp / (fp + tn)
    binary_normal_predict_np = scorebinary(normal_ano_predict_np, threshold=0.5)
    # tn, fp, fn, tp = confusion_matrix(y_true=normal_label_np, y_pred=binary_normal_predict_np).ravel()
    fp_n = binary_normal_predict_np.sum()
    normal_count = normal_ano_predict_np.shape[0]
    normal_ano_false_alarm = fp_n / normal_count

    abnormal_auc_score = roc_auc_score(y_true=abnormal_ano_label_np, y_score= abnormal_ano_predict_np)
    binary_abnormal_predict_np = scorebinary(abnormal_ano_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=abnormal_ano_label_np, y_pred=binary_abnormal_predict_np).ravel()
    abnormal_ano_false_alarm = fp / (fp + tn)
    print('Iteration: {} CLS_score_all_video is {}'.format(itr, ano_category_acc))
    print('Iteration: {} AUC_score_all_video is {}'.format(itr, all_auc_score))
    print('Iteration: {} AUC_score_abnormal_video is {}'.format(itr, abnormal_auc_score))
    print('Iteration: {} ano_false_alarm_all_video is {}'.format(itr, all_ano_false_alarm))
    print('Iteration: {} ano_false_alarm_normal_video is {}'.format(itr, normal_ano_false_alarm))
    print('Iteration: {} ano_false_alarm_abnormal_video is {}'.format(itr, abnormal_ano_false_alarm))
    if plot:
        anomap(ano_result, cls_result, frame_label_dict, save_path, itr, save_root, extend_numbers  , args, zip)
    if logger:
        logger.log_value('Test_CLS_all_video', ano_category_acc, itr)
        logger.log_value('Test_AUC_all_video', all_auc_score, itr)
        logger.log_value('Test_AUC_abnormal_video', abnormal_auc_score, itr)
        logger.log_value('Test_false_alarm_all_video', all_ano_false_alarm, itr)
        logger.log_value('Test_false_alarm_normal_video', normal_ano_false_alarm, itr)
        logger.log_value('Test_false_alarm_abnormal_video', abnormal_ano_false_alarm, itr)
    if os.path.exists(os.path.join(save_root, save_path)) == 0:
        os.makedirs(os.path.join(save_root, save_path))
    with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:
        f.write('itration_{}_CLS_Score_all_video is {}\n'.format(itr, ano_category_acc))
        f.write('itration_{}_AUC_Score_all_video is {}\n'.format(itr, all_auc_score))
        f.write('itration_{}_AUC_Score_abnormal_video is {}\n'.format(itr, abnormal_auc_score))
        f.write('itration_{}_ano_false_alarm_all_video is {}\n'.format(itr, all_ano_false_alarm))
        f.write('itration_{}_ano_false_alarm_normal_video is {}\n'.format(itr, normal_ano_false_alarm))
        f.write('itration_{}_ano_false_alarm_abnormal_video is {}\n'.format(itr, abnormal_ano_false_alarm))




