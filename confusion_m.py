import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import sys
from utils import scorebinary, anomap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns



def cm(itr, dataset, predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):
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

    all_cls_predict_np = np.zeros(0)
    all_cls_label_np = np.zeros(0)
    for video_names, ano_scores in ano_result.items():
        cls_scores = cls_result[video_names]
        video_label = np.asarray(video_label_dict[video_names])
        all_cls_predict_np = np.concatenate((all_cls_predict_np, cls_scores))
        all_cls_label_np = np.concatenate((all_cls_label_np, video_label.repeat(len(cls_scores))))
    ano_category_acc = accuracy_score(y_true=all_cls_label_np, y_pred=all_cls_predict_np)
    c_matrix = confusion_matrix(y_true=all_cls_label_np, y_pred=all_cls_predict_np)
    c_matrix_norm = c_matrix.astype('float') /c_matrix.sum(axis=1)[:, np.newaxis]
    class_dict = {'Drop': 3, 'Loitering': 9, 'Crash': 0, 'Violence': 13, 'FallIntoWater': 4, 'Fire': 7, 'Fighting': 6,
                  'Crowd': 1, 'Destroy': 2, 'Falling': 5, 'Trampled': 12, 'Thiefing': 11, 'Panic': 10, 'Hurt': 8}
    label_index = np.zeros(shape=(len(class_dict))).tolist()
    for c, i in class_dict.items():
        label_index[i] = c

    xticklabels = label_index
    yticklabels = label_index
    f1, ax1 = plt.subplots(figsize=(len(class_dict), len(class_dict)))
    sns.heatmap(c_matrix_norm * 100, xticklabels=xticklabels, yticklabels=yticklabels, annot=True, ax=ax1, fmt='.1f',annot_kws={"size":16})
    label_y = ax1.get_yticklabels()
    plt.setp(label_y, rotation=45)
    label_x = ax1.get_xticklabels()
    plt.setp(label_x, rotation=45)
    image_name_bij = './result/' + save_path + '/matrix.jpg'
    # plt.show()
    f1.savefig(fname=image_name_bij, dpi=300)
    # f1.close()












