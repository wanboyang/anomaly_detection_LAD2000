import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import sys
from utils import scorebinary, anomap

"""
Evaluation functions for Anomaly Event Detection (AED) / 异常事件检测评估函数
This module contains functions for evaluating model performance and generating metrics.
此模块包含评估模型性能和生成指标的函数。
"""

def eval_p(itr, dataset, predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):
    """
    Evaluate model performance and generate metrics / 评估模型性能并生成指标
    
    Args:
        itr: Iteration number / 迭代次数
        dataset: Dataset name / 数据集名称
        predict_dict: Dictionary containing prediction results / 包含预测结果的字典
        logger: Tensorboard logger / Tensorboard日志记录器
        save_path: Path to save results / 保存结果的路径
        args: Command line arguments / 命令行参数
        plot: Whether to generate visualization plots / 是否生成可视化图
        zip: Whether to save plots to zip file / 是否将图保存到zip文件
        manual: Whether this is a manual test / 是否为手动测试
    """
    [ano_result, cls_result, extend_numbers] = predict_dict
    global label_dict_path
    
    # Set save root directory / 设置保存根目录
    if manual:
        save_root = './manul_test_result'
    else:
        save_root = './result'
    
    # Set label dictionary path based on dataset / 根据数据集设置标签字典路径
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
        raise ValueError("Unknown dataset: {}".format(dataset))

    # Load ground truth labels / 加载真实标签
    with open(file=os.path.join(label_dict_path, 'frame_label_IET.pickle'), mode='rb') as f:
        frame_label_dict = pickle.load(f)
    with open(file=os.path.join(label_dict_path, 'video_label_IET.pickle'), mode='rb') as f:
        video_label_dict = pickle.load(f)

    # Initialize arrays for storing predictions and labels / 初始化存储预测和标签的数组
    all_ano_predict_np = np.zeros(0)  # All anomaly predictions / 所有异常预测
    all_ano_label_np = np.zeros(0)    # All anomaly labels / 所有异常标签
    normal_ano_predict_np = np.zeros(0)  # Normal video predictions / 正常视频预测
    normal_ano_label_np = np.zeros(0)    # Normal video labels / 正常视频标签
    abnormal_ano_predict_np = np.zeros(0)  # Abnormal video predictions / 异常视频预测
    abnormal_ano_label_np = np.zeros(0)    # Abnormal video labels / 异常视频标签
    all_cls_predict_np = np.zeros(0)  # All classification predictions / 所有分类预测
    all_cls_label_np = np.zeros(0)    # All classification labels / 所有分类标签
    
    # Process each video in the prediction results / 处理预测结果中的每个视频
    for video_names, ano_scores in ano_result.items():
        cls_scores = cls_result[video_names]
        extend_number = extend_numbers[video_names]
        frame_label = frame_label_dict[video_names]
        video_label = np.asarray(video_label_dict[video_names])
        
        # Calculate clip number and anomaly index / 计算片段数和异常索引
        clip_number = len(ano_scores) // (16 * args.max_seqlen)
        if extend_number[0]:
            ano_index = (clip_number - 1) * (16 * args.max_seqlen)
        else:
            ano_index = clip_number * (16 * args.max_seqlen)
        
        # Check shape consistency / 检查形状一致性
        if ano_scores[:ano_index].shape != frame_label[0, :ano_index].shape:
            print(video_names,ano_scores[:ano_index].shape,frame_label[0, :ano_index].shape)
        
        # Separate normal and abnormal videos / 分离正常和异常视频
        if frame_label.sum():  # Abnormal video / 异常视频
            all_ano_predict_np = np.concatenate((all_ano_predict_np, ano_scores[:ano_index]))
            all_ano_label_np = np.concatenate((all_ano_label_np, frame_label[0, :ano_index]))
            abnormal_ano_predict_np = np.concatenate((abnormal_ano_predict_np, ano_scores[:ano_index]))
            abnormal_ano_label_np = np.concatenate((abnormal_ano_label_np, frame_label[0, :ano_index]))
        else:  # Normal video / 正常视频
            all_ano_predict_np = np.concatenate((all_ano_predict_np, ano_scores[:ano_index]))
            all_ano_label_np = np.concatenate((all_ano_label_np, frame_label[0, :ano_index]))
            normal_ano_predict_np = np.concatenate((normal_ano_predict_np, ano_scores[:ano_index]))
            normal_ano_label_np = np.concatenate((normal_ano_label_np, frame_label[0, :ano_index]))
        
        # Collect classification results / 收集分类结果
        all_cls_predict_np = np.concatenate((all_cls_predict_np, cls_scores))
        all_cls_label_np = np.concatenate((all_cls_label_np, video_label.repeat(len(cls_scores))))
    
    # Calculate evaluation metrics / 计算评估指标
    ano_category_acc = accuracy_score(y_true=all_cls_label_np, y_pred=all_cls_predict_np)  # Classification accuracy / 分类准确率
    all_auc_score = roc_auc_score(y_true=all_ano_label_np, y_score=all_ano_predict_np)  # Overall AUC / 总体AUC
    
    # Calculate false alarm rates / 计算误报率
    binary_all_predict_np = scorebinary(all_ano_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=all_ano_label_np, y_pred=binary_all_predict_np).ravel()
    all_ano_false_alarm = fp / (fp + tn)  # False alarm rate for all videos / 所有视频的误报率
    
    binary_normal_predict_np = scorebinary(normal_ano_predict_np, threshold=0.5)
    fp_n = binary_normal_predict_np.sum()
    normal_count = normal_ano_predict_np.shape[0]
    normal_ano_false_alarm = fp_n / normal_count  # False alarm rate for normal videos / 正常视频的误报率

    abnormal_auc_score = roc_auc_score(y_true=abnormal_ano_label_np, y_score= abnormal_ano_predict_np)  # AUC for abnormal videos / 异常视频的AUC
    binary_abnormal_predict_np = scorebinary(abnormal_ano_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=abnormal_ano_label_np, y_pred=binary_abnormal_predict_np).ravel()
    abnormal_ano_false_alarm = fp / (fp + tn)  # False alarm rate for abnormal videos / 异常视频的误报率
    
    # Print evaluation results / 打印评估结果
    print('Iteration: {} CLS_score_all_video is {}'.format(itr, ano_category_acc))
    print('Iteration: {} AUC_score_all_video is {}'.format(itr, all_auc_score))
    print('Iteration: {} AUC_score_abnormal_video is {}'.format(itr, abnormal_auc_score))
    print('Iteration: {} ano_false_alarm_all_video is {}'.format(itr, all_ano_false_alarm))
    print('Iteration: {} ano_false_alarm_normal_video is {}'.format(itr, normal_ano_false_alarm))
    print('Iteration: {} ano_false_alarm_abnormal_video is {}'.format(itr, abnormal_ano_false_alarm))
    
    # Generate visualization plots if requested / 如果请求则生成可视化图
    if plot:
        anomap(ano_result, cls_result, frame_label_dict, save_path, itr, save_root, extend_numbers, args, zip)
    
    # Log metrics to tensorboard / 将指标记录到tensorboard
    if logger:
        logger.log_value('Test_CLS_all_video', ano_category_acc, itr)
        logger.log_value('Test_AUC_all_video', all_auc_score, itr)
        logger.log_value('Test_AUC_abnormal_video', abnormal_auc_score, itr)
        logger.log_value('Test_false_alarm_all_video', all_ano_false_alarm, itr)
        logger.log_value('Test_false_alarm_normal_video', normal_ano_false_alarm, itr)
        logger.log_value('Test_false_alarm_abnormal_video', abnormal_ano_false_alarm, itr)
    
    # Save results to file / 将结果保存到文件
    if os.path.exists(os.path.join(save_root, save_path)) == 0:
        os.makedirs(os.path.join(save_root, save_path))
    with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:
        f.write('itration_{}_CLS_Score_all_video is {}\n'.format(itr, ano_category_acc))
        f.write('itration_{}_AUC_Score_all_video is {}\n'.format(itr, all_auc_score))
        f.write('itration_{}_AUC_Score_abnormal_video is {}\n'.format(itr, abnormal_auc_score))
        f.write('itration_{}_ano_false_alarm_all_video is {}\n'.format(itr, all_ano_false_alarm))
        f.write('itration_{}_ano_false_alarm_normal_video is {}\n'.format(itr, normal_ano_false_alarm))
        f.write('itration_{}_ano_false_alarm_abnormal_video is {}\n'.format(itr, abnormal_ano_false_alarm))
