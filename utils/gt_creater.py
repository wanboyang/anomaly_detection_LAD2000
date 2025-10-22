"""
Ground truth creator for Anomaly Event Detection (AED) / 异常事件检测真实标签创建器
This script creates frame-level and video-level ground truth labels for the LAD2000 dataset.
此脚本为LAD2000数据集创建帧级和视频级真实标签。
"""

import pickle
import os
import glob
import scipy.io as sio
import numpy as np

# Dataset configuration / 数据集配置
dataset = 'LAD2000'
target_dir = '/home/tu-wan/windowswan/dataset/{}/GT/GT_mat/all'.format(dataset)
GT_dir = glob.glob('/home/tu-wan/windowswan/dataset/{}/GT/GT_mat/*/*/*.mat'.format(dataset))
frame_dirs = glob.glob('/home/tu-wan/windowswan/dataset/{}/denseflow_img/*/*/*'.format(dataset))

# Class dictionary mapping anomaly types to indices / 将异常类型映射到索引的类别字典
class_dict = {'Drop': 3, 'Loitering': 9, 'Crash': 0, 'Violence': 13, 'FallIntoWater': 4, 'Fire': 7, 'Fighting': 6, 'Crowd': 1, 'Destroy': 2, 'Falling': 5, 'Trampled': 12, 'Thiefing': 11, 'Panic': 10, 'Hurt': 8}

# classes = os.listdir(target_dir)
# classes.sort()
# class_index = 0
# class_dict = {}
# for c in classes:
#     class_dict[c] = class_index
#     class_index += 1

# gts = {}
# for gt in GT_dir:
#     gts[gt.split('/')[-1].split('.')[0]] = gt




# for frame_dir in frame_dirs:
#     video_name = frame_dir.split('/')[-2]
#     gt_name = video_name.replace('v_', 'gt_')
#     if gt_name in list(gts.keys()):
#         continue
#     else:
#         if video_name.split('_')[-3] != 'n':
#             print('{} errors'.format(video_name))
#             exit()
#         video_class = frame_dir.split('/')[-3]
#         frame_num = len(os.listdir(frame_dir))
#         label = np.zeros((1, frame_num))
#         label_name = os.path.join(target_dir, video_class, gt_name + '.mat')
#         sio.savemat(label_name,{'gt': label})

# Create dictionaries for frame-level and video-level labels / 创建帧级和视频级标签的字典
frame_label_dict = {}
video_label_dict = {}

# Process each frame directory to extract labels / 处理每个帧目录以提取标签
for frame_dir in frame_dirs:
    video_name = frame_dir.split('/')[-2]  # Extract video name / 提取视频名称
    gt_name = video_name.replace('v_', 'gt_')  # Convert to ground truth filename / 转换为真实标签文件名
    video_class = frame_dir.split('/')[-3]  # Extract video class / 提取视频类别
    
    # Load frame-level labels from .mat file / 从.mat文件加载帧级标签
    frame_label = sio.loadmat(os.path.join(target_dir, gt_name + '.mat'))['gt']  # [1, frame_num]
    
    # Get video-level label from class dictionary / 从类别字典获取视频级标签
    video_label = class_dict[video_class]
    
    # Store in dictionaries / 存储在字典中
    frame_label_dict[video_name] = frame_label
    video_label_dict[video_name] = video_label

# Save frame-level labels to pickle file / 将帧级标签保存到pickle文件
with open(file=os.path.join('/home/tu-wan/windowswan/dataset/LAD2000/GT', 'frame_label_IET.pickle'), mode='wb') as f:
    pickle.dump(frame_label_dict,f)

# Save video-level labels to pickle file / 将视频级标签保存到pickle文件
with open(file=os.path.join('/home/tu-wan/windowswan/dataset/LAD2000/GT', 'video_label_IET.pickle'), mode='wb') as f:
    pickle.dump(video_label_dict,f)
