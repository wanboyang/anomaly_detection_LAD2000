import pickle
import os
import glob
import scipy.io as sio
import numpy as np

dataset = 'LAD2000'
target_dir = '/home/tu-wan/windowswan/dataset/{}/GT/GT_mat/all'.format(dataset)
GT_dir = glob.glob('/home/tu-wan/windowswan/dataset/{}/GT/GT_mat/*/*/*.mat'.format(dataset))
frame_dirs = glob.glob('/home/tu-wan/windowswan/dataset/{}/denseflow_img/*/*/*'.format(dataset))
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

frame_label_dict = {}
video_label_dict = {}
for frame_dir in frame_dirs:
    video_name = frame_dir.split('/')[-2]
    gt_name = video_name.replace('v_', 'gt_')
    video_class = frame_dir.split('/')[-3]
    frame_label = sio.loadmat(os.path.join(target_dir, gt_name + '.mat'))['gt'] # [1, frame_num]
    video_label = class_dict[video_class]
    frame_label_dict[video_name] = frame_label
    video_label_dict[video_name] = video_label


with open(file=os.path.join('/home/tu-wan/windowswan/dataset/LAD2000/GT', 'frame_label_IET.pickle'), mode='wb') as f:
    pickle.dump(frame_label_dict,f)
with open(file=os.path.join('/home/tu-wan/windowswan/dataset/LAD2000/GT', 'video_label_IET.pickle'), mode='wb') as f:
    pickle.dump(video_label_dict,f)










