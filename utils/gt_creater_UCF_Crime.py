import pickle
import os
import glob
import scipy.io as sio
import numpy as np

dataset = 'UCF_Crime'
target_dir = '/home/tu-wan/windows4t/dataset/UCF_Crime/denseflow'
GT_dir = os.listdir('/home/tu-wan/windowswan/dataset/UCF_Crime/GT/Numpy_formate')
frame_dirs = glob.glob('/home/tu-wan/windowswan/dataset/UCF_Crime/features_video/i3d/combine/*')
classes = os.listdir(target_dir)
classes.sort()
class_index = 0
class_dict = {}
for c in classes:
    if c.rfind('Normal') == -1:
        class_dict[c] = class_index
        class_index += 1

class_dict['Normal'] = class_index

frame_label_dict = {}
video_label_dict = {}
for frame_dir in frame_dirs:
    feature = np.load(os.path.join(frame_dir, 'feature.npy'))
    frame_num = feature.shape[0] * 16
    video_name = frame_dir.split('/')[-1]
    video_class = video_name[:-8]
    if video_name + '.npy' in GT_dir:
        frame_label = np.load(os.path.join('/home/tu-wan/windowswan/dataset/UCF_Crime/GT/Numpy_formate', video_name + '.npy'))
        frame_label = frame_label[np.newaxis,:]
        if video_name.rfind('Normal') == -1:
            video_label = class_dict[video_class]
        else:
            video_label = class_dict['Normal']
    else:
        if video_name.rfind('Normal') == -1:
            video_label = class_dict[video_class]
            frame_label = np.ones((1,frame_num))
        else:
            frame_label = np.zeros((1,frame_num))
            video_label = class_dict['Normal']



    frame_label_dict[video_name] = frame_label
    video_label_dict[video_name] = video_label


with open(file=os.path.join('/home/tu-wan/windowswan/dataset/{}/GT'.format(dataset), 'frame_label_IET.pickle'), mode='wb') as f:
    pickle.dump(frame_label_dict,f)
with open(file=os.path.join('/home/tu-wan/windowswan/dataset/{}/GT'.format(dataset), 'video_label_IET.pickle'), mode='wb') as f:
    pickle.dump(video_label_dict,f)










