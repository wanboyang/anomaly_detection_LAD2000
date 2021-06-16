import pickle
import os
import glob
import scipy.io as sio
import numpy as np

dataset = 'shanghaitech'

GT_dir = os.listdir('/home/tu-wan/windowswan/dataset/shanghaitech/gt/f1')
frame_dirs = glob.glob('/home/tu-wan/windowswan/dataset/shanghaitech/features_video/i3d/combine/*')


frame_label_dict = {}
video_label_dict = {}
for frame_dir in frame_dirs:
    feature = np.load(os.path.join(frame_dir, 'feature.npy'))
    frame_num = feature.shape[0] * 16
    video_name = frame_dir.split('/')[-1]
    if video_name + '.npy' in GT_dir:
        frame_label = np.load(os.path.join('/home/tu-wan/windowswan/dataset/shanghaitech/gt/f1', video_name + '.npy'))
        frame_label = frame_label[np.newaxis,:]
        if frame_label.sum():
            video_label = 1
        else:
            video_label = 0
    else:
        frame_label = np.zeros((1,frame_num))
        video_label = 0



    frame_label_dict[video_name] = frame_label
    video_label_dict[video_name] = video_label


with open(file=os.path.join('/home/tu-wan/windowswan/dataset/{}/GT'.format(dataset), 'frame_label_IET.pickle'), mode='wb') as f:
    pickle.dump(frame_label_dict,f)
with open(file=os.path.join('/home/tu-wan/windowswan/dataset/{}/GT'.format(dataset), 'video_label_IET.pickle'), mode='wb') as f:
    pickle.dump(video_label_dict,f)










