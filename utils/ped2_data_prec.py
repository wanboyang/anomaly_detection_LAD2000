import os
import json
import glob
import random
import scipy.io as sio
import numpy as np
import pickle

def json_reader(jsonfile):
    with open (jsonfile, 'r') as f:
        dict = json.load(f)
    return dict

def list_reader(txt):
    with open(txt, 'r') as f:
        list = f.readlines()
    return list

def list_writer(list,txt):
    with open(txt, 'w') as f:
        for i in list:
            f.writelines(str(i)+'\n')

def dict_save2_json(dict,savename):
    with open(savename, 'w') as f:
        json.dump(dict, f)

def dict_sort(dict, order='descend'):
    if order == 'descend':
        dict_sorted = sorted(dict.items(), key=lambda d:d[1], reverse=True)
        return dict_sorted
    elif order == 'ascend':
        dict_sorted = sorted(dict.items(), key=lambda d: d[1])
        return dict_sorted
    else:
        print('only descend or ascend is supported')
        exit()

if __name__ == '__main__':
    dataset = 'UCSDPed2'
    features_video_dir = '/home/tu-wan/windowswan/dataset/UCSDPed2/features_video/i3d/combine'
    video_names = os.listdir(features_video_dir)
    abnormal_videos = []
    normal_videos = []
    for v in video_names:
        if v.rfind('_a_') != -1:
            abnormal_videos.append(v)
        else:
            normal_videos.append(v)
    if os.path.exists('/home/tu-wan/windowswan/dataset/UCSDPed2/train_split.txt') == 0:
        random.shuffle(abnormal_videos)
        random.shuffle(normal_videos)
        train_list = abnormal_videos[:6] + normal_videos[:8]
        test_list = abnormal_videos[6:] + normal_videos[8:]
        list_writer(list=train_list, txt='/home/tu-wan/windowswan/dataset/UCSDPed2/train_split.txt')
        list_writer(list=test_list, txt='/home/tu-wan/windowswan/dataset/UCSDPed2/test_split.txt')
    GT_dir = '/home/tu-wan/windowswan/dataset/UCSDPed2/GT/frame_level_mat/'
    frame_dirs = glob.glob('/home/tu-wan/windowswan/dataset/UCSDPed2/denseflow_img/*/img')
    frame_label_dict = {}
    video_label_dict = {}
    for frame_dir in frame_dirs:
        video_name = frame_dir.split('/')[-2]
        if video_name.rfind('_a_') != -1:
            label_file = 'Test' + video_name[-3:] + '.mat'
            label_file_data = np.transpose(sio.loadmat(os.path.join(GT_dir, label_file))['gt'], [1, 0])
            video_label = 1
        else:
            label_file_data = np.zeros((1, len(os.listdir(frame_dir))))
            video_label = 0
        frame_label_dict[video_name] = label_file_data
        video_label_dict[video_name] = video_label
    with open(file=os.path.join('/home/tu-wan/windowswan/dataset/{}/GT'.format(dataset), 'frame_label_IET.pickle'),
              mode='wb') as f:
        pickle.dump(frame_label_dict, f)
    with open(file=os.path.join('/home/tu-wan/windowswan/dataset/{}/GT'.format(dataset), 'video_label_IET.pickle'),
              mode='wb') as f:
        pickle.dump(video_label_dict, f)


