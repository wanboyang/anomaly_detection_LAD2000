import os
import json
import glob

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




videos = glob.glob('/home/tu-wan/windowswan/dataset/LAD2000/denseflow_img/*/*/img')

rgb_frames = {}
for video in videos:
    video_name = video.split('/')[-2]
    video_rgb_frame = glob.glob(os.path.join(video, '*'))
    video_rgb_frame.sort()
    rgb_frames[video_name] = video_rgb_frame
with open(file='/home/tu-wan/windowswan/dataset/LAD2000/Frames.json', mode='w') as f:
    json.dump(rgb_frames,f)
