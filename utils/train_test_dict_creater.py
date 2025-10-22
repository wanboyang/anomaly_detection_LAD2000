import os
import json
import glob

"""
Utility functions for data processing and dictionary operations / 数据处理和字典操作的工具函数
This module contains helper functions for reading/writing JSON files, lists, and dictionary operations.
此模块包含用于读取/写入JSON文件、列表和字典操作的辅助函数。
"""

def json_reader(jsonfile):
    """
    Read JSON file and return dictionary / 读取JSON文件并返回字典
    
    Args:
        jsonfile: Path to JSON file / JSON文件路径
    
    Returns:
        Dictionary loaded from JSON file / 从JSON文件加载的字典
    """
    with open (jsonfile, 'r') as f:
        dict = json.load(f)
    return dict

def list_reader(txt):
    """
    Read text file and return list of lines / 读取文本文件并返回行列表
    
    Args:
        txt: Path to text file / 文本文件路径
    
    Returns:
        List of lines from text file / 文本文件中的行列表
    """
    with open(txt, 'r') as f:
        list = f.readlines()
    return list

def list_writer(list,txt):
    """
    Write list to text file / 将列表写入文本文件
    
    Args:
        list: List to write / 要写入的列表
        txt: Path to output text file / 输出文本文件路径
    """
    with open(txt, 'w') as f:
        for i in list:
            f.writelines(str(i)+'\n')

def dict_save2_json(dict,savename):
    """
    Save dictionary to JSON file / 将字典保存到JSON文件
    
    Args:
        dict: Dictionary to save / 要保存的字典
        savename: Path to output JSON file / 输出JSON文件路径
    """
    with open(savename, 'w') as f:
        json.dump(dict, f)


def dict_sort(dict, order='descend'):
    """
    Sort dictionary by values / 按值对字典进行排序
    
    Args:
        dict: Dictionary to sort / 要排序的字典
        order: Sorting order ('descend' or 'ascend') / 排序顺序（'descend' 或 'ascend'）
    
    Returns:
        Sorted list of dictionary items / 排序后的字典项列表
    """
    if order == 'descend':
        dict_sorted = sorted(dict.items(), key=lambda d:d[1], reverse=True)
        return dict_sorted
    elif order == 'ascend':
        dict_sorted = sorted(dict.items(), key=lambda d: d[1])
        return dict_sorted
    else:
        print('only descend or ascend is supported')
        exit()




# Example usage: Create frame dictionary for LAD2000 dataset / 示例用法：为LAD2000数据集创建帧字典
videos = glob.glob('/home/tu-wan/windowswan/dataset/LAD2000/denseflow_img/*/*/img')

rgb_frames = {}
for video in videos:
    video_name = video.split('/')[-2]  # Extract video name from path / 从路径提取视频名称
    video_rgb_frame = glob.glob(os.path.join(video, '*'))  # Get all frame files / 获取所有帧文件
    video_rgb_frame.sort()  # Sort frames in order / 按顺序排序帧
    rgb_frames[video_name] = video_rgb_frame  # Add to dictionary / 添加到字典

# Save frame dictionary to JSON file / 将帧字典保存到JSON文件
with open(file='/home/tu-wan/windowswan/dataset/LAD2000/Frames.json', mode='w') as f:
    json.dump(rgb_frames,f)
