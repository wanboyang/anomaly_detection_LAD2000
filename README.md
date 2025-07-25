# Introduction
This repository is established for Anomaly detection in video sequences: A benchmark and computational model. In this repository, we provide a video anomaly detection database named LAD2000, which contains 2000 videos and 14 anomaly categories. The original paper can be found [here](https://arxiv.org/abs/2106.08570).

If this work have some benefits to your research, please cite with the following BibTeX:

```
@article{wan2021anomaly,
  title={Anomaly detection in video sequences: A benchmark and computational model},
  author={Wan, Boyang and Jiang, Wenhui and Fang, Yuming and Luo, Zhiyuan and Ding, Guanqun},
  journal={IET Image Processing},
  year={2021},
  publisher={Wiley Online Library}
}
```


## Requirements
* Python 3
* CUDA
* numpy
* tqdm
* [PyTorch](http://pytorch.org/) (1.2)
* [torchvision](http://pytorch.org/)  
Recommend: the environment can be established by run

```
conda env create -f environment.yaml
```


## Data preparation


1.Downloading the original videos from "videos_compressed" dir in (https://pan.baidu.com/s/1LmNAWnR-RPqo-azCgASvfg password: avt8).

ps: The RGB and Flow frames of the original videos are provided in the "denseflow" dir of (https://pan.baidu.com/s/1LmNAWnR-RPqo-azCgASvfg password: avt8).


2.Extract i3d visual features for LAD2000, you can clone this project (https://github.com/wanboyang/anomly_feature.pytorch) and just set --dataset to LAD2000.

or you can directly use the i3d features of our LAD2000

1.Download the i3d features in the "features" dir of (https://pan.baidu.com/s/1LmNAWnR-RPqo-azCgASvfg password: avt8).

2.change the "dataset_path" to "you/path/i3d"

For LAD2000, Auenve, ped2, shanghaitech and UCF_Crime, we provide the full-supervised data splits and groundtruth by data_splits.zip and GTs.zip in (https://pan.baidu.com/s/1LmNAWnR-RPqo-azCgASvfg password: avt8).

For Auenve, ped2, shanghaitech, and UCF_Crime, we provide the i3d features in (link: https://pan.baidu.com/s/1fYAlFoTdcg8BgRdqoLQ2Bg pw: njy2).

## Class_index
```
class_dict = {'Drop': 3, 'Loitering': 9, 'Crash': 0, 'Violence': 13, 'FallIntoWater': 4, 'Fire': 7, 'Fighting': 6, 'Crowd': 1, 'Destroy': 2, 'Falling': 5, 'Trampled': 12, 'Thiefing': 11, 'Panic': 10, 'Hurt': 8}

```

## Training

For LAD2000 database:

```
sh LAD2000T_i3d.sh
```

For ped2 database:

```
sh ped2_i3d.sh
```

For shanghaitech database:

```
sh ped2_i3d.sh
```

For Avenue database:

```
sh Avenue_i3d.sh
```

For UCF_Crime database:


```
sh UCF_i3d.sh
```

The models and testing results will be created on ./ckpt and ./results respectively

## Acknowledgements
Thanks the contribution of [W-TALC](https://github.com/sujoyp/wtalc-pytorch) and awesome PyTorch team.

## Contact
Please contact the first author of the associated paper - Boyang Wan （wanboyangjerry@163.com) for any further queries.
