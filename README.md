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


1.Download the original videos from (link: https://pan.baidu.com/s/1DIe-MU_Zww7IpVQjgBB3Lw pw: nfge)  or (link: https://stujxufeeducn-my.sharepoint.com/:f:/g/personal/2201810057_stu_jxufe_edu_cn/EqgRSGKhWJFNuKwak8CmC3QBC_Vp34KJ7vF48Fz7D_P3yA?e=K5YYOb pw：123456)

PS: Using the download link from BaiduDrive may encounter problems with some videos not being able to be downloaded. Therefore, we provide zip files containing all videos. (link: https://pan.baidu.com/s/1lQTqzqdA0dpi6Gl3ws9h5w pw:b1j8)

2.Extract i3d visual features for LAD2000, you can clone this project (https://github.com/wanboyang/anomly_feature.pytorch) and just set --dataset to LAD2000.

or you can directly use the i3d features of our LAD2000

1.Download the i3d features(link: https://pan.baidu.com/s/1rzEfdY3PBND-5O1ScxTTUQ pw: jkjz ) or （link:https://stujxufeeducn-my.sharepoint.com/:f:/g/personal/2201810057_stu_jxufe_edu_cn/ElFomOTAEi1NsH_Oa63VYbQB0xrPMQIdNUaXLX3U-BHPkg?e=sViE5H pw:123456） and unzip the i3d.zip.

2.change the "dataset_path" to "you/path/i3d"

For LAD2000, Auenve, ped2, shanghaitech and UCF_Crime, we provide the full-supervised data splits and groundtruth in (link:https://stujxufeeducn-my.sharepoint.com/:u:/g/personal/2201810057_stu_jxufe_edu_cn/EQBo6YEqwLpMq0BwhIIu4KUBjZ5Cof2s96h_ebJQTCrcDA?e=fhtbs8 pw:123456 
link:https://stujxufeeducn-my.sharepoint.com/:u:/g/personal/2201810057_stu_jxufe_edu_cn/EY22ebuTjM5LvmTIUlLjg2UBZCyskwMeNaCIu5zQjrNqHQ?e=PmYs8v pw:123456)

For Auenve, ped2, shanghaitech and UCF_Crime, we provide the i3d features in (link: https://pan.baidu.com/s/1fYAlFoTdcg8BgRdqoLQ2Bg pw: njy2) or
(link:https://stujxufeeducn-my.sharepoint.com/:f:/g/personal/2201810057_stu_jxufe_edu_cn/EuQvbLCDoIxLgmcgpJRqcbIBzJSc7D6V-q151gLsWyFTrQ?e=JNbUEc pw:123456)

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
