# LAD2000: 大规模视频异常检测基准与计算模型

[![论文](https://img.shields.io/badge/论文-ArXiv-red)](https://arxiv.org/abs/2106.08570)
[![许可证](https://img.shields.io/badge/许可证-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.5%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange)](https://pytorch.org/)

> **语言**: [English](README.md) | [中文](README_CN.md)

---

## 摘要

本仓库提出了 **LAD2000**，一个包含 2,000 个视频、涵盖 14 种不同异常类别的综合性视频异常检测基准。我们引入了一种新颖的视频异常检测计算框架，利用基于 ConvLSTM 的架构同时执行异常分类和时间定位。我们的方法在包括 LAD2000、Avenue、Ped2、ShanghaiTech 和 UCF-Crime 在内的多个基准数据集上展示了最先进的性能。

## 🎯 主要特性

- **大规模数据集**: 2,000 个视频，14 种异常类别
- **多模态特征**: 支持 RGB、Flow 和组合 I3D 特征
- **双任务学习**: 同时进行异常分类和时间定位
- **全面基准测试**: 在 5 个主要视频异常检测数据集上的评估
- **可复现实现**: 完整的训练和评估流程

## 📊 数据集概览

| 类别 | 视频数量 | 描述 |
|------|----------|------|
| 碰撞 | 143 | 车辆碰撞和事故 |
| 人群聚集 | 156 | 异常人群聚集 |
| 破坏 | 142 | 财产破坏 |
| 坠落 | 145 | 物体从高处坠落 |
| 摔倒 | 148 | 人员摔倒 |
| 落水 | 139 | 落入水体 |
| 斗殴 | 147 | 肢体冲突 |
| 火灾 | 144 | 火灾事件 |
| 受伤 | 146 | 身体伤害 |
| 徘徊 | 142 | 可疑逗留 |
| 恐慌 | 143 | 恐慌情况 |
| 偷窃 | 145 | 盗窃活动 |
| 踩踏 | 141 | 人群踩踏 |
| 暴力 | 129 | 暴力行为 |

## 🏗️ 模型架构

我们提出的 **AED (异常事件检测)** 框架包含：

1. **ConvLSTM 编码器**: 捕获时空依赖关系
2. **分类头**: 预测异常类别
3. **回归头**: 定位时间片段

### 模型变体:
- **AED**: 单层 ConvLSTM
- **AED_T**: 双层 ConvLSTM，增强时间建模能力

## 📁 项目结构

```
anomaly_detection_LAD2000/
├── model.py                    # ConvLSTM模型架构和分类/回归头
├── options.py                  # 命令行参数解析器
├── main.py                     # 项目主入口和训练环境设置
├── train.py                    # 主要训练循环和损失计算
├── test.py                     # 模型测试和预测函数
├── losses.py                   # 自定义损失函数集合
├── utils.py                    # 数据处理和可视化工具函数
├── eval.py                     # 模型性能评估和指标计算
├── confusion_m.py              # 混淆矩阵可视化
├── demo.py                     # 模型推理演示脚本
├── video_dataset_anomaly_balance_uni_sample.py  # 数据集加载和处理类
├── utils/                      # 工具脚本
│   ├── train_test_dict_creater.py    # 数据字典创建工具
│   ├── gt_creater.py                 # 真实标签创建器
│   ├── gt_creater_shanghaitech.py    # ShanghaiTech数据集真实标签
│   ├── gt_creater_UCF_Crime.py       # UCF-Crime数据集真实标签
│   ├── Avenue_data_prec.py           # Avenue数据集预处理
│   ├── LV_data_prec.py               # LV数据集预处理
│   └── ped2_data_prec.py             # Ped2数据集预处理
├── *.sh                        # 不同数据集的训练脚本
├── environment.yaml            # Conda环境配置
└── README.md                   # 项目文档
```

## 🔧 代码组件

### 核心模块

#### 模型架构 (`model.py`)
- **基于ConvLSTM的编码器** 用于时空特征提取
- **分类头** 用于异常类别预测
- **回归头** 用于时间异常定位
- **多任务学习** 框架

#### 训练流程 (`train.py`)
- **多示例学习** 使用KMXMILL损失
- **平衡采样** 在正常和异常视频之间
- **梯度累积** 用于稳定训练
- **学习率调度** 使用余弦退火

#### 数据处理 (`video_dataset_anomaly_balance_uni_sample.py`)
- **帧级特征提取** 从预计算的I3D特征
- **时间序列处理** 使用随机采样和填充
- **平衡批次构建** 包含正常和异常样本
- **多数据集支持** (LAD2000, Avenue, Ped2, ShanghaiTech, UCF-Crime)

#### 损失函数 (`losses.py`)
- **KMXMILL损失**: 弱监督场景的多示例学习
- **时间一致性损失**: 确保平滑的时间预测
- **分类损失**: 异常类别预测的交叉熵
- **回归损失**: 时间定位的平滑L1损失

#### 评估框架 (`eval.py`, `confusion_m.py`)
- **帧级AUC**: 时间定位的ROC曲线下面积
- **视频级准确率**: 异常类型的分类准确率
- **误报率**: 假阳性率分析
- **混淆矩阵**: 多类分类可视化

### 主要特性

#### 多模态支持
- **RGB特征**: 基于外观的异常检测
- **Flow特征**: 基于运动的异常检测
- **组合特征**: 外观和运动线索的融合

#### 数据集兼容性
- **LAD2000**: 包含14种异常类别的大规模基准
- **Avenue**: 经典异常检测数据集
- **UCSD Ped2**: 行人异常检测
- **ShanghaiTech**: 复杂校园场景
- **UCF-Crime**: 真实世界监控视频

#### 高级训练技术
- **弱监督学习**: 使用视频级标签进行帧级预测
- **时间建模**: ConvLSTM用于序列理解
- **多尺度处理**: 处理可变长度视频序列
- **数据增强**: 随机时间采样和扰动

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/wanboyang/anomaly_detection_LAD2000.git
cd anomaly_detection_LAD2000

# 创建环境
conda env create -f environment.yaml
conda activate anomaly_icme
```

### 数据准备

1. 从[百度网盘](https://pan.baidu.com/s/1LmNAWnR-RPqo-azCgASvfg)下载 LAD2000 数据集 (密码: avt8)
2. 提取 I3D 特征或使用预提取的特征
3. 在配置中更新数据集路径

### 训练

```bash
# 在 LAD2000 数据集上训练
sh LAD2000T_i3d.sh

# 在其他数据集上训练
sh ped2_i3d.sh      # UCSD Ped2
sh Avenue_i3d.sh    # Avenue
sh shanghaitech_i3d.sh  # ShanghaiTech
sh UCF_i3d.sh       # UCF-Crime
```

### 评估

```bash
python test.py --dataset_name LAD2000 --model_name AED_T --feature_modal combine
```

## 📈 实验结果

| 数据集 | AUC | 帧级 AP | 视频级 AP |
|--------|-----|----------|-----------|
| LAD2000 | 87.2 | 85.6 | 89.1 |
| Avenue | 91.4 | 90.2 | 92.8 |
| Ped2 | 96.8 | 95.3 | 97.5 |
| ShanghaiTech | 84.7 | 82.9 | 86.3 |
| UCF-Crime | 83.5 | 81.2 | 85.1 |

## 📚 引用

如果您发现这项工作对您的研究有用，请引用：

```bibtex
@article{wan2021anomaly,
  title={Anomaly detection in video sequences: A benchmark and computational model},
  author={Wan, Boyang and Jiang, Wenhui and Fang, Yuming and Luo, Zhiyuan and Ding, Guanqun},
  journal={IET Image Processing},
  year={2021},
  publisher={Wiley Online Library}
}
```

## 🤝 致谢

我们感谢 [W-TALC](https://github.com/sujoyp/wtalc-pytorch) 的贡献者和 PyTorch 团队提供的优秀框架。

## 📧 联系方式

如有问题和建议，请联系：
- **万博洋** - wanboyangjerry@163.com
