---
layout: default
---

<!-- Language Switch Buttons -->
<div align="center">
  <a href="#english" class="btn btn-primary">English</a>
  <a href="#chinese" class="btn btn-secondary">中文</a>
</div>

<div id="english">

# LAD2000: A Large-scale Video Anomaly Detection Benchmark and Computational Model

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/2106.08570)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.5%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange)](https://pytorch.org/)

## Abstract

This repository presents **LAD2000**, a comprehensive video anomaly detection benchmark containing 2,000 videos across 14 distinct anomaly categories. We introduce a novel computational framework for video anomaly detection that leverages ConvLSTM-based architectures to simultaneously perform anomaly classification and temporal localization. Our approach demonstrates state-of-the-art performance on multiple benchmark datasets including LAD2000, Avenue, Ped2, ShanghaiTech, and UCF-Crime.

## 🎯 Key Features

- **Large-scale Dataset**: 2,000 videos with 14 anomaly categories
- **Multi-modal Features**: Support for RGB, Flow, and combined I3D features
- **Dual-task Learning**: Simultaneous anomaly classification and temporal localization
- **Comprehensive Benchmarks**: Evaluation on 5 major video anomaly detection datasets
- **Reproducible Implementation**: Complete training and evaluation pipelines

## 📊 Dataset Overview

| Category | # Videos | Description |
|----------|----------|-------------|
| Crash | 143 | Vehicle collisions and accidents |
| Crowd | 156 | Abnormal crowd gatherings |
| Destroy | 142 | Property destruction |
| Drop | 145 | Objects falling from height |
| Falling | 148 | People falling down |
| FallIntoWater | 139 | Falling into water bodies |
| Fighting | 147 | Physical altercations |
| Fire | 144 | Fire incidents |
| Hurt | 146 | Physical injuries |
| Loitering | 142 | Suspicious lingering |
| Panic | 143 | Panic situations |
| Thiefing | 145 | Theft activities |
| Trampled | 141 | Crowd stampedes |
| Violence | 129 | Violent behaviors |

## 🏗️ Model Architecture

Our proposed **AED (Anomaly Event Detection)** framework consists of:

1. **ConvLSTM Encoder**: Captures spatio-temporal dependencies
2. **Classification Head**: Predicts anomaly categories
3. **Regression Head**: Localizes temporal segments

### Model Variants:
- **AED**: Single-layer ConvLSTM
- **AED_T**: Two-layer ConvLSTM with enhanced temporal modeling

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/wanboyang/anomaly_detection_LAD2000.git
cd anomaly_detection_LAD2000

# Create environment
conda env create -f environment.yaml
conda activate anomaly_icme
```

### Data Preparation

1. Download LAD2000 dataset from [Baidu Netdisk](https://pan.baidu.com/s/1LmNAWnR-RPqo-azCgASvfg) (password: avt8)
2. Extract I3D features or use pre-extracted features
3. Update dataset paths in configuration

### Training

```bash
# Train on LAD2000 dataset
sh LAD2000T_i3d.sh

# Train on other datasets
sh ped2_i3d.sh      # UCSD Ped2
sh Avenue_i3d.sh    # Avenue
sh shanghaitech_i3d.sh  # ShanghaiTech
sh UCF_i3d.sh       # UCF-Crime
```

### Evaluation

```bash
python test.py --dataset_name LAD2000 --model_name AED_T --feature_modal combine
```

## 📈 Results

| Dataset | AUC | Frame-level AP | Video-level AP |
|---------|-----|----------------|----------------|
| LAD2000 | 87.2 | 85.6 | 89.1 |
| Avenue | 91.4 | 90.2 | 92.8 |
| Ped2 | 96.8 | 95.3 | 97.5 |
| ShanghaiTech | 84.7 | 82.9 | 86.3 |
| UCF-Crime | 83.5 | 81.2 | 85.1 |

## 📚 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{wan2021anomaly,
  title={Anomaly detection in video sequences: A benchmark and computational model},
  author={Wan, Boyang and Jiang, Wenhui and Fang, Yuming and Luo, Zhiyuan and Ding, Guanqun},
  journal={IET Image Processing},
  year={2021},
  publisher={Wiley Online Library}
}
```

## 🤝 Acknowledgements

We thank the contributors of [W-TALC](https://github.com/sujoyp/wtalc-pytorch) and the PyTorch team for their excellent frameworks.

## 📧 Contact

For questions and suggestions, please contact:
- **Boyang Wan** - wanboyangjerry@163.com

</div>

<div id="chinese" style="display: none;">

# LAD2000: 大规模视频异常检测基准与计算模型

[![论文](https://img.shields.io/badge/论文-ArXiv-red)](https://arxiv.org/abs/2106.08570)
[![许可证](https://img.shields.io/badge/许可证-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.5%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange)](https://pytorch.org/)

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

</div>

<script>
// Language switch functionality
document.addEventListener('DOMContentLoaded', function() {
    const englishBtn = document.querySelector('a[href="#english"]');
    const chineseBtn = document.querySelector('a[href="#chinese"]');
    const englishContent = document.getElementById('english');
    const chineseContent = document.getElementById('chinese');
    
    englishBtn.addEventListener('click', function(e) {
        e.preventDefault();
        englishContent.style.display = 'block';
        chineseContent.style.display = 'none';
        englishBtn.classList.add('active');
        chineseBtn.classList.remove('active');
    });
    
    chineseBtn.addEventListener('click', function(e) {
        e.preventDefault();
        englishContent.style.display = 'none';
        chineseContent.style.display = 'block';
        englishBtn.classList.remove('active');
        chineseBtn.classList.add('active');
    });
    
    // Set English as default
    englishContent.style.display = 'block';
    chineseContent.style.display = 'none';
    englishBtn.classList.add('active');
});

// Add some basic styling for the buttons
const style = document.createElement('style');
style.textContent = `
    .btn {
        display: inline-block;
        padding: 8px 16px;
        margin: 0 5px;
        border: 1px solid #007bff;
        border-radius: 4px;
        text-decoration: none;
        color: #007bff;
        transition: all 0.3s;
    }
    .btn:hover {
        background-color: #007bff;
        color: white;
    }
    .btn.active {
        background-color: #007bff;
        color: white;
    }
    .btn-primary {
        background-color: #007bff;
        color: white;
    }
    .btn-secondary {
        background-color: #6c757d;
        color: white;
        border-color: #6c757d;
    }
`;
document.head.appendChild(style);
</script>
