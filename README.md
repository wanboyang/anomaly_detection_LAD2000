# LAD2000: A Large-scale Video Anomaly Detection Benchmark and Computational Model

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/2106.08570)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.5%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange)](https://pytorch.org/)

> **Language**: [English](README.md) | [中文](README_CN.md)

---

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

## 📁 Project Structure

```
anomaly_detection_LAD2000/
├── model.py                    # ConvLSTM model architecture and classification/regression heads
├── options.py                  # Command line argument parser
├── main.py                     # Main entry point and training environment setup
├── train.py                    # Main training loop and loss computation
├── test.py                     # Model testing and prediction functions
├── losses.py                   # Custom loss functions collection
├── utils.py                    # Data processing and visualization utilities
├── eval.py                     # Model performance evaluation and metrics calculation
├── confusion_m.py              # Confusion matrix visualization
├── demo.py                     # Demo script for model inference
├── video_dataset_anomaly_balance_uni_sample.py  # Dataset loading and processing class
├── utils/                      # Utility scripts
│   ├── train_test_dict_creater.py    # Data dictionary creation tools
│   ├── gt_creater.py                 # Ground truth label creator
│   ├── gt_creater_shanghaitech.py    # ShanghaiTech dataset ground truth
│   ├── gt_creater_UCF_Crime.py       # UCF-Crime dataset ground truth
│   ├── Avenue_data_prec.py           # Avenue dataset preprocessing
│   ├── LV_data_prec.py               # LV dataset preprocessing
│   └── ped2_data_prec.py             # Ped2 dataset preprocessing
├── *.sh                        # Training scripts for different datasets
├── environment.yaml            # Conda environment configuration
└── README.md                   # Project documentation
```

## 🔧 Code Components

### Core Modules

#### Model Architecture (`model.py`)
- **ConvLSTM-based encoder** for spatio-temporal feature extraction
- **Classification head** for anomaly category prediction
- **Regression head** for temporal anomaly localization
- **Multi-task learning** framework

#### Training Pipeline (`train.py`)
- **Multi-instance learning** with KMXMILL loss
- **Balanced sampling** between normal and abnormal videos
- **Gradient accumulation** for stable training
- **Learning rate scheduling** with cosine annealing

#### Data Processing (`video_dataset_anomaly_balance_uni_sample.py`)
- **Frame-level feature extraction** from pre-computed I3D features
- **Temporal sequence processing** with random sampling and padding
- **Balanced batch construction** with normal and abnormal samples
- **Multi-dataset support** (LAD2000, Avenue, Ped2, ShanghaiTech, UCF-Crime)

#### Loss Functions (`losses.py`)
- **KMXMILL Loss**: Multiple instance learning for weakly supervised scenarios
- **Temporal Consistency Loss**: Ensures smooth temporal predictions
- **Classification Loss**: Cross-entropy for anomaly category prediction
- **Regression Loss**: Smooth L1 loss for temporal localization

#### Evaluation Framework (`eval.py`, `confusion_m.py`)
- **Frame-level AUC**: Area under ROC curve for temporal localization
- **Video-level Accuracy**: Classification accuracy for anomaly types
- **False Alarm Rate**: False positive rate analysis
- **Confusion Matrix**: Multi-class classification visualization

### Key Features

#### Multi-Modal Support
- **RGB features**: Appearance-based anomaly detection
- **Flow features**: Motion-based anomaly detection
- **Combined features**: Fusion of appearance and motion cues

#### Dataset Compatibility
- **LAD2000**: Large-scale benchmark with 14 anomaly categories
- **Avenue**: Classical anomaly detection dataset
- **UCSD Ped2**: Pedestrian anomaly detection
- **ShanghaiTech**: Complex campus scenes
- **UCF-Crime**: Real-world surveillance videos

#### Advanced Training Techniques
- **Weakly Supervised Learning**: Video-level labels for frame-level prediction
- **Temporal Modeling**: ConvLSTM for sequence understanding
- **Multi-scale Processing**: Handling variable-length video sequences
- **Data Augmentation**: Random temporal sampling and perturbation

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
