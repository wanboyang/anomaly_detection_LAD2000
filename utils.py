import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import zipfile
import io
import torch

"""
Utility functions for Anomaly Event Detection (AED) / 异常事件检测工具函数
This module contains various utility functions for data processing and visualization.
此模块包含数据处理和可视化的各种工具函数。
"""

def random_extract(feat, t_max):
    """
    Randomly extract a contiguous sequence from features / 从特征中随机提取连续序列
    
    Args:
        feat: Input feature sequence / 输入特征序列
        t_max: Length of sequence to extract / 要提取的序列长度
    
    Returns:
        extracted_feat: Extracted feature sequence / 提取的特征序列
        r: Starting index of extraction / 提取的起始索引
    """
    r = np.random.randint(len(feat)-t_max)
    return feat[r:r+t_max], r

def random_extract_step(feat, t_max, step):
    """
    Randomly extract sequence with step sampling / 使用步长采样随机提取序列
    
    Args:
        feat: Input feature sequence / 输入特征序列
        t_max: Length of sequence to extract / 要提取的序列长度
        step: Sampling step size / 采样步长
    
    Returns:
        extracted_feat: Extracted feature sequence / 提取的特征序列
        r: Starting index of extraction / 提取的起始索引
    """
    if len(feat) - step * t_max > 0:
        r = np.random.randint(len(feat) - step * t_max)
    else:
        r = np.random.randint(step)
    return feat[r:r+t_max:step], r


def random_perturb(feat, length):
    """
    Randomly perturb feature sequence by sampling / 通过采样随机扰动特征序列
    
    Args:
        feat: Input feature sequence / 输入特征序列
        length: Target sequence length / 目标序列长度
    
    Returns:
        perturbed_feat: Perturbed feature sequence / 扰动的特征序列
        samples: Sampling indices / 采样索引
    """
    samples = np.arange(length) * len(feat) / length
    for i in range(length):
        if i < length - 1:
            if int(samples[i]) != int(samples[i + 1]):
                samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
            else:
                samples[i] = int(samples[i])
        else:
            if int(samples[i]) < length - 1:
                samples[i] = np.random.choice(range(int(samples[i]), length))
            else:
                samples[i] = int(samples[i])
    # feat = feat[samples]
    return feat[samples.astype('int')], samples.astype('int')



def pad(feat, min_len):
    """
    Pad feature sequence to minimum length / 将特征序列填充到最小长度
    
    Args:
        feat: Input feature sequence / 输入特征序列
        min_len: Minimum sequence length / 最小序列长度
    
    Returns:
        Padded feature sequence / 填充后的特征序列
    """
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def pad_label(feat, min_len):
    """
    Pad label sequence to minimum length / 将标签序列填充到最小长度
    
    Args:
        feat: Input label sequence / 输入标签序列
        min_len: Minimum sequence length / 最小序列长度
    
    Returns:
        Padded label sequence / 填充后的标签序列
    """
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0])), mode='constant', constant_values=0)
    else:
       return feat


def process_feat(feat, length, step):
    """
    Process features by extracting and padding / 通过提取和填充处理特征
    
    Args:
        feat: Input feature sequence / 输入特征序列
        length: Target sequence length / 目标序列长度
        step: Sampling step size / 采样步长
    
    Returns:
        processed_feat: Processed feature sequence / 处理后的特征序列
        r: Starting index of extraction / 提取的起始索引
    """
    if len(feat) > length:
        if step and step > 1:
            features, r = random_extract_step(feat, length, step)
            return pad(features, length), r
        else:
            features, r = random_extract(feat, length)
            return features, r
    else:
        return pad(feat, length), 0

def process_label(label, r, t_max):
    """
    Process labels based on feature extraction indices / 基于特征提取索引处理标签
    
    Args:
        label: Input label sequence / 输入标签序列
        r: Starting index of feature extraction / 特征提取的起始索引
        t_max: Sequence length / 序列长度
    
    Returns:
        Processed label sequence / 处理后的标签序列
    """
    if label.shape[1] - (16 * (r + t_max)) < 0:
        label = pad_label(label.squeeze(), 16 * (r + t_max))[np.newaxis, :]
        return label[:, 16 * r: 16 * (r + t_max)]
    else:
        return label[:, 16 * r: 16 * (r + t_max)]


def process_feat_sample(feat, length):
    """
    Process features with random perturbation sampling / 使用随机扰动采样处理特征
    
    Args:
        feat: Input feature sequence / 输入特征序列
        length: Target sequence length / 目标序列长度
    
    Returns:
        processed_feat: Processed feature sequence / 处理后的特征序列
        samples: Sampling indices / 采样索引
    """
    if len(feat) > length:
            features, samples = random_perturb(feat, length)
            return features, samples
    else:
        return pad(feat, length), 0


def scorebinary(scores=None, threshold=0.5):
    """
    Convert continuous scores to binary values / 将连续得分转换为二进制值
    
    Args:
        scores: Continuous anomaly scores / 连续异常得分
        threshold: Binary threshold / 二进制阈值
    
    Returns:
        Binary scores (0 or 1) / 二进制得分 (0 或 1)
    """
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold < threshold] = 0
    scores_threshold[scores_threshold >= threshold] = 1
    return scores_threshold



def fill_context_mask(mask, sizes, v_mask, v_unmask):
    """
    Fill attention mask inplace for a variable length context / 为可变长度上下文填充注意力掩码
    
    Args:
        mask: Tensor of size (B, N, D) / 大小为(B, N, D)的张量
            Tensor to fill with mask values / 要填充掩码值的张量
        sizes: list[int] / 整数列表
            List giving the size of the context for each item in the batch / 批次中每个项目的上下文大小列表
            Positions beyond each size will be masked / 超出每个大小的位置将被掩码
        v_mask: float / 浮点数
            Value to use for masked positions / 用于掩码位置的值
        v_unmask: float / 浮点数
            Value to use for unmasked positions / 用于非掩码位置的值
    
    Returns:
        mask: Filled with values in {v_mask, v_unmask} / 填充了{v_mask, v_unmask}值的掩码
    """
    mask.fill_(v_unmask)  # Initialize with unmasked values / 用非掩码值初始化
    n_context = mask.size(2)  # Get context dimension / 获取上下文维度
    for i, size in enumerate(sizes):
        if size < n_context:
            mask[i, :, size:] = v_mask  # Mask positions beyond context size / 掩码超出上下文大小的位置
    return mask


def median(attention_logits, args):
    """
    Apply median-based attention masking / 应用中位数基础的注意力掩码
    
    Args:
        attention_logits: Attention logits / 注意力logits
        args: Command line arguments / 命令行参数
    
    Returns:
        Normalized attention logits after median filtering / 中位数过滤后的归一化注意力logits
    """
    attention_medians = torch.zeros(0).to(args.device)
    # attention_logits_median = torch.zeros(0).to(args.device)
    batch_size = attention_logits.shape[0]
    for i in range(batch_size):
        attention_logit = attention_logits[i][attention_logits[i] > 0].unsqueeze(0)
        attention_medians = torch.cat((attention_medians, attention_logit.median(1, keepdims=True)[0]), dim=0)
    attention_medians = attention_medians.unsqueeze(1)
    attention_logits_mask = attention_logits.clone()
    attention_logits_mask[attention_logits <= attention_medians] = 0
    attention_logits_mask[attention_logits > attention_medians] = 1
    attention_logits = attention_logits * attention_logits_mask
    attention_logits_sum = attention_logits.sum(dim=2, keepdim=True)
    attention_logits = attention_logits / attention_logits_sum
    return attention_logits

#



def anomap(predict_dict, cls_result ,label_dict, save_path, itr, save_root, extend_numbers, args, zip=False):
    """
    Generate anomaly detection visualization plots / 生成异常检测可视化图
    
    Args:
        predict_dict: Dictionary of prediction scores / 预测得分字典
        cls_result: Classification results / 分类结果
        label_dict: Dictionary of ground truth labels / 真实标签字典
        save_path: Path to save plots / 保存图的路径
        itr: Iteration number / 迭代次数
        save_root: Root directory for saving / 保存的根目录
        extend_numbers: Dictionary of extended frame counts / 扩展帧数字典
        args: Command line arguments / 命令行参数
        zip: Whether to save plots to a zip file / 是否将图保存到zip文件
    """
    if os.path.exists(os.path.join(save_root, save_path, 'plot')) == 0:
        os.makedirs(os.path.join(save_root, save_path, 'plot'))

    for k, v in predict_dict.items():
        extend_number = extend_numbers[k]
        clip_number = len(v) // (16 * args.max_seqlen)
        if extend_number[0]:
            ano_index = (clip_number - 1) * (16 * args.max_seqlen)
        else:
            ano_index = clip_number * (16 * args.max_seqlen)
        predict_np = v[:ano_index]
        label_np = label_dict[k][0, :ano_index]
        x = np.arange(len(predict_np))
        plt.plot(x, predict_np, color='b', label='predicted scores', linewidth=1)
        plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xlabel('Frames')
        plt.ylabel('Anomaly scores')
        plt.grid(True, linestyle='-.')
        plt.legend()
        # plt.show()
        k = k + 'max'
        if os.path.exists(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))) == 0:
            os.makedirs(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr)))
            plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))
        else:
            plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))
        plt.close()
