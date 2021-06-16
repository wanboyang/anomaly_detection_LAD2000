import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import zipfile
import io
import torch


def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max], r

def random_extract_step(feat, t_max, step):
    if len(feat) - step * t_max > 0:
        r = np.random.randint(len(feat) - step * t_max)
    else:
        r = np.random.randint(step)
    return feat[r:r+t_max:step], r


def random_perturb(feat, length):
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
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def pad_label(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0])), mode='constant', constant_values=0)
    else:
       return feat


def process_feat(feat, length, step):
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
    if label.shape[1] - (16 * (r + t_max)) < 0:
        label = pad_label(label.squeeze(), 16 * (r + t_max))[np.newaxis, :]
        return label[:, 16 * r: 16 * (r + t_max)]
    else:
        return label[:, 16 * r: 16 * (r + t_max)]


def process_feat_sample(feat, length):
    if len(feat) > length:
            features, samples = random_perturb(feat, length)
            return features, samples
    else:
        return pad(feat, length), 0


def scorebinary(scores=None, threshold=0.5):
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold < threshold] = 0
    scores_threshold[scores_threshold >= threshold] = 1
    return scores_threshold



def fill_context_mask(mask, sizes, v_mask, v_unmask):
    """Fill attention mask inplace for a variable length context.
    Args
    ----
    mask: Tensor of size (B, N, D)
        Tensor to fill with mask values.
    sizes: list[int]
        List giving the size of the context for each item in
        the batch. Positions beyond each size will be masked.
    v_mask: float
        Value to use for masked positions.
    v_unmask: float
        Value to use for unmasked positions.
    Returns
    -------
    mask:
        Filled with values in {v_mask, v_unmask}
    """
    mask.fill_(v_unmask)
    n_context = mask.size(2)
    for i, size in enumerate(sizes):
        if size < n_context:
            mask[i, :, size:] = v_mask
    return mask


def median(attention_logits, args):
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

    :param predict_dict:
    :param label_dict:
    :param save_path:
    :param itr:
    :param zip: boolen, whether save plots to a zip
    :return:
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


