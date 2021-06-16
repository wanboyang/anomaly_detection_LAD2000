import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
import options
import os
import pickle
import random
import torch


class dataset(Dataset):
    def __init__(self, args, train=True, trainlist=None, testlist=None):
        """
        :param args:
        self.dataset_path: path to dir contains anomaly datasets
        self.dataset_name: name of dataset which use now
        self.feature_modal: features from different input, contain rgb, flow or combine of above type
        self.feature_pretrain_model: the model name of feature extraction
        self.feature_path: the dir contain all features, use for training and testing
        self.videoname: videonames of dataset
        self.trainlist: videonames of dataset for training
        self.testlist: videonames of dataset for testing
        self.train: boolen type, if it is True, the dataset class return training data
        self.t_max: the max of sampling in training
        """
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.feature_modal = args.feature_modal
        self.feature_pretrain_model = args.feature_pretrain_model
        if self.feature_pretrain_model == 'c3d' or self.feature_pretrain_model == 'c3d_ucf':
            self.feature_layer = args.feature_layer
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_layer, self.feature_modal)
        else:
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_modal)
        self.videoname = os.listdir(self.feature_path)
        # if self.args.larger_mem:
        #     self.data_dict = self.data_dict_creater()
        if trainlist:
            self.trainlist = self.txt2list(trainlist)
            self.testlist = self.txt2list(testlist)
        else:
            self.trainlist = self.txt2list(
                txtpath=os.path.join(self.dataset_path, self.dataset_name, 'train_split.txt'))
            self.testlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'test_split.txt'))
        self.video_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, self.dataset_name, 'GT', 'video_label_IET.pickle'))
        self.frame_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, self.dataset_name, 'GT', 'frame_label_IET.pickle'))
        if self.dataset_name == 'LV':
            self.normal_video_train, self.anomaly_video_train = self.p_n_split_dataset_LV(self.frame_label_dict, self.trainlist)
        else:
            self.normal_video_train, self.anomaly_video_train = self.p_n_split_dataset(self.frame_label_dict, self.trainlist)
        self.train = train
        self.t_max = args.max_seqlen



    def data_dict_creater(self):
        data_dict = {}
        for _i in self.videoname:
            data_dict[_i] = np.load(
                file=os.path.join(self.feature_path, _i.replace('\n', '').replace('Ped', 'ped'), 'feature.npy'))
        return data_dict

    def txt2list(self, txtpath=''):
        """
        use for generating list from text file
        :param txtpath: path of text file
        :return: list of text file
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def pickle_reader(self, file=''):
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)
        return video_label_dict

    def p_n_split_dataset(self, frame_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            if frame_label_dict[t.replace('\n', '')].sum():
                anomaly_video_train.append(t.replace('\n', ''))
            else:
                normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))
        return normal_video_train, anomaly_video_train

    def p_n_split_dataset_LV(self, frame_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            anomaly_video_train.append(t.replace('\n', ''))
            normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))

        return normal_video_train, anomaly_video_train
        # for k, v in video_label_dict.items():
        #     if v[0] == 1.:
        #         anomaly_video_train.append(k)
        #     else:
        #         normal_video_train.append(k)
        # return normal_video_train, anomaly_video_train

    def __getitem__(self, index):

        if self.train:
            anomaly_train_video_name = []
            normaly_train_video_name = []
            anomaly_start_index = []
            anomaly_len_index = []
            normaly_start_index = []
            normaly_len_index = []
            anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size)
            normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)
            anomaly_features = torch.zeros(0)
            normaly_features = torch.zeros(0)
            anomaly_frame_labels = torch.zeros(0, dtype=torch.long)
            normaly_frame_labels = torch.zeros(0, dtype=torch.long)
            anomaly_video_labels = torch.zeros(0, dtype=torch.long)
            normaly_video_labels = torch.zeros(0, dtype=torch.long)

            for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                anomaly_data_video_name = a_i.replace('\n', '').replace('Ped', 'ped')
                normaly_data_video_name = n_i.replace('\n', '').replace('Ped', 'ped')
                anomaly_train_video_name.append(anomaly_data_video_name)
                normaly_train_video_name.append(normaly_data_video_name)
                anomaly_feature = np.load(
                    file=os.path.join(self.feature_path, anomaly_data_video_name, 'feature.npy'))
                anomaly_len_index.append(anomaly_feature.shape[0])
                tmp_num = 0
                while True:
                    anomaly_feature, r = utils.process_feat(anomaly_feature, self.t_max,step=1)
                    anomaly_frame_label = self.frame_label_dict[anomaly_data_video_name]
                    anomaly_frame_label = utils.process_label(anomaly_frame_label, r, self.t_max)
                    anomaly_video_label = np.asarray(self.video_label_dict[anomaly_data_video_name])
                    if anomaly_frame_label.sum():
                        break
                    tmp_num += 1
                    if tmp_num > 10:
                        break
                anomaly_start_index.append(r)
                anomaly_feature = torch.from_numpy(anomaly_feature).unsqueeze(0)
                anomaly_frame_label = torch.from_numpy(anomaly_frame_label).unsqueeze(0).long()
                anomaly_video_label = torch.from_numpy(anomaly_video_label).unsqueeze(0).unsqueeze(0).long()
                normaly_feature = np.load(
                    file=os.path.join(self.feature_path, normaly_data_video_name, 'feature.npy'))
                normaly_len_index.append(normaly_feature.shape[0])
                normaly_feature, r = utils.process_feat(normaly_feature, self.t_max, 1)
                normaly_frame_label = self.frame_label_dict[normaly_data_video_name]
                normaly_frame_label = utils.process_label(normaly_frame_label, r, self.t_max)
                normaly_video_label = np.asarray(self.video_label_dict[normaly_data_video_name])
                normaly_feature = torch.from_numpy(normaly_feature).unsqueeze(0)
                normaly_frame_label = torch.from_numpy(normaly_frame_label).unsqueeze(0).long()
                normaly_video_label = torch.from_numpy(normaly_video_label).unsqueeze(0).unsqueeze(0).long()
                normaly_start_index.append(r)
                anomaly_features = torch.cat((anomaly_features, anomaly_feature), dim=0)  # combine anomaly_feature of different a_i
                normaly_features = torch.cat((normaly_features, normaly_feature), dim=0)  # combine normaly_feature of different n_i
                anomaly_frame_labels = torch.cat((anomaly_frame_labels, anomaly_frame_label), dim=0)
                normaly_frame_labels = torch.cat((normaly_frame_labels, normaly_frame_label), dim=0)
                anomaly_video_labels = torch.cat((anomaly_video_labels, anomaly_video_label), dim=0)
                normaly_video_labels = torch.cat((normaly_video_labels, normaly_video_label), dim=0)
            train_video_name = anomaly_train_video_name + normaly_train_video_name
            start_index = anomaly_start_index + normaly_start_index
            len_index = anomaly_len_index + normaly_len_index

            return [anomaly_features, normaly_features], [anomaly_frame_labels, normaly_frame_labels, anomaly_video_labels, normaly_video_labels], [train_video_name, start_index,len_index]
        else:
            data_video_name = self.testlist[index].replace('\n', '').replace('Ped', 'ped')
            self.feature = np.load(file=os.path.join(self.feature_path, data_video_name, 'feature.npy'))
            return self.feature, data_video_name

    def __len__(self):
        if self.train:
            return len(self.trainlist)
        else:
            return len(self.testlist)


if __name__ == "__main__":
    args = options.parser.parse_args()
    train_dataset = dataset(args=args, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, pin_memory=True,
                              num_workers=5, shuffle=True)
    test_dataset = dataset(args=args, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=5, shuffle=False)
    for epoch in range(1):
        for i, data in enumerate(test_loader):
            features, _ = data
            print(features.shape)
    for epoch in range(2):
        for i, data in enumerate(train_loader):
            [anomaly_features, normaly_features], [anomaly_label, normaly_label] = data
            print(anomaly_features.squeeze(0).shape)
            print(normaly_label.squeeze(0).shape)
