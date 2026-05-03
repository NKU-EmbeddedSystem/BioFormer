import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data_provider.uea import (
    normalize_batch_ts,
)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from natsort import natsorted


def subject_wise_zscore(X, subject_ids, eps=1e-6):
    """
    X: np.ndarray, shape (N, T, C)
    subject_ids: np.ndarray, shape (N,)
    """
    X_norm = X.copy()

    for sid in np.unique(subject_ids):
        idx = subject_ids == sid
        X_s = X[idx]        # (Ns, T, C)
        mean = X_s.mean(axis=(0,1), keepdims=True)
        std = X_s.std(axis=(0,1), keepdims=True) + eps

        X_norm[idx] = (X_s - mean) / std

    return X_norm

class sub_Single_label_DependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        # load data in subject-dependent manner
        self.X, self.y = self.load_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]
        args.is_cross_subject == False

    def load_dependent(self, data_path, label_path, flag=None):
        '''
        Loads data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        # subject_label.shape = (num_subjects, 2)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        # print(filenames)
        for j in range(len(filenames)):
            trial_label = subject_label[j]
            # trial_label.shape = (2,)
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            # subject_feature.shape = (num_trials, seq_len, feat_dim)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)

        # 60 : 20 : 20
        X_all, y_all = np.array(feature_list), np.array(label_list)
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

        if flag == 'TRAIN':
            return X_train, y_train[:, 0]
        elif flag == 'VAL':
            return X_val, y_val[:, 0]
        elif flag == 'TEST':
            return X_test, y_test[:, 0]
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)

class sub_Multi_label_DependentLoader(Dataset):
    def __init__(self, args, root_path, flag='TRAIN', label_map={'A': 'B'}):
        self.root_path = root_path
        self.flag = flag
        self.label_map = label_map or {}      
        self.X, self.y = self.load_dependent()
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]
        args.is_cross_subject == False

    def load_dependent(self):
        X_tr_list, X_val_list, X_te_list = [], [], []
        y_tr_list, y_val_list, y_te_list = [], [], []

        data_files = natsorted([f for f in os.listdir(self.root_path)
                                if f.endswith('_data.npy')])
        for df in data_files:
            prefix = df.split('_data.npy')[0]
            label_prefix = self.label_map.get(prefix, prefix)
            lf = f'{label_prefix}_label.npy'

            X_sub = np.load(os.path.join(self.root_path, df))   # (trials, L, D)
            y_sub = np.load(os.path.join(self.root_path, lf))   # (trials,)

            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub)
            X_val, X_te, y_val, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)

            X_tr_list.append(X_tr);  y_tr_list.append(y_tr)
            X_val_list.append(X_val); y_val_list.append(y_val)
            X_te_list.append(X_te);   y_te_list.append(y_te)

        X_tr = np.concatenate(X_tr_list, axis=0)
        X_val = np.concatenate(X_val_list, axis=0)
        X_te = np.concatenate(X_te_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0).squeeze()
        y_val = np.concatenate(y_val_list, axis=0).squeeze()
        y_te = np.concatenate(y_te_list, axis=0).squeeze()
        if self.flag == 'TRAIN':
            return X_tr, y_tr
        elif self.flag == 'VAL':
            return X_val, y_val
        elif self.flag == 'TEST':
            return X_te, y_te
        else:
            raise ValueError('flag must be TRAIN/VAL/TEST')

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].astype(np.float32)), \
               torch.from_numpy(np.array(self.y[idx], dtype=int))

    def __len__(self):
        return len(self.y)
    
class APAVALoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)

        all_ids = list(data_list[:, 1])  # id of all samples
        val_ids = [15, 16, 19, 20]  # 15, 19 are AD; 16, 20 are HC
        test_ids = [1, 2, 17, 18]  # 1, 17 are AD; 2, 18 are HC
        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]
        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y, self.id = self.load_apava(self.data_path, self.label_path, flag=flag)
        mapping = {v: i for i, v in enumerate(sorted(set(self.id)))}
        self.id[:] = [mapping[v] for v in self.id]
        # pre_process
        if args.method == "SubjNorm":
            self.X = subject_wise_zscore(self.X, self.id)
            print("Applied subject-wise z-score normalization.")
        else:
            self.X = normalize_batch_ts(self.X)
        
        self.max_seq_len = self.X.shape[1]
    

    def load_apava(self, data_path, label_path, flag=None):
        """
        Loads APAVA data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        # subject_label.shape = (num_subjects, 2)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("train subject number :", len(ids))
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("validation subject number :", len(ids))
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = set(self.test_ids)
            # print("test ids:", ids)
            print("test subject number :", len(ids))
        else:
            ids = set(subject_label[:, 1])
            print("all subject number :", len(ids))
            # print("all ids:", ids)
        
        for j, fname in enumerate(filenames):
            if j + 1 not in ids:
                continue 
            trial_label = subject_label[j]
            subject_feature = np.load(data_path + fname)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)
        
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y,random_state=42)

        return X, y[:, 0], y[:,1]   # only use the first column (label)
    # return one sample of the dataset
    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.array(self.y[index])),torch.from_numpy(np.array(self.id[index]))

    # return the length of the dataset
    def __len__(self):
        return len(self.y)

class ADFTDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y, self.id = self.load_adfd(self.data_path, self.label_path, flag=flag)

        mapping = {v: i for i, v in enumerate(sorted(set(self.id)))}
        self.id[:] = [mapping[v] for v in self.id]  

        # pre_process
        if args.method == "SubjNorm":
            self.X = subject_wise_zscore(self.X, self.id)
            print("Applied subject-wise z-score normalization.")
        else:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ftd_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Frontotemporal Dementia IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ftd_list[: int(a * len(ftd_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ftd_list[int(a * len(ftd_list)) : int(b * len(ftd_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ftd_list[int(b * len(ftd_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_adfd(self, data_path, label_path, flag=None):
        """
        Loads adfd or cnbpm data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("train subject number :", len(ids))
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("validation subject number :", len(ids))
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("test subject number :", len(ids))
        else:
            ids = set(subject_label[:, 1])
            print("all subject number :", len(ids))

        for j, fname in enumerate(filenames):
            if j + 1 not in ids:
                continue  
            trial_label = subject_label[j]
            subject_feature = np.load(data_path + fname)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)

        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0], y[:, 1]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        ), torch.from_numpy(
            np.asarray(self.id[index])
        )

    def __len__(self):
        return len(self.y)

class PTBLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y, self.id = self.load_ptb(self.data_path, self.label_path, flag=flag)

        
        # pre_process
        if args.method == "SubjNorm":
            self.X = subject_wise_zscore(self.X, self.id)
            print("Applied subject-wise z-score normalization.")
        else:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        my_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # Myocardial infarction IDs

        train_ids = hc_list[: int(a * len(hc_list))] + my_list[: int(a * len(my_list))]
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + my_list[int(a * len(my_list)) : int(b * len(my_list))]
        )
        test_ids = hc_list[int(b * len(hc_list)) :] + my_list[int(b * len(my_list)) :]

        return train_ids, val_ids, test_ids

    def load_ptb(self, data_path, label_path, flag=None):
        """
        Loads ptb data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("train subject number :", len(ids))
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("validation subject number :", len(ids))
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("test subject number :", len(ids))
        else:
            ids = set(subject_label[:, 1])
            print("all subject number :", len(ids))

        for j, fname in enumerate(filenames):
            if j + 1 not in ids:
                continue
            trial_label = subject_label[j]
            subject_feature = np.load(data_path + fname)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)

        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0], y[:,1]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.asarray(self.y[index])), torch.from_numpy(np.asarray(self.id[index]))

    def __len__(self):
        return len(self.y)

class PTBXLLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y, self.id = self.load_ptbxl(self.data_path, self.label_path, flag=flag)

        mapping = {v: i for i, v in enumerate(sorted(set(self.id)))}

        self.id[:] = [mapping[v] for v in self.id]  
        
         # pre_process
        if args.method == "SubjNorm":
            self.X = subject_wise_zscore(self.X, self.id)
            print("Applied subject-wise z-score normalization.")
        else:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        no_list = list(
            data_list[np.where(data_list[:, 0] == 0)][:, 1]
        )  # Normal ECG IDs
        mi_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial Infarction IDs
        sttc_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # ST/T Change IDs
        cd_list = list(
            data_list[np.where(data_list[:, 0] == 3)][:, 1]
        )  # Conduction Disturbance IDs
        hyp_list = list(
            data_list[np.where(data_list[:, 0] == 4)][:, 1]
        )  # Hypertrophy IDs

        train_ids = (
            no_list[: int(a * len(no_list))]
            + mi_list[: int(a * len(mi_list))]
            + sttc_list[: int(a * len(sttc_list))]
            + cd_list[: int(a * len(cd_list))]
            + hyp_list[: int(a * len(hyp_list))]
        )
        val_ids = (
            no_list[int(a * len(no_list)) : int(b * len(no_list))]
            + mi_list[int(a * len(mi_list)) : int(b * len(mi_list))]
            + sttc_list[int(a * len(sttc_list)) : int(b * len(sttc_list))]
            + cd_list[int(a * len(cd_list)) : int(b * len(cd_list))]
            + hyp_list[int(a * len(hyp_list)) : int(b * len(hyp_list))]
        )
        test_ids = (
            no_list[int(b * len(no_list)) :]
            + mi_list[int(b * len(mi_list)) :]
            + sttc_list[int(b * len(sttc_list)) :]
            + cd_list[int(b * len(cd_list)) :]
            + hyp_list[int(b * len(hyp_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_ptbxl(self, data_path, label_path, flag=None):
        """
        Loads ptb-xl data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = set(self.train_ids)    # list of training IDs
            print("train subject number :", len(ids))
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("validation subject number :", len(ids))
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("test subject number :", len(ids))
        else:
            ids = set(subject_label[:, 1])
            print("all subject number :", len(ids))

        for j, fname in enumerate(filenames):
            if j + 1 not in ids:
                continue 
            trial_label = subject_label[j]
            subject_feature = np.load(data_path + fname)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)
        # reshape and shuffle 
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)
        # print("x.shape:",X.shape,"y.shape", y.shape)
        return X, y[:, 0], y[:,1]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        ), torch.from_numpy(
            np.asarray(self.id[index])
        )
    

    def __len__(self):
        return len(self.y)

class BCI2aLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path

        all_ids = list(range(1, 10))  # id of all samples, from 1 to 9
        val_ids = [6, 8]  
        test_ids = [7, 9]  
        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]
        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y, self.id = self.load_bci2a(self.root_path, flag=flag)

        mapping = {v: i for i, v in enumerate(sorted(set(self.id)))}

        self.id[:] = [mapping[v] for v in self.id]
        # pre_process
        if args.method == "SubjNorm":
            self.X = subject_wise_zscore(self.X, self.id)
            print("Applied subject-wise z-score normalization.")
        else:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_bci2a(self, data_path, flag=None):
        feature_list = []
        label_list = []
        subject_list = []

        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("train subject number :", len(ids))
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("validation subject number :", len(ids))
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("test subject number :", len(ids))
        else:
            ids = set(self.train_ids + self.val_ids + self.test_ids)
            print("all subject number :", len(ids))
        
        for j in ids:
            j_data_path = data_path + f"/A0{j}_data.npy"
            j_label_path = data_path + f"/A0{j}_label.npy"
            subject_feature = np.load(j_data_path)
            subject_label = np.load(j_label_path)
            for trial_feature, trial_label in zip(subject_feature, subject_label):
                feature_list.append(trial_feature)
                label_list.append(trial_label)
                subject_list.append(j)
        label_array = np.asarray(label_list, dtype=int).squeeze()
        subject_array = np.asarray(subject_list, dtype=int)
        label_subject = np.stack([label_array, subject_array], axis=1)   # shape (N, 2)
        X = np.array(feature_list)
        y = np.array(label_subject)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:,0], y[:,1]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.asarray(self.y[index])),torch.from_numpy(np.asarray(self.id[index]))

    def __len__(self):
        return len(self.y)
    
class BCI2bLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path

        all_ids = list(range(1, 10))  # id of all samples, from 1 to 9
        val_ids = [6, 8]  
        test_ids = [7, 9]  
        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]
        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y, self.id = self.load_bci2b(self.root_path, flag=flag)

        mapping = {v: i for i, v in enumerate(sorted(set(self.id)))}

        self.id[:] = [mapping[v] for v in self.id]
        # pre_process
        if args.method == "SubjNorm":
            self.X = subject_wise_zscore(self.X, self.id)
            print("Applied subject-wise z-score normalization.")
        else:
            self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_bci2b(self, data_path, flag=None):
        feature_list = []
        label_list = []
        subject_list = []

        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("train subject number :", len(ids))
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("validation subject number :", len(ids))
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("test subject number :", len(ids))
        else:
            ids = set(self.train_ids + self.val_ids + self.test_ids)
            print("all subject number :", len(ids))
        
        for j in ids:
            j_data_path = data_path + f"/B0{j}_data.npy"
            j_label_path = data_path + f"/B0{j}_label.npy"
            subject_feature = np.load(j_data_path)
            subject_label = np.load(j_label_path)
            for trial_feature, trial_label in zip(subject_feature, subject_label):
                feature_list.append(trial_feature)
                label_list.append(trial_label)
                subject_list.append(j)
        label_array = np.asarray(label_list, dtype=int).squeeze()
        subject_array = np.asarray(subject_list, dtype=int)
        label_subject = np.stack([label_array, subject_array], axis=1)   # shape (N, 2)
        X = np.array(feature_list)
        y = np.array(label_subject)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:,0], y[:,1]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.asarray(self.y[index])),torch.from_numpy(np.asarray(self.id[index]))

    def __len__(self):
        return len(self.y)
    