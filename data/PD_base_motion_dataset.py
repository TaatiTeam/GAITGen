import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from os.path import join as pjoin

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils.log import log

def get_pdgam_config(opt):
    dataconfig = getattr(opt, 'pdgam', None)
    if dataconfig is None:
        raise AttributeError("Expected PDGaM dataset config on opt.pdgam")
    return dataconfig

class BaseClinicalMotionDataset(Dataset):
    def __init__(self, opt, split='train'):
        self.opt = opt
        dataconfig = get_pdgam_config(opt)
        self.dataconfig = dataconfig
        self.joints_num = dataconfig['joints_num']
        self.root_dir = dataconfig['root_dir']
        self.motion_dir = dataconfig['motion_dir']
        self.split = split
        splitfile = split + '_tiny' if opt.tiny else split
        self.split_file = pjoin(dataconfig['split_file_dir'], f'{splitfile}.txt')
        self.annot_split_file = pjoin(dataconfig['annot_dir'], f'{split}.csv')
        self.data, self.data_M, self.labels = self.load_data()
        self.mean, self.std = self.get_mean_std(dataconfig['motion_dir'], opt)
        
        
    def get_mean_std(self, motion_dir, opt):     
        mean = np.load(pjoin(motion_dir, 'Mean.npy'))
        std = np.load(pjoin(motion_dir, 'Std.npy'))
        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (self.joints_num - 1) * 3] = std[4: 4 + (self.joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (self.joints_num - 1) * 3: 4 + (self.joints_num - 1) * 9] = std[4 + (self.joints_num - 1) * 3: 4 + (self.joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (self.joints_num - 1) * 9: 4 + (self.joints_num - 1) * 9 + self.joints_num * 3] = std[4 + (self.joints_num - 1) * 9: 4 + (self.joints_num - 1) * 9 + self.joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (self.joints_num - 1) * 9 + self.joints_num * 3:] = std[4 + (self.joints_num - 1) * 9 + self.joints_num * 3:] / opt.feat_bias

            assert 4 + (self.joints_num - 1) * 9 + self.joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)
        else:
            mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
            std = np.load(pjoin(opt.meta_dir, 'std.npy'))
        return mean, std

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]
    
    def __getitem__(self, item):
        # motion = self.data[item]
        # "Z Normalization"
        # motion = (motion - self.mean) / self.std
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
            
        if random.random() > 0.5 and self.split == 'train' and self.opt.augment:
            motion = self.data_M[motion_id][idx:idx + self.opt.window_size]
        else:
            motion = self.data[motion_id][idx:idx + self.opt.window_size]
        
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        
        label = self.labels[motion_id]
        
        return motion, label
    
class PDGaMDataset(BaseClinicalMotionDataset):
    def load_data(self):
        self.rep_dir = pjoin(self.motion_dir, 'new_joint_vecs')
        annotations = pd.read_csv(self.annot_split_file, names=['walk', '#frames', 'score'])
        
        self.lengths = []
        data, labels = [], []
        data_M = []
        id_list = []
        with open(self.split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
                
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.rep_dir, name + '.npy'))
                motion_M = np.load(pjoin(self.rep_dir, name + '_M.npy'))
                if motion.shape[0] < self.opt.window_size: #TODO: Check if this is true for us
                    continue
                self.lengths.append(motion.shape[0] - self.opt.window_size) #WHY??????
                data.append(motion)
                data_M.append(motion_M)
                # read score labels
                if 'mixmatch' in name:
                    score = 3
                else:
                    subject_id = name.split('_')[0][:3]
                    visit_ID = name.split('_')[0][4:]
                    annot_format_ID = f'{visit_ID}_{subject_id}'
                    score = annotations[annotations['walk'] == annot_format_ID]['score'].values[0]
                labels.append(score)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass
            
        self.cumsum = np.cumsum([0] + self.lengths)
        
        log('info', f"Total number of motions {len(data)}, snippets {-1}") 
        return data, data_M, labels
    

class FullSequencePDGaMDataset(BaseClinicalMotionDataset):
    def load_data(self):
        self.rep_dir = pjoin(self.motion_dir, 'new_joint_vecs')
        annotations = pd.read_csv(self.annot_split_file, names=['walk', '#frames', 'score'])
        
        data, data_M, labels = [], [], []
        id_list = []
        with open(self.split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
                
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.rep_dir, name + '.npy'))
                data.append(motion)
                if not 'mixmatch' in name:
                    motion_M = np.load(pjoin(self.rep_dir, name + '_M.npy'))
                    data_M.append(motion_M)
                # read score labels
                if 'mixmatch' in name:
                    score = 3
                else:
                    subject_id = name.split('_')[0][:3]
                    visit_ID = name.split('_')[0][4:]
                    annot_format_ID = f'{visit_ID}_{subject_id}'
                    score = annotations[annotations['walk'] == annot_format_ID]['score'].values[0]
                labels.append(score)
            except Exception as e:
                print(e)
                pass
        
        log('info', f"Total number of motions {len(data)}, snippets {-1}")
        return data, data_M, labels

    def __getitem__(self, idx):
        motion = self.data[idx]
        # "Z Normalization"
        motion = (motion - self.mean) / self.std
        label = self.labels[idx]
        
        m_length = motion.shape[0]
        if m_length < 200:
            motion = np.concatenate([motion,
                                     np.zeros((200 - m_length, motion.shape[1]))
                                     ], axis=0)
        elif m_length > 200:
            motion = motion[:200]
            m_length = 200
            
        return motion, label, m_length
    
    def __len__(self):
        return len(self.data)
    
    
    
class UnifiedMotionDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.combined_dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        return self.combined_dataset[idx]
    
    def inv_transform(self, data):
        # Assuming all datasets have the same mean and std
        # This will call the inv_transform of the first dataset
        return self.datasets[0].inv_transform(data)
    #TODO: make it multi-dataset
    # # Determine which dataset the idx belongs to
    #     dataset_idx = np.searchsorted(self.dataset_cum_lengths, idx, side='right') - 1
    #     dataset_local_idx = idx - self.dataset_cum_lengths[dataset_idx]

    #     # Apply the inv_transform method of the corresponding dataset
    #     return self.datasets[dataset_idx].inv_transform(data)
