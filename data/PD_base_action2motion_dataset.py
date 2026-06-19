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

class BaseClinicalAction2MotionDataset(Dataset):
    def __init__(self, opt, split='train'):
        self.opt = opt
        dataconfig = get_pdgam_config(opt)
        self.dataconfig = dataconfig
        self.joints_num = dataconfig['joints_num']
        self.root_dir = dataconfig['root_dir']
        self.motion_dir = dataconfig['motion_dir']
        
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.min_motion_len = 30
        
        splitfile = split + '_tiny' if opt.tiny else split
        self.split_file = pjoin(dataconfig['split_file_dir'], f'{splitfile}.txt')
        self.annot_split_file = pjoin(dataconfig['annot_dir'], f'{split}.csv')
        self.load_data()
        self.mean, self.std = self.get_mean_std(opt)
        
    def get_mean_std(self, opt):     
        mean = np.load(pjoin(opt.checkpoints_dir, opt.db_save_name, opt.vq_name, 'meta', 'mean.npy'))
        std = np.load(pjoin(opt.checkpoints_dir, opt.db_save_name, opt.vq_name, 'meta', 'std.npy'))
        return mean, std

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_data, label = data['motion'], data['length'], data['text'], data['label']
        caption, tokens = text_data['caption'], text_data['tokens']
        
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length, label
    
    
class PDGaMDataset(BaseClinicalAction2MotionDataset):
    def load_data(self):
        label_to_text = {
                        0: {
                            'caption': "zero",
                            'tokens': ['zero/NUM']
                        },
                        1: {
                            'caption': "one",
                            'tokens': ['one/NUM']
                        },
                        2: {
                            'caption': "two",
                            'tokens': ['two/NUM']
                        },
                        3: {
                            'caption': "three",
                            'tokens': ['three/NUM']
                        }
                    }
        
        self.rep_dir = pjoin(self.motion_dir, 'new_joint_vecs')
        annotations = pd.read_csv(self.annot_split_file, names=['walk', '#frames', 'score'])
        
        self.lengths = []
        data, labels = [], []
        id_list = []
        with open(self.split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
                
        data_dict = {}
        length_list, name_list = [], []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.rep_dir, name + '.npy'))
                if (len(motion)) < self.min_motion_len:
                    continue
                if (len(motion) >= self.max_motion_length):   #TODO: Change this
                    half_len = int(self.max_motion_length / 2)
                    motion = motion[len(motion) // 2 - half_len:len(motion) // 2 + half_len, ...]
                
                data.append(motion)
                # read score labels
                if 'mixmatch' in name:
                    score = 3
                else:
                    subject_id = name.split('_')[0][:3]
                    visit_ID = name.split('_')[0][4:]
                    annot_format_ID = f'{visit_ID}_{subject_id}'
                    score = annotations[annotations['walk'] == annot_format_ID]['score'].values[0]
                labels.append(score)
                length_list.append(len(motion))
                
                text_data = label_to_text.get(score, {'caption': '', 'tokens': []})
                if text_data['caption'] == '':
                    raise Exception("No caption found")

                data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'label': score,
                                       'text': text_data}
                name_list.append(name)
                
            except Exception as e:
                print(e)
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict  
        self.name_list = name_list   
        
    
class UnifiedAction2MotionDataset(Dataset):
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
    
    
class BaseClinicalAction2MotionDatasetEval(Dataset):
    def __init__(self, opt, w_vectorizer, split='train'):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        dataconfig = get_pdgam_config(opt)
        self.dataconfig = dataconfig
        self.joints_num = dataconfig['joints_num']
        self.root_dir = dataconfig['root_dir']
        self.motion_dir = dataconfig['motion_dir']
        
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.min_motion_len = 30
        
        splitfile = split + '_tiny' if opt.tiny else split
        self.split_file = pjoin(dataconfig['split_file_dir'], f'{splitfile}.txt')
        self.annot_split_file = pjoin(dataconfig['annot_dir'], f'{split}.csv')
        self.load_data()
        self.mean, self.std = self.get_mean_std(opt)
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length
        
    def get_mean_std(self, opt):     
        mean = np.load(pjoin(opt.checkpoints_dir, opt.db_save_name, opt.vq_name, 'meta', 'mean.npy'))
        std = np.load(pjoin(opt.checkpoints_dir, opt.db_save_name, opt.vq_name, 'meta', 'std.npy'))
        return mean, std

    # def load_data(self):
    #     raise NotImplementedError("Subclasses should implement this method")
    def load_data(self):
        label_to_text = {
                        1: {
                            'caption': "one",
                            'tokens': ['one/NUM']
                        },
                        2: {
                            'caption': "two",
                            'tokens': ['two/NUM']
                        },
                        3: {
                            'caption': "three",
                            'tokens': ['three/NUM']
                        },
                        0: {
                            'caption': "zero",
                            'tokens': ['zero/NUM']
                        }
                    }
        
        self.rep_dir = pjoin(self.motion_dir, 'new_joint_vecs')
        annotations = pd.read_csv(self.annot_split_file, names=['walk', '#frames', 'score'])
        
        self.lengths = []
        data, labels = [], []
        id_list = []
        with open(self.split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
                
        data_dict = {}
        length_list, name_list = [], []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.rep_dir, name + '.npy'))
                if (len(motion)) < self.min_motion_len:
                    continue
                if (len(motion) >= self.max_motion_length):   #TODO: Change this
                    half_len = int(self.max_motion_length / 2)
                    motion = motion[len(motion) // 2 - half_len:len(motion) // 2 + half_len, ...]
                
                data.append(motion)
                # read score labels
                if 'mixmatch' in name:
                    score = 3
                else:
                    subject_id = name.split('_')[0][:3]
                    visit_ID = name.split('_')[0][4:]
                    annot_format_ID = f'{visit_ID}_{subject_id}'
                    score = annotations[annotations['walk'] == annot_format_ID]['score'].values[0]
                labels.append(score)
                length_list.append(len(motion))
                
                text_data = label_to_text.get(score, {'caption': '', 'tokens': []})
                if text_data['caption'] == '':
                    raise Exception("No caption found")

                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'label': score,
                                    'text': text_data}
                name_list.append(name)
                
            except Exception as e:
                text_dict = {}
                text_dict['caption'] = ''
                text_dict['tokens'] = ''
                text_data.append(text_dict)
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data}
                name_list.append(name)
                length_list.append(len(motion))
                
        name_list, length_list = zip(*sorted(zip(name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict  
        self.name_list = name_list  

        
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_data, label = data['motion'], data['length'], data['text'], data['label']
        caption, tokens = text_data['caption'], text_data['tokens']
        
        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), label
    
class PDGaMDatasetEval(BaseClinicalAction2MotionDatasetEval):
    def load_data(self):
        label_to_text = {
                        0: {
                            'caption': "zero",
                            'tokens': ['zero/NUM']
                        },
                        1: {
                            'caption': "one",
                            'tokens': ['one/NUM']
                        },
                        2: {
                            'caption': "two",
                            'tokens': ['two/NUM']
                        },
                        3: {
                            'caption': "three",
                            'tokens': ['three/NUM']
                        }
                    }
        
        self.rep_dir = pjoin(self.motion_dir, 'new_joint_vecs')
        annotations = pd.read_csv(self.annot_split_file, names=['walk', '#frames', 'score'])
        
        self.lengths = []
        data, labels = [], []
        id_list = []
        with open(self.split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
                
        data_dict = {}
        length_list, name_list = [], []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.rep_dir, name + '.npy'))
                if (len(motion)) < self.min_motion_len:
                    continue
                if (len(motion) >= self.max_motion_length):   #TODO: Change this
                    half_len = int(self.max_motion_length / 2)
                    motion = motion[len(motion) // 2 - half_len:len(motion) // 2 + half_len, ...]
                
                data.append(motion)
                # read score labels
                if 'mixmatch' in name:
                    score = 3
                else:
                    subject_id = name.split('_')[0][:3]
                    visit_ID = name.split('_')[0][4:]
                    annot_format_ID = f'{visit_ID}_{subject_id}'
                    score = annotations[annotations['walk'] == annot_format_ID]['score'].values[0]
                labels.append(score)
                length_list.append(len(motion))
                
                text_data = label_to_text.get(score, {'caption': '', 'tokens': []})

                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'label': score,
                                    'text': text_data}
                name_list.append(name)
                
            except Exception as e:
                text_dict = {}
                text_dict['caption'] = ''
                text_dict['tokens'] = ''
                text_data.append(text_dict)
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data}
                name_list.append(name)
                length_list.append(len(motion))
                
        name_list, length_list = zip(*sorted(zip(name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict  
        self.name_list = name_list  

        
