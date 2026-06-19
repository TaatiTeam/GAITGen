from data.t2m_dataset import Text2MotionDatasetEval, collate_fn # TODO
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt
from utils.log import log
from data.PD_base_action2motion_dataset import BaseClinicalAction2MotionDatasetEval
from argparse import Namespace

def get_dataset_motion_loader(opt, opt_path, batch_size, fname, device):
    opt2 = get_opt(opt_path, device)
    
    opt2.checkpoints_dir = opt.checkpoints_dir
    merged_dict = {**vars(opt), **vars(opt2)}
    opt = Namespace(**merged_dict)
    dataset_names = opt.dataset_name if isinstance(opt.dataset_name, (list, tuple)) else [opt.dataset_name]
    dataset_names = [str(name).lower() for name in dataset_names]

    if 'pdgam' in dataset_names:
        print('Loading dataset pdgam ...')

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        dataset = BaseClinicalAction2MotionDatasetEval(opt, w_vectorizer, fname)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True, pin_memory=True, persistent_workers=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    log('success', 'Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset
