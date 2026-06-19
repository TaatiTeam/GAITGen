import os
from argparse import Namespace
import re
import ast
from os.path import join as pjoin
from utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+') 
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, **kwargs):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path, 'r') as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip('\n').split(': ', 1)  # Split on the first ': '
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    try:
                        # Attempt to parse value as a dictionary or list
                        opt_dict[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        # Fallback to storing as a string
                        opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'finest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.db_save_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    dataset_names = opt.dataset_name if isinstance(opt.dataset_name, (list, tuple)) else [opt.dataset_name]
    dataset_names = [str(name).lower() for name in dataset_names]

    if 'pdgam' in dataset_names:
        unified_root = os.environ.get('UNIFIED_DB_ROOT', './data')
        opt.data_root = os.environ.get('PDGAM_ROOT', pjoin(unified_root, 'PDGaM'))
        opt.motion_dir = pjoin(opt.data_root, 'HumanML3Drep_30fps_score3adjusted', 'new_joint_vecs')
        opt.text_dir = ''
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
        opt.num_classes = 4
    else:
        raise KeyError('Dataset not recognized')
    if not hasattr(opt, 'unit_length'):
        opt.unit_length = 4
    opt.dim_word = 300
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    opt_dict.update(kwargs) # Overwrite with kwargs params

    return opt
