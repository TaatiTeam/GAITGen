import os
import torch
import numpy as np
import wandb

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.transformer import ResidualTransformer
from models.mask_transformer.transformer_disentangled import DResidualTransformer
from models.mask_transformer.transformer_trainer import ResidualTransformerTrainer
from models.mask_transformer.transformer_trainer_disentangled import DResidualTransformerTrainer
from models.vq.model import RVQVAE, Conditional_RVQVAE, Disentangled_RVQVAE

from options.train_option import TrainT2MOptions

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain
from utils.utils import load_yaml
from utils import paramUtil
from data.PD_Unified_dataloader import get_data_loaders

from data.t2m_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper

base_config_path = './data/configs/base.yaml'
base_config = load_yaml(base_config_path)
os.environ['UNIFIED_DB_ROOT'] = base_config['UNIFIED_DB_ROOT']


def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)

def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.db_save_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    if not vq_opt.disentangled:
        vq_opt.w_clsloss = 0
        if vq_opt.conditional:
            vq_model = Conditional_RVQVAE(vq_opt,
                    dim_pose,
                    vq_opt.nb_code,
                    vq_opt.code_dim,
                    vq_opt.output_emb_width,
                    vq_opt.down_t,
                    vq_opt.stride_t,
                    vq_opt.width,
                    vq_opt.depth,
                    vq_opt.dilation_growth_rate,
                    vq_opt.vq_act,
                    vq_opt.vq_norm)
            print(f'Loading Conditional VQ Model {opt.vq_name}')
        else:
            vq_model = RVQVAE(vq_opt,
                        dim_pose,
                        vq_opt.nb_code,
                        vq_opt.code_dim,
                        vq_opt.output_emb_width,
                        vq_opt.down_t,
                        vq_opt.stride_t,
                        vq_opt.width,
                        vq_opt.depth,
                        vq_opt.dilation_growth_rate,
                        vq_opt.vq_act,
                        vq_opt.vq_norm)
            print(f'Loading VQ Model {opt.vq_name}')
    else:
        vq_opt.code_dim_m = vq_opt.code_dim
        vq_opt.nb_code_m = vq_opt.nb_code
        output_emb_width = vq_opt.code_dim_m * 2 if not vq_opt.style and not vq_opt.mdcombine == 'add' else vq_opt.code_dim_m
        vq_model = Disentangled_RVQVAE(vq_opt, vq_opt.dim_pose, 
                                  vq_opt.nb_code_m, vq_opt.nb_code_d,
                                  vq_opt.code_dim_m, vq_opt.code_dim_d,
                                  output_emb_width, 
                                  vq_opt.down_t, vq_opt.down_t_d,
                                  vq_opt.stride_t, vq_opt.width, vq_opt.depth, vq_opt.dilation_growth_rate, vq_opt.vq_act, vq_opt.vq_norm)
        print(f'Loading Disentangled VQ Model {vq_opt.vq_name} Completed!')
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.db_save_name, vq_opt.name, 'model', 'latest.tar'), map_location='cpu', weights_only=True) #ToDo: 'net_best_fid.tar'
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    checkpoint_state_dict = ckpt[model_key]
    model_state_dict = vq_model.state_dict()
    missing_keys = [k for k in model_state_dict if k not in checkpoint_state_dict]
    assert len(missing_keys) == 0, f"Missing keys in checkpoint: {missing_keys}"
    extra_keys = [k for k in checkpoint_state_dict if k not in model_state_dict]
    if extra_keys:
        print(f"Extra keys in checkpoint not used in model: {extra_keys}")
    vq_model.load_state_dict(checkpoint_state_dict, strict=False)
    return vq_model, vq_opt

if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.db_save_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/t2m/', opt.db_save_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    
    assert all(item.lower() in ['pdgam'] for item in opt.dataset_name), 'Dataset Does not Exists'
    dim_pose = -1
    train_split_files, val_split_files = [], []
    for dataset_name in opt.dataset_name:
        dataset_config = load_yaml(f'./data/configs/{dataset_name.lower()}.yaml')
        if dim_pose == -1:
            dim_pose = dataset_config['dim_pose']
        else:
            assert dim_pose == dataset_config['dim_pose'], 'Different Pose Dimensionality'
        setattr(opt, dataset_name.lower(), dataset_config)
        opt.joints_num = dataset_config['joints_num']
        opt.num_classes = dataset_config['num_classes']
        fps = 30
        radius = 4   #TODO: Check this
        opt.max_motion_len = 55
        kinematic_chain = paramUtil.t2m_kinematic_chain
        # dataset_config = getattr(opt, dataset_name.lower(), None)
        dataset_opt_path = f'./checkpoints/{opt.db_save_name}/Comp_v6_KLD005/opt.txt'


    vq_model, vq_opt = load_vq_model()

    clip_version = 'ViT-B/32'

    opt.num_tokens = vq_opt.nb_code
    opt.num_quantizers = vq_opt.num_quantizers

    if not opt.disentangled:
        # if opt.is_v2:
        res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=opt.latent_dim,
                                            ff_size=opt.ff_size,
                                            num_layers=opt.n_layers,
                                            num_heads=opt.n_heads,
                                            dropout=opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                                share_weight=opt.share_weight,
                                            clip_version=clip_version,
                                            opt=opt)
        # else:
        #     res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
        #                                           cond_mode='text',
        #                                           latent_dim=opt.latent_dim,
        #                                           ff_size=opt.ff_size,
        #                                           num_layers=opt.n_layers,
        #                                           num_heads=opt.n_heads,
        #                                           dropout=opt.dropout,
        #                                           clip_dim=512,
        #                                           shared_codebook=vq_opt.shared_codebook,
        #                                           cond_drop_prob=opt.cond_drop_prob,
        #                                           # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
        #                                           clip_version=clip_version,
        #                                           opt=opt)
        trainer = ResidualTransformerTrainer(opt, res_transformer, vq_model)
    else:
        opt.num_motion_tokens = vq_opt.nb_code_m
        opt.num_disease_tokens = vq_opt.nb_code_d
        opt.num_tokens = vq_opt.nb_code_m + vq_opt.nb_code_d
        opt.vq_conditional = vq_opt.conditional
        res_transformer = DResidualTransformer(code_dim_m=vq_opt.code_dim_m,
                                               code_dim_d=vq_opt.code_dim_d,
                                            cond_mode='text',
                                            latent_dim=opt.latent_dim,
                                            ff_size=opt.ff_size,
                                            num_layers=opt.n_layers,
                                            num_heads=opt.n_heads,
                                            dropout=opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                                share_weight=opt.share_weight,
                                            clip_version=clip_version,
                                            opt=opt)
        trainer = DResidualTransformerTrainer(opt, res_transformer, vq_model)


    all_params = 0
    pc_transformer = sum(param.numel() for param in res_transformer.parameters_wo_clip())

    print(res_transformer)
    # print("Total parameters of t2m_transformer net: {:.2f}M".format(pc_transformer / 1000_000))
    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))
    
    train_loader, train_dataset = get_data_loaders(opt, base_config_path, split='train')
    val_loader, val_dataset = get_data_loaders(opt, base_config_path, split='test')
    
    eval_val_loader, _ = get_dataset_motion_loader(opt, dataset_opt_path, 32, 'test', device=opt.device)

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)
    
    
