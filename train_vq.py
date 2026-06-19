import os
from os.path import join as pjoin
import wandb

import torch
from data.PD_Unified_dataloader import get_data_loaders
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

from models.vq.model import RVQVAE, Conditional_RVQVAE, Disentangled_RVQVAE
from models.vq.vq_trainer import RVQTokenizerTrainer
from models.vq.vq_trainer_disentangled import DisentangledRVQTokenizerTrainer
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from options.vq_option import arg_parse, save_opt

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion, plot_3d_motion_overlaid2
from utils.fixseed import fixseed
from utils.utils import load_yaml
from utils.get_opt import get_opt
from utils import paramUtil

os.environ["OMP_NUM_THREADS"] = "1"

base_config_path = './data/configs/base.yaml'
base_config = load_yaml(base_config_path)
os.environ['UNIFIED_DB_ROOT'] = base_config['UNIFIED_DB_ROOT']

def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        if isinstance(joint, torch.Tensor):
            joint = joint.cpu().numpy()
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)
        
def plot_t2m_codebooklayers(all_layers_data, save_dir, sample_num=4):
    orig_data = all_layers_data[0][:sample_num]
    orig_data = train_dataset.inv_transform(orig_data)
    num_layers = len(all_layers_data)
    # Plot original data
    for i in range(len(orig_data)):
        joint_data = orig_data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, 'orig_%02d.mp4' % i)
        if isinstance(joint, torch.Tensor):
            joint = joint.cpu().numpy()
        plot_3d_motion(save_path, kinematic_chain, joint, title="Original", fps=fps, radius=radius)
        
    # Plot each layer's prediction
    for sample_idx in range(sample_num, len(all_layers_data[0])):
        for layer_idx in range(0, num_layers):
            layer_data = all_layers_data[layer_idx][sample_idx]
            layer6_data = all_layers_data[len(all_layers_data)-1][sample_idx] # len(all_layers_data)=6 which contains the quantization of all layers
            layer_data = train_dataset.inv_transform(layer_data)
            layer6_data = train_dataset.inv_transform(layer6_data)
            
            joint_layer = recover_from_ric(torch.from_numpy(layer_data).float(), opt.joints_num).numpy()
            joint_layer6 = recover_from_ric(torch.from_numpy(layer6_data).float(), opt.joints_num).numpy()
        
            save_path = pjoin(save_dir, 'pred_%02d_L%1dL%1d.mp4' % (sample_idx-4, 6, layer_idx))
            plot_3d_motion_overlaid2(save_path, kinematic_chain, joint_layer6, joint_layer, title=f"BLUE: all layers - RED: layer{layer_idx}", fps=fps, radius=radius)

def plot_t2m_GTPRED(data, save_dir, numsample=4):
    orig_data = data[:numsample]
    rec_data = data[numsample:]
    orig_data = train_dataset.inv_transform(orig_data)
    rec_data = train_dataset.inv_transform(rec_data)

    for i in range(len(orig_data)):
        joint_data = orig_data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        joint_data_rec = rec_data[i]
        joint_rec = recover_from_ric(torch.from_numpy(joint_data_rec).float(), opt.joints_num).numpy()
        
        save_path = pjoin(save_dir, '%02d.mp4' % i)
        if isinstance(joint, torch.Tensor):
            joint = joint.cpu().numpy()
        plot_3d_motion_overlaid2(save_path, kinematic_chain, joint_rec, joint, title=f"BLUE: Reconst - RED: GT", fps=fps, radius=radius) 

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)
    fixseed(opt.seed)

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        opt.device = torch.device("mps")
    elif torch.cuda.is_available() and opt.gpu_id != -1:
        opt.device = torch.device(f"cuda:{opt.gpu_id}")
    else:
        opt.device = torch.device("cpu")
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.db_save_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/vq/', opt.db_save_name, opt.name)
    opt.outdir = pjoin(opt.save_root, 'outputs')
    
    if os.path.exists( pjoin(opt.model_dir, 'latest.tar')):
        opt.is_continue = True

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.outdir, exist_ok=True)
    
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
        opt.dim_pose = dim_pose
        opt.beta = 0.25
        fps = 30
        radius = 4
        kinematic_chain = paramUtil.t2m_kinematic_chain
        # dataset_config = getattr(opt, dataset_name.lower(), None)
        dataset_opt_path = f'./checkpoints/{opt.db_save_name}/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device(opt.device))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    if opt.is_continue:
        with open(pjoin(opt.save_root, 'wandbID.txt'), 'r') as file:
            opt.wandb_runid = file.read()
        wandb.init(project="VQ_PD", name=opt.name, config=vars(opt), id=opt.wandb_runid, resume="allow")
    else:
        wandb.init(project="VQ_PD", name=opt.name, config=vars(opt))
        with open(pjoin(opt.save_root, 'wandbID.txt'), 'w') as file:
            file.write(wandb.run.id)
    print(f"***********************************************")
    print(f"WandB Run ID: {wandb.run.id}")
    print(f"***********************************************")
    opt.vq_name = opt.name
    
    if not opt.disentangled:
        inp = (opt,
                opt.dim_pose,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm)
        if not opt.conditional:
            net = RVQVAE(*inp)
            print("-----> RVQVAE Model used")
        else:
            net = Conditional_RVQVAE(*inp)
            print("-----> Conditional RVQVAE Model used")
        trainer = RVQTokenizerTrainer(opt, vq_model=net)
    else:
        opt.code_dim_m = opt.code_dim
        opt.nb_code_m = opt.nb_code
        output_emb_width = opt.code_dim_m * 2 if not opt.style and not opt.mdcombine == 'add' else opt.code_dim_m
        net = Disentangled_RVQVAE(opt, opt.dim_pose, 
                                  opt.nb_code_m, opt.nb_code_d,
                                  opt.code_dim_m, opt.code_dim_d,
                                  output_emb_width, 
                                  opt.down_t, opt.down_t_d,
                                  opt.stride_t, opt.width, opt.depth, opt.dilation_growth_rate, opt.vq_act, opt.vq_norm)
        if opt.conditional:
            print("-----> Disentangled Conditional RVQVAE Model used")
        else:
            print("-----> Disentangled RVQVAE Model used")
        trainer = DisentangledRVQTokenizerTrainer(opt, vq_model=net)
            
    save_opt(opt)
    
    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)
    # print("Total parameters of discriminator net: {}".format(pc_vq))
    # all_params += pc_vq_dis

    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))
    
    train_loader, train_dataset = get_data_loaders(opt, base_config_path, split='train')
    val_loader, val_dataset = get_data_loaders(opt, base_config_path, split='test', drop_last=False)
    
    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Val Dataset Length: {len(val_dataset)}")


    eval_val_loader, _ = get_dataset_motion_loader(opt, dataset_opt_path, 32, 'test', device=opt.device)
    if opt.pretrain_motion_encoder and not opt.is_continue:
        trainer.pretrain_motion_encoder(train_loader, val_loader)
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, plot_t2m_GTPRED)
    wandb.finish()

    
