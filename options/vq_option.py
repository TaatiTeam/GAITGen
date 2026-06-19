import argparse
import os
import torch
import json

def add_bool_argument(parser, name, default=False, help=''):
    parser.add_argument(f'--{name}', dest=name, action='store_true', help=help)
    parser.add_argument(f'--no-{name}', dest=name, action='store_false', help=argparse.SUPPRESS)
    parser.set_defaults(**{name: default})

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ##  ===== dataloader =====
    parser.add_argument("--dataset_name", default=['pdgam'], help="datasets e.g., ['pdgam']")
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
    parser.add_argument('--augment', default=0, type=int)
    parser.add_argument("--get_whole_motion", type=int, default=0)

    ## ===== optimization =====
    parser.add_argument('--max_epoch', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-6, type=float, help='max learning rate')
    parser.add_argument('--milestones', default=[100000, 120000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    
    ## ===== loss =====
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss_vel', type=float, default=0.5, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    parser.add_argument('--rec_rotloss', type=str, default='geol1', help='reconstruction rotational loss')
    # classifier
    parser.add_argument('--severity_pool', type=str, default='attention', help='quantized pooling type for classifier')
    parser.add_argument('--switch_clsloss', type=float, default=10e20, help='switch to classification loss after this many epochs')
    parser.add_argument('--w_clsloss', type=float, default=0.01, help='hyper-parameter for the classification loss')
    parser.add_argument('--predictor_lr', default=2e-7, type=float, help='max learning rate for disease predictors network')
    add_bool_argument(parser, 'use_mpredictor', default=True, help='use motion-latent predictor')
    parser.add_argument('--w_mcls', type=float, default=1.0, help='weight for motion latent adversarial classification loss')
    # discriminator
    parser.add_argument('--w_discrimloss', type=float, default=0.0, help='hyper-parameter for the discriminator loss')
    parser.add_argument('--discrim_pool', type=str, default='mean', help='quantized pooling type for discriminator')
    # orthogonality
    parser.add_argument('--w_md_orthogonalloss', type=float, default=0.0, help='hyper-parameter for the motion disease space orthogonality loss')
    # dlv_h_loss
    parser.add_argument('--w_dlvhloss', type=float, default=0.0, help='hyper-parameter for Disease Latent Variance for Healthy Samples (DLV-H) loss')
    # Cross-Covariance Loss
    parser.add_argument('--w_crossCov', type=float, default=0.0, help='hyper-parameter for Cross-Covariance loss')
    # Interference Loss
    parser.add_argument('--w_interf', type=float, default=0.0, help='hyper-parameter for Interference loss')
    parser.add_argument('--interf_loss_for_healthy_only', action="store_true", help='Interference loss for healthy samples only')

    ## ===== vqvae arch ===== 
    parser.add_argument("--code_dim", type=int, default=64, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="num of resblocks for each res")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices=['relu', 'silu', 'gelu'],
                        help='dataset directory')
    parser.add_argument('--vq_norm', type=str, default=None, help='dataset directory')
    add_bool_argument(parser, 'conditional', default=True, help='conditional vq')
    parser.add_argument('--style', action="store_true", help='Style decoder')
    parser.add_argument('--num_quantizers', type=int, default=6, help='num_quantizers')
    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')
    # parser.add_argument('--use_vq_prob', type=float, default=0.8, help='quantize_dropout_prob')
    ## ===== vqvae arch (disease) ===== 
    add_bool_argument(parser, 'disentangled', default=True, help="disentangled vqvae version")
    parser.add_argument("--code_dim_d", type=int, default=64, help="embedding dimension")
    parser.add_argument("--nb_code_d", type=int, default=128, help="nb of embedding")
    parser.add_argument("--down_t_d", type=int, default=2, help="downsampling rate")

    parser.add_argument('--ext', type=str, default='default', help='experiment folder')
    parser.add_argument('--mdcombine', type=str, default='add', help='motion disease combination')
    parser.add_argument('--disease_dropprob', type=float, default=0.0, help='dropout probability for disease')
    add_bool_argument(parser, 'Healthyzeroout', default=True, help='zero out disease latent')
    parser.add_argument('--addfactor', type=float, default=1, help='add factor for disease latent')
    
    ## ===== Pretrain motion encoder ===== 
    add_bool_argument(parser, 'pretrain_motion_encoder', default=True, help='Whether to pretrain the motion encoder')
    add_bool_argument(parser, 'use_pretrained_motion_encoder', default=True, help='Whether to use the pretrained motion encoder')
    parser.add_argument('--motion_encoder_lr_factor', type=float, default=0.1, help='Learning rate factor for motion encoder during main training')
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='Number of epochs to pretrain the motion encoder')

    ##  ===== other ===== 
    parser.add_argument('--name', type=str, default="test", help='Name of this trial')
    parser.add_argument('--is_continue', action="store_true", help='continue training')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log_every', default=10, type=int, help='iter log frequency')
    parser.add_argument('--codebook_plot_every', default=50, type=int, help='iter log frequency')
    parser.add_argument('--save_latest', default=500, type=int, help='iter save latest model frequency')
    parser.add_argument('--save_every_e', default=2, type=int, help='save model every n epoch')
    parser.add_argument('--eval_every_e', default=1, type=int, help='save eval results every n epoch')
    parser.add_argument('--evalvisual_every_e', default=50, type=int, help='eval frequency')
    # parser.add_argument('--early_stop_e', default=5, type=int, help='early stopping epoch')
    parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')

    parser.add_argument('--which_epoch', type=str, default="all", help='Name of this trial')

    ## For Res Predictor only
    parser.add_argument('--vq_name', type=str, default="rvq_nq6_dc512_nc512_noshare_qdp0.2", help='Name of this trial')
    # parser.add_argument('--n_res', type=int, default=2, help='Name of this trial')
    # parser.add_argument('--do_vq_res', action="store_true")
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument('--tiny', action="store_true", help='flag for using all data or tiny version of it')
    parser.add_argument("--wandb_runid", type=str, default="")
    
    opt = parser.parse_args()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        opt.device = torch.device("mps")
    elif torch.cuda.is_available():
        opt.device = torch.device(f"cuda:{opt.gpu_id}")
    else:
        opt.device = torch.device("cpu")

    if isinstance(opt.dataset_name, str):
        opt.dataset_name = json.loads(opt.dataset_name)
    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    opt.model_stage = 'vq'
    opt.db_save_name = ''
    for item in opt.dataset_name:
        opt.db_save_name = opt.db_save_name + item + '_'
    opt.db_save_name = opt.db_save_name[:-1]
    return opt

def save_opt(opt):
    if opt.is_train:
        args = vars(opt)
        expr_dir = os.path.join(opt.checkpoints_dir, opt.db_save_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            
        with open(file_name, 'rt') as opt_file:
            for line in opt_file:
                print(line.strip())
