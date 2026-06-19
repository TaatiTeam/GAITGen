import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import pandas as pd

import torch.optim as optim

import time
import numpy as np
from collections import OrderedDict, defaultdict
from utils.eval_t2m import evaluation_vqvae, evaluation_vqvae_plus_mpjpe
from utils.utils import print_current_loss
from models.vq.model import Discriminator, SeverityPredictor, grad_reverse
from torchmetrics.classification import F1Score

import os
import sys
import wandb
import time

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
from torch import nn
from utils.losses import MDWALoss, cal_geodesic_l1_loss

def def_value():
    return 0.0


class DisentangledRVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.tsne=False
        self.vq_model = vq_model
        self.device = args.device
        if self.opt.w_clsloss:
            self.disease_predictor_d = SeverityPredictor(args.code_dim_m, args.num_classes, args.severity_pool).to(self.device) # used code_dim_m to use classifier after the linear projection
            self.f1_metric_d = F1Score(task='multiclass', num_classes=args.num_classes, average='macro').to(self.device)
            if self.opt.use_mpredictor:
                self.disease_predictor_m = SeverityPredictor(args.code_dim_m, args.num_classes, args.severity_pool).to(self.device)
                self.f1_metric_m = F1Score(task='multiclass', num_classes=args.num_classes, average='macro').to(self.device)
        if self.opt.w_discrimloss:
            self.discriminator = Discriminator(args.code_dim_m, args.num_classes, args.discrim_pool, args.window_size, args.down_t).to(self.device)

        if args.is_train:
            #========Losses
            # Reconstruction loss
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()
            # Severity Predictor loss
            self.cls_CE_criterion = torch.nn.CrossEntropyLoss()
            self.cls_mdwa_criterion = MDWALoss(alpha=0.2, beta=0.8)
            self.severity_criterion = self.cls_CE_criterion  # Set the default classification criterion
            self.switch_clsloss = args.switch_clsloss # By defualt it is a very large number and acts as regulare cls_CE_criterion if smaller values: severity_criterion switches to cls_mdwa_criterion
            self.discriminator_loss = torch.nn.BCEWithLogitsLoss()  
            
            #========Optimizers
            # VQ-VAE model Optimizer
            if self.opt.w_clsloss:
                # optimizer_params = [{'params': self.vq_model.parameters(), 'lr': self.opt.lr},
                #         {'params': self.disease_predictor_d.parameters(), 'lr': self.opt.predictor_lr}]
                # if self.opt.use_mpredictor:
                #     optimizer_params.append({'params': self.disease_predictor_m.parameters(), 'lr': self.opt.predictor_lr})
                optimizer_params = [
                    # Motion components (updated slowly)
                    {'params': self.vq_model.motion_encoder.parameters(), 'lr': self.opt.lr * self.opt.motion_encoder_lr_factor},
                    {'params': self.vq_model.motion_quantizer.parameters(), 'lr': self.opt.lr * self.opt.motion_encoder_lr_factor},
                    {'params': self.vq_model.decoder.parameters(), 'lr': self.opt.lr * self.opt.motion_encoder_lr_factor},
                    # Disease components
                    {'params': self.vq_model.disease_encoder.parameters(), 'lr': self.opt.lr},
                    {'params': self.vq_model.disease_quantizer.parameters(), 'lr': self.opt.lr},
                    # Include disease_fc if it exists
                    {'params': self.vq_model.disease_fc.parameters(), 'lr': self.opt.lr} if hasattr(self.vq_model, 'disease_fc') else None,
                    # Include cross_attention if used
                    {'params': self.vq_model.cross_attention.parameters(), 'lr': self.opt.lr} if hasattr(self.vq_model, 'cross_attention') else None,
                    # Predictors
                    {'params': self.disease_predictor_d.parameters(), 'lr': self.opt.predictor_lr},
                    {'params': self.disease_predictor_m.parameters(), 'lr': self.opt.predictor_lr} if self.opt.use_mpredictor else None,
                ]
                optimizer_params = [param for param in optimizer_params if param is not None]      
                self.check_missing_params()     
                
                self.opt_vq_model = optim.AdamW(optimizer_params, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
            else:
                self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)
            # Auxiliary cls Optimizers
            # self.opt_disease_predictor_d = optim.Adam(self.disease_predictor_d.parameters(), lr=self.opt.lr)
            # self.opt_disease_predictor_m = optim.Adam(self.disease_predictor_m.parameters(), lr=self.opt.lr)
            # Discriminator Optimizer
            if self.opt.w_discrimloss:
                self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
    
    def check_missing_params(self):
        all_vq_params = set(self.vq_model.parameters())
        included_params = set()
        included_params.update(self.vq_model.motion_encoder.parameters())
        included_params.update(self.vq_model.motion_quantizer.parameters())
        included_params.update(self.vq_model.decoder.parameters())
        included_params.update(self.vq_model.disease_encoder.parameters())
        included_params.update(self.vq_model.disease_quantizer.parameters())
        if hasattr(self.vq_model, 'disease_fc'):
            included_params.update(self.vq_model.disease_fc.parameters())
        if hasattr(self.vq_model, 'cross_attention'):
            included_params.update(self.vq_model.cross_attention.parameters())
        missing_params = all_vq_params - included_params
        if missing_params:
            error_message = "Missing parameters in optimizer:\n"
            for param in missing_params:
                error_message += f"{param.shape}\n"
            raise RuntimeError(error_message)
        
    def calculate_dlv_h(self, d_latents, labels):
        """
        Calculate the Disease Latent Variance for Healthy Samples (DLV-H).
        """
        healthy_indices = (labels == 0).nonzero(as_tuple=True)[0]
        
        if len(healthy_indices) == 0:
            return torch.tensor(0.0, device=d_latents.device)
        healthy_d_latents = d_latents[healthy_indices]
        squared_norms = torch.norm(healthy_d_latents, dim=1) ** 2
        dlv_h = torch.mean(squared_norms)
        return dlv_h

    def prepare_vq_input(self, motions, severity_labels):
        if self.opt.conditional:
            return (motions, severity_labels.to(self.device))
        else:
            return (motions,)
        
    def forward(self, batch_data, num_layers=None, collect_clslogits=False, collect_latents=False):
        motions, severity_labels = batch_data
        motions = motions.detach().to(self.device).float()
        inp = self.prepare_vq_input(motions, severity_labels)
        
        # VQ-VAE forward pass
        if num_layers is None:
            pred_motion, loss_commit_m, loss_commit_d, perplexity, m_quantized, d_quantized = self.vq_model(*inp)
        else:
            with torch.no_grad():
                pred_motion, loss_commit_m, loss_commit_d, perplexity, m_quantized, d_quantized = self.vq_model(*inp, num_layers=num_layers)
                
        loss_commit = loss_commit_m + loss_commit_d
        
        # Flatten the latents to (batch_size, -1) before computing similarity
        m_flat = m_quantized.view(m_quantized.size(0), -1)
        d_flat = d_quantized.view(d_quantized.size(0), -1)
        # cosine_sim = F.cosine_similarity(m_flat, d_flat, dim=1)  # Similarity for each sample in the batch
        # avg_cosine_sim = cosine_sim.mean().item()
        m_flat_mean = m_flat - m_flat.mean(dim=0, keepdim=True) # latents are zero-centered
        d_flat_mean = d_flat - d_flat.mean(dim=0, keepdim=True)
        N = m_flat.size(0)
        C = (m_flat_mean.T @ d_flat_mean) / N  # Cross-Covariance Matrix Shape: (latent_dim_m, latent_dim_d)
        cross_covariance_loss = torch.norm(C, p='fro') ** 2  # Squared Frobenius norm


        self.motions = motions
        self.pred_motion = pred_motion
        
        # Reconstruction losses and explicit loss
        if self.opt.rec_rotloss == 'l1':
            loss_rec = self.l1_criterion(pred_motion, motions)
            pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            loss_explicit = self.l1_criterion(pred_local_pos, local_pos)
        elif self.opt.rec_rotloss == 'geol1':
            # Geodesic Loss + l1
            loss_rec = cal_geodesic_l1_loss(pred_motion, motions, self.opt.joints_num, self.l1_criterion)
            pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            loss_explicit = self.l1_criterion(pred_local_pos, local_pos)
            loss_recl1 = self.l1_criterion(pred_motion, motions)           
        else:
            raise ValueError('Invalid reconstruction loss specified!')
            
        
        # Severity Predictor loss
        total_cls_loss = 0
        severity_pred_logits_d = None
        severity_pred_logits_m = None
        if self.opt.w_clsloss:
            severity_labels = severity_labels.to(self.device)
            # Disease prediction from disease latent
            severity_pred_logits_d = self.disease_predictor_d(d_quantized)
            loss_cls_d = self.severity_criterion(severity_pred_logits_d, severity_labels)
            total_cls_loss = loss_cls_d
            # Disease prediction from motion latent with GRL
            if self.opt.use_mpredictor:
                m_quantized_reversed = grad_reverse(m_quantized)
                severity_pred_logits_m = self.disease_predictor_m(m_quantized_reversed)
                loss_cls_m_rev = self.severity_criterion(severity_pred_logits_m, severity_labels)
                total_cls_loss += self.opt.w_mcls*loss_cls_m_rev

        # Compute orthogonal loss
        md_orthogonal_loss = 0
        if self.opt.w_md_orthogonalloss:
            md_orthogonal_loss = torch.mean(torch.sum(m_quantized * d_quantized, dim=-1) ** 2)
        
        # # For Discriminator training
        discriminator_loss = 0
        if self.opt.w_discrimloss:            
            discriminator_loss_motion = self.discriminator_loss(
                                        self.discriminator(m_quantized.detach()),  # Detach to prevent gradients flowing into VQ-VAE
                                        torch.zeros(len(severity_labels), 1, device=self.device)
                                    )
            discriminator_loss_disease = self.discriminator_loss(
                                            self.discriminator(d_quantized.detach()),
                                            torch.ones(len(severity_labels), 1, device=self.device)
                                        )
            discriminator_loss = discriminator_loss_motion + discriminator_loss_disease
        
        #  Disease Latent Variance for Healthy Samples (DLV-H) -> Zero
        dlv_h_loss = 0
        if self.opt.w_dlvhloss:
            dlv_h_loss = self.calculate_dlv_h(d_quantized.view(d_quantized.size(0), -1), severity_labels)
            
        # Compute loss_interf
        if self.opt.w_interf:
            if self.opt.interf_loss_for_healthy_only:
                # Apply only to healthy samples
                mask = (severity_labels == 0).unsqueeze(1).unsqueeze(2).float()  # Shape: (batch_size, 1, 1)
                loss_interf = (d_quantized.abs() * mask).mean()
            else:
                # Apply to all samples
                loss_interf = d_quantized.abs().mean()
        else:
            loss_interf = torch.tensor(0.0, device=self.device)
        
        loss = loss_rec + self.opt.loss_vel * loss_explicit \
                + self.opt.commit * loss_commit + self.opt.w_clsloss * total_cls_loss  \
                + self.opt.w_md_orthogonalloss * md_orthogonal_loss \
                + self.opt.w_discrimloss * discriminator_loss \
                + self.opt.w_dlvhloss * dlv_h_loss \
                + self.opt.w_crossCov * cross_covariance_loss \
                + self.opt.w_interf * loss_interf
                    
        if collect_latents:
            self.collected_m_quantized.append(m_flat.cpu().numpy())
            self.collected_d_quantized.append(d_flat.cpu().numpy())
            self.collected_labels.append(severity_labels.detach().cpu().numpy())
        
        losses = {
            'loss': loss,
            'loss_rec': loss_rec,
            'loss_vel': loss_explicit,
            'loss_commit': loss_commit,
            'loss_commit_m': loss_commit_m,
            'loss_commit_d': loss_commit_d,
            'severity_loss': total_cls_loss if self.opt.w_clsloss else torch.tensor(0.0),
            'discriminator_loss': discriminator_loss if self.opt.w_discrimloss else torch.tensor(0.0),
            'md_orthogonal_loss': md_orthogonal_loss if self.opt.w_md_orthogonalloss else torch.tensor(0.0),
            'DLV-H': dlv_h_loss if self.opt.w_dlvhloss else torch.tensor(0.0),
            'severity_loss_d': loss_cls_d if self.opt.w_clsloss else torch.tensor(0.0),
            'severity_revloss_m': loss_cls_m_rev if self.opt.w_clsloss and self.opt.use_mpredictor else torch.tensor(0.0),
            'cross_covariance_loss': cross_covariance_loss,
            'loss_recl1_metric': loss_recl1 if self.opt.rec_rotloss == 'geol1' else torch.tensor(0.0),
            'interference_loss': loss_interf
            # 'cosine_similarity': avg_cosine_sim 
        }

        if collect_clslogits:
            return losses, perplexity, severity_pred_logits_d, severity_pred_logits_m
        return losses, perplexity

    def forward_discrim(self, batch_data):
        motions, severity_labels = batch_data
        motions = motions.detach().to(self.device).float()
        inp = self.prepare_vq_input(motions, severity_labels)
        
        # VQ-VAE forward pass
        _, _, _, _, m_quantized, d_quantized = self.vq_model(*inp)

        # # For Discriminator training
        discriminator_loss_motion = self.discriminator_loss(
                                        self.discriminator(m_quantized.detach()),  # Detach to prevent gradients flowing into VQ-VAE
                                        torch.zeros(len(severity_labels), 1, device=self.device)
                                    )
        discriminator_loss_disease = self.discriminator_loss(
                                        self.discriminator(d_quantized.detach()),
                                        torch.ones(len(severity_labels), 1, device=self.device)
                                    )
        discriminator_loss = discriminator_loss_motion + discriminator_loss_disease
        return discriminator_loss
    
    def plot_tsne(self, m_quantized, d_quantized, labels, epoch, mode='train'):
        tsne = TSNE(n_components=2)
        combined_quantized = np.concatenate([m_quantized, d_quantized], axis=0)
        tsne_results = tsne.fit_transform(combined_quantized)
        # Split back into `m_quantized` and `d_quantized` parts
        tsne_m = tsne_results[:m_quantized.shape[0]]
        tsne_d = tsne_results[m_quantized.shape[0]:]
        cmap = cm.get_cmap('tab10', len(np.unique(labels)))
        
        plt.figure(figsize=(20, 14))
        for i, class_label in enumerate(np.unique(labels)):
            indices = labels == class_label
            plt.scatter(tsne_m[indices, 0], tsne_m[indices, 1], label=f'Class {class_label} (m_quantized)', 
                        marker='*', s=100, alpha=0.7, color=cmap(i))
            plt.scatter(tsne_d[indices, 0], tsne_d[indices, 1], label=f'Class {class_label} (d_quantized)', 
                        marker='o', s=60, alpha=0.5, color=cmap(i))
        plt.title(f't-SNE of Quantized Vectors ({mode.capitalize()}) at Epoch {epoch}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        os.makedirs(pjoin(self.opt.eval_dir, 'tsne'), exist_ok=True)
        plt_path = pjoin(self.opt.eval_dir, 'tsne', f'tsne_{mode}_epoch_{epoch}.png')
        plt.savefig(plt_path)
        plt.close()
        wandb.log({f'tsne/({mode.capitalize()}) Epoch {epoch}': wandb.Image(plt_path), "epoch": epoch})
    
    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
            # Save discriminator and its optimizer if present
            "discriminator": self.discriminator.state_dict() if self.opt.w_discrimloss else None,
            "opt_discriminator": self.opt_discriminator.state_dict() if self.opt.w_discrimloss else None,
            # Classifiers
            "disease_predictor_d": self.disease_predictor_d.state_dict() if self.opt.w_clsloss else None,
            "disease_predictor_m": self.disease_predictor_m.state_dict() if self.opt.w_clsloss and self.opt.use_mpredictor else None,
                
        }
        torch.save(state, file_name)
        
    def get_codebook_vectors(self):
        return self.vq_model.get_codebook_vectors()
    
    def visualize_and_log_codebook_vectors(self, iteration):
        codebook_vectors = self.get_codebook_vectors()
        num_layers, nb_codes, code_dim = codebook_vectors.shape
        # Flatten the codebook vectors for t-SNE
        codebook_flat = codebook_vectors.view(-1, code_dim)
        # Create a layer index array for color coding
        layer_indices = torch.arange(num_layers).repeat_interleave(nb_codes).detach().cpu().numpy()
        
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(codebook_flat.detach().cpu().numpy())
        
        plt.figure(figsize=(10, 8))
        for layer in range(num_layers):
            indices = layer_indices == layer
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Layer {layer + 1}', s=10)
        
        # sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1])
        # ax = fig.add_subplot(111, projection='3d')
        # scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c='b', marker='o')
        plt.title(f't-SNE of Codebook Vectors at Iteration {iteration}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        # ax.set_title(f't-SNE of Codebook Vectors at Iteration {iteration}')
        # ax.set_xlabel('Component 1')
        # ax.set_ylabel('Component 2')
        # ax.set_zlabel('Component 3')
        plt_path = pjoin(self.opt.eval_dir, f'codebook_tsne_iter_{iteration}.png')
        plt.savefig(plt_path)
        plt.close()
        wandb.log({f'Codebook t-SNE': wandb.Image(plt_path), "iteration": iteration})

    def report_to_dataframe(self, report):
        report_data = []
        lines = report.split('\n')
        for line in lines[2:len(lines)-3]:
            row = {}
            row_data = line.split()
            if len(row_data) < 2: continue
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1-score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        return dataframe

    def dataframe_to_markdown_table(self, dataframe):
        return dataframe.to_markdown(index=False)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        # Load discriminator and its optimizer if present
        if self.opt.w_discrimloss and 'discriminator' in checkpoint and checkpoint['discriminator'] is not None:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        # Load svrity classifiers if present
        if self.opt.w_clsloss:
            if 'disease_predictor_d' in checkpoint:
                self.disease_predictor_d.load_state_dict(checkpoint['disease_predictor_d'])
            if 'disease_predictor_m' in checkpoint:
                self.disease_predictor_m.load_state_dict(checkpoint['disease_predictor_m'])
        return checkpoint['ep'], checkpoint['total_it']

    def pretrain_motion_encoder(self, train_loader, val_loader):
        print("=========Pretraining motion encoder and decoder...=========")
        self.vq_model.to(self.device)
        tlogs = defaultdict(def_value, OrderedDict())
        # Set up optimizer for motion encoder and decoder only
        motion_params = list(self.vq_model.motion_encoder.parameters()) + \
                        list(self.vq_model.motion_quantizer.parameters()) + \
                        list(self.vq_model.decoder.parameters())
        optimizer = optim.AdamW(motion_params, lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        # Training loop
        for epoch in range(self.opt.pretrain_epochs):
            st = time.time()
            self.vq_model.train()
            for batch_data in train_loader:
                motions, _ = batch_data  # We don't need severity_labels here
                motions = motions.to(self.device)
                # Forward pass through motion encoder and decoder
                pred_motion, loss_commit_m, m_perplexity, m_quantized = self.vq_model.forward_motion_only(motions)
                loss_rec = self.l1_criterion(pred_motion, motions)
                loss = loss_rec + self.opt.commit * loss_commit_m
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                tlogs['pretrainMotion/train_loss'] += loss.item()
                tlogs['pretrainMotion/train_loss_rec'] += loss_rec.item()
                tlogs['pretrainMotion/train_loss_commit_m'] += loss_commit_m.item()
            if epoch == 0: 
                print(f'Epoch 0 completed in {time.time() - st:.2f} seconds')
                
            self.vq_model.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    motions, _ = batch_data
                    motions = motions.to(self.device)
                    pred_motion, loss_commit_m, m_perplexity, m_quantized = self.vq_model.forward_motion_only(motions)
                    loss_rec = self.l1_criterion(pred_motion, motions)
                    loss = loss_rec + self.opt.commit * loss_commit_m
                    
                    tlogs['pretrainMotion/val_loss'] += loss.item()
                    tlogs['pretrainMotion/val_loss_rec'] += loss_rec.item()
                    tlogs['pretrainMotion/val_loss_commit_m'] += loss_commit_m.item()
            
            mean_loss = OrderedDict()
            for tag, value in tlogs.items():
                if 'val' in tag:
                    cnt = len(val_loader)
                else:
                    cnt = len(train_loader)
                mean_loss[tag] = value / cnt
                wandb.log({tag: value / cnt, "epoch": epoch})
            tlogs = defaultdict(def_value, OrderedDict())
        # Save the pretrained weights
        torch.save(self.vq_model.motion_encoder.state_dict(), pjoin(self.opt.model_dir, 'motion_encoder_pretrained.pth'))
        torch.save(self.vq_model.motion_quantizer.state_dict(), pjoin(self.opt.model_dir, 'motion_quantizer_pretrained.pth'))
        torch.save(self.vq_model.decoder.state_dict(), pjoin(self.opt.model_dir, 'decoder_pretrained.pth'))

    
    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        if self.opt.use_pretrained_motion_encoder:
            self.vq_model.motion_encoder.load_state_dict(torch.load(pjoin(self.opt.model_dir, 'motion_encoder_pretrained.pth')))
            self.vq_model.motion_quantizer.load_state_dict(torch.load(pjoin(self.opt.model_dir, 'motion_quantizer_pretrained.pth')))
            self.vq_model.decoder.load_state_dict(torch.load(pjoin(self.opt.model_dir, 'decoder_pretrained.pth')))
            print("Loaded pretrained motion encoder and decoder -> Training disentangled VQ-VAE model...")

        self.vq_model.to(self.device)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            epoch += 1
            print("========================> Load model, resuming training from epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(eval_val_loader)))

        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())

        # =======Evaluation========= 
        best = {
            'fid': 1000,
            'div': 100,
            'top1': 0,
            'top2': 0,
            'top3': 0,
            'matching': 100
        }
        best['fid'], best['div'], best['top1'], best['top2'], best['top3'], best['matching'] = evaluation_vqvae(self.opt, 
            self.opt.model_dir, eval_val_loader, self.vq_model, epoch, best_fid=best['fid'],
            best_div=best['div'], best_top1=best['top1'],
            best_top2=best['top2'], best_top3=best['top3'], best_matching=best['matching'],
            eval_wrapper=eval_wrapper, save=False)
        # ==========================
        
        while epoch <= self.opt.max_epoch:
            print(f'Epoch: {epoch}/{self.opt.max_epoch}')
            ep_st = time.time()
            self.vq_model.train()
            self.collected_m_quantized, self.collected_d_quantized, self.collected_labels = [], [], []
            self.tsne = False
            if epoch == self.opt.max_epoch+1: #Never Run this -> change if you want TSNE plots
                self.tsne = True
            for i, batch_data in enumerate(train_loader):
                _, severity_labels = batch_data
                severity_labels = severity_labels.to(self.device)
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                    
                # Update the classification criterion at the specified iteration
                if epoch == self.switch_clsloss:
                    self.severity_criterion = self.cls_mdwa_criterion
                
                #=====Update the discriminator model=====
                if self.opt.w_discrimloss:
                    # Disable gradients for VQ-VAE model
                    for p in self.vq_model.parameters():
                        p.requires_grad = False
                    for p in self.discriminator.parameters():
                        p.requires_grad = True
                    self.opt_discriminator.zero_grad()
                    discriminator_loss = self.forward_discrim(batch_data)
                    discriminator_loss.backward()
                    self.opt_discriminator.step()
                    
                    # Enable gradients for VQ-VAE model
                    for p in self.vq_model.parameters():
                        p.requires_grad = True
                    # Disable gradients for discriminator
                    for p in self.discriminator.parameters():
                        p.requires_grad = False
                # ================================
                
                losses, perplexity, cls_logits_d, cls_logits_m = self.forward(batch_data, collect_clslogits=True, collect_latents=self.tsne)
                loss = losses['loss']
                #=====Update the VQ-VAE model=====
                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                logs['train/loss'] += loss.item()
                logs['train/loss_rec'] += losses['loss_rec'].item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['train/loss_vel'] += losses['loss_vel'].item()
                logs['train_codebookloss/loss_commit'] += losses['loss_commit'].item()
                logs['train_codebookloss/loss_commit_mot'] += losses['loss_commit_m'].item()
                logs['train_codebookloss/loss_commit_dis'] += losses['loss_commit_d'].item()
                logs['train_codebookloss/perplexity'] += perplexity.item()
                if self.opt.w_clsloss:
                    logs['train/loss_cls'] += losses['severity_loss'].item()
                    logs['train/loss_cls_d'] += losses['severity_loss_d'].item()
                    if self.opt.use_mpredictor:
                        logs['train/revloss_cls_m'] += losses['severity_revloss_m'].item()
                if self.opt.w_discrimloss:
                    logs['train/discriminator_loss'] += losses['discriminator_loss'].item()
                if self.opt.w_md_orthogonalloss:
                    logs['train/motdis_orthogonal_loss'] += losses['md_orthogonal_loss'].item()
                if self.opt.w_dlvhloss:
                    logs['train/DLV-H'] += losses['DLV-H'].item()
                if self.opt.rec_rotloss == 'geol1':
                    logs['train/loss_recl1_metric'] += losses['loss_recl1_metric'].item()
                logs['train/cross_covariance_loss'] += losses['cross_covariance_loss'].item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                        wandb.log({tag: value / self.opt.log_every, "iteration": it})
                    logs = defaultdict(def_value, OrderedDict())
                    #print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                # if it == 1 or it % self.opt.codebook_plot_every == 0:
                #     self.visualize_and_log_codebook_vectors(it)
                if cls_logits_d is not None:
                    self.f1_metric_d.update(cls_logits_d, severity_labels)
                    train_macro_f1_d = self.f1_metric_d.compute()
                    wandb.log({'train/macro_f1_disease': train_macro_f1_d, 'iteration': it})
                    self.f1_metric_d.reset()
                if cls_logits_m is not None:
                    self.f1_metric_m.update(cls_logits_m, severity_labels)
                    train_macro_f1_m = self.f1_metric_m.compute()
                    disentanglement_score = train_macro_f1_d - train_macro_f1_m
                    wandb.log({
                        'train/macro_f1_motion': train_macro_f1_m,
                        'train/disentanglement_score': disentanglement_score,
                        'iteration': it})
                    self.f1_metric_m.reset()

            if epoch == 0:
                print('Epoch %d Time: %.2f' % (epoch, time.time() - ep_st))
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            
            if self.tsne:
                ttt = time.time()
                m_quantized_all_train = np.concatenate(self.collected_m_quantized, axis=0)
                d_quantized_all_train = np.concatenate(self.collected_d_quantized, axis=0)
                labels_all_train = np.concatenate(self.collected_labels, axis=0)
                self.plot_tsne(m_quantized_all_train, d_quantized_all_train, labels_all_train, epoch, mode='train')
                print('TSNE TRAIN Time: %.2f' % (time.time() - ttt))

            best = self.validate(val_loader, epoch, plot_eval, eval_val_loader, eval_wrapper, best)            
            # if epoch - min_val_epoch >= self.opt.early_stop_e:
            #     print('Early Stopping!~')
            epoch += 1
        
        print('-------> Training Finished! Final evaluation results:')
        _, _, _, _, _ = evaluation_vqvae_plus_mpjpe(self.opt, eval_val_loader, self.vq_model, 0, eval_wrapper=eval_wrapper, num_joint=self.opt.joints_num, eval_dir=self.opt.eval_dir)

    def validate(self, val_loader, epoch, plot_eval, eval_val_loader, eval_wrapper, best):
        self.vq_model.eval()
        self.collected_m_quantized, self.collected_d_quantized, self.collected_labels = [], [], []
        val_loss_rec, val_loss_vel = [], []
        val_loss_commit, val_loss_commit_m, val_loss_commit_d = [], [], []
        val_loss, val_loss_cls, val_loss_cls_d, val_revloss_cls_m = [], [], [], []
        val_discriminator_loss, val_md_ortho_loss = [], []
        val_dlvh, val_cross_covariance_loss = [], []
        val_perplexity = []
        val_loss_recl1_metric = []
        all_preds_d, all_preds_m, all_labels = [], [], []
        val_st = time.time()
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                motions, severity_labels = batch_data
                losses, perplexity, cls_logits_d, cls_logits_m = self.forward(batch_data, collect_clslogits=True, collect_latents=self.tsne)
                val_loss.append(losses['loss'].item())
                val_loss_rec.append(losses['loss_rec'].item())
                val_loss_vel.append(losses['loss_vel'].item())
                val_loss_commit.append(losses['loss_commit'].item())
                val_loss_commit_m.append(losses['loss_commit_m'].item())
                val_loss_commit_d.append(losses['loss_commit_d'].item())
                val_loss_cls.append(losses['severity_loss'].item())
                val_loss_cls_d.append(losses['severity_loss_d'].item())
                val_revloss_cls_m.append(losses['severity_revloss_m'].item())
                val_discriminator_loss.append(losses['discriminator_loss'].item())
                val_dlvh.append(losses['DLV-H'].item())
                val_cross_covariance_loss.append(losses['cross_covariance_loss'].item())
                val_md_ortho_loss.append(losses['md_orthogonal_loss'].item())
                val_loss_recl1_metric.append(losses['loss_recl1_metric'].item())
                val_perplexity.append(perplexity.item())
                
                severity_labels = severity_labels.cpu().numpy()
                all_labels.extend(severity_labels)
                if cls_logits_d is not None:
                    preds_d = cls_logits_d.argmax(dim=1).cpu().numpy() 
                    all_preds_d.extend(preds_d)
                if cls_logits_m is not None:
                    preds_m = cls_logits_m.argmax(dim=1).cpu().numpy() 
                    all_preds_m.extend(preds_m)

        if len(all_preds_d):
            self.create_validation_report(all_labels, all_preds_d, all_preds_m, epoch)
            macro_f1_d = f1_score(all_labels, all_preds_d, average='macro')
        if len(all_preds_m):
            macro_f1_m = f1_score(all_labels, all_preds_m, average='macro')
        log_dict = {
            "val/loss": sum(val_loss) / len(val_loss),
            "val/loss_rec": sum(val_loss_rec) / len(val_loss_rec),
            "val/loss_vel": sum(val_loss_vel) / len(val_loss_vel),
            "val_codebookloss/loss_commit": sum(val_loss_commit) / len(val_loss_commit),
            "val_codebookloss/loss_commit_mot": sum(val_loss_commit_m) / len(val_loss_commit_m),
            "val_codebookloss/loss_commit_dis": sum(val_loss_commit_d) / len(val_loss_commit_d),
            "val/loss_cls": sum(val_loss_cls) / len(val_loss_cls) if self.opt.w_clsloss else None,
            "val/loss_cls_d": sum(val_loss_cls_d) / len(val_loss_cls_d) if self.opt.w_clsloss else None,
            "val/revloss_cls_m": sum(val_revloss_cls_m) / len(val_revloss_cls_m) if self.opt.w_clsloss else None,
            "val/discriminator_loss": sum(val_discriminator_loss) / len(val_discriminator_loss) if self.opt.w_discrimloss else None,
            "val/motdis_orthogonal_loss": sum(val_md_ortho_loss) / len(val_md_ortho_loss) if self.opt.w_md_orthogonalloss else None,
            "val/DLV-H": sum(val_dlvh) / len(val_dlvh) if self.opt.w_dlvhloss else None,
            "val_codebookloss/perplexity": sum(val_perplexity) / len(val_perplexity),
            "val/cross_covariance_loss": sum(val_cross_covariance_loss) / len(val_cross_covariance_loss),
            "val/macro_f1_disease": macro_f1_d if len(all_preds_d) else None,
            "val/macro_f1_motion": macro_f1_m if len(all_preds_m) else None,
            'val/disentanglement_score': macro_f1_d - macro_f1_m if len(all_preds_d) and len(all_preds_m) else None,
            "val/loss_recl1_metric": sum(val_loss_recl1_metric) / len(val_loss_recl1_metric) if self.opt.rec_rotloss == 'geol1' else None,
            "epoch": epoch
        }
        log_dict = {k: v for k, v in log_dict.items() if v is not None}
        wandb.log(log_dict)

        if epoch == 0:
            print('Validation Time: %.2f' % (time.time() - val_st))
        
        # if sum(val_loss) / len(val_loss) < min_val_loss:
        #     min_val_loss = sum(val_loss) / len(val_loss)
        # # if sum(val_loss_vel) / len(val_loss_vel) < min_val_loss:
        # #     min_val_loss = sum(val_loss_vel) / len(val_loss_vel)
        #     min_val_epoch = epoch
        #     self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
        #     print('Best Validation Model So Far!~')

        best['fid'], best['div'], best['top1'], best['top2'], best['top3'], best['matching'] = evaluation_vqvae(self.opt,
            self.opt.model_dir, eval_val_loader, self.vq_model, epoch, best_fid=best['fid'],
            best_div=best['div'], best_top1=best['top1'],
            best_top2=best['top2'], best_top3=best['top3'], best_matching=best['matching'], eval_wrapper=eval_wrapper)
        
        if self.tsne:
            ttt = time.time()
            m_quantized_all_val = np.concatenate(self.collected_m_quantized, axis=0)
            d_quantized_all_val = np.concatenate(self.collected_d_quantized, axis=0)
            labels_all_val = np.concatenate(self.collected_labels, axis=0)
            self.plot_tsne(m_quantized_all_val, d_quantized_all_val, labels_all_val, epoch, mode='val')
            print('TSNE VAL Time: %.2f' % (time.time() - ttt))
        
        # =================Plotting random 8 reconstructed eval motions=================
        vis_time = time.time()
        # if epoch % self.opt.evalvisual_every_e == 0 or epoch == self.opt.max_epoch:
        if epoch == self.opt.max_epoch:
            with torch.no_grad():
                all_layers_data, gt_motions, pred_motions = [], [], []
                motions, labels = batch_data
                unique_labels = torch.unique(labels)
                numsample = len(unique_labels) * 2
                _, _ = self.forward(batch_data)
                for label in unique_labels:
                    label_filter = (labels == label)
                    class_motions = self.motions[label_filter][:2]
                    class_pred_motion = self.pred_motion[label_filter][:2] 
                    gt_motions.append(class_motions.detach().cpu().numpy())
                    pred_motions.append(class_pred_motion.detach().cpu().numpy())
                gt_motions = np.concatenate(gt_motions, axis=0)  # Shape should be (8, 64, 263)
                pred_motions = np.concatenate(pred_motions, axis=0)  # Shape should also be (8, 64, 263)
                # Concatenate the ground truth and predicted motions to get (16, 64, 263)
                all_layers_data = np.concatenate([gt_motions, pred_motions], axis=0)
            
                # all_layers_data = torch.cat([self.motions[:numsample], self.pred_motion[:numsample]], dim=0).detach().cpu().numpy()
                # Ucomment to visualize per Quantization layer and change the plot eval function to "plot_t2m_codebooklayers"
                # all_layers_data = []
                # for num_layers in range(1, self.vq_model.motion_quantizer.num_quantizers + 1):
                #     _, _ = self.forward(batch_data, num_layers=num_layers)
                #     data = torch.cat([self.motions[:numsample], self.pred_motion[:numsample]], dim=0).detach().cpu().numpy()
                #     all_layers_data.append(data)
                save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(all_layers_data, save_dir, numsample)
        if epoch == 0:
            print('Eval Visualization Time: %.2f' % (time.time() - vis_time))
        return best
        
    def create_validation_report(self, all_labels, all_preds_d, all_preds_m, epoch):
        report = classification_report(all_labels, all_preds_d, output_dict=True)
        classification_report_df = pd.DataFrame(report).transpose()
        classification_report_markdown = self.dataframe_to_markdown_table(classification_report_df)
        if epoch == self.opt.max_epoch:
            print(f"Classification Report for DISEASE LATENT Epoch {epoch}:\n{classification_report_df}")
        
        confusion_matrix_data = confusion_matrix(all_labels, all_preds_d)
        confusion_matrix_data_normalized = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis]
        df_cm_normalized = pd.DataFrame(confusion_matrix_data_normalized, index=[i for i in range(len(confusion_matrix_data_normalized))],
                            columns=[i for i in range(len(confusion_matrix_data_normalized))])
        df_cm = pd.DataFrame(confusion_matrix_data, index=[i for i in range(len(confusion_matrix_data))],
                            columns=[i for i in range(len(confusion_matrix_data))])
        plt.figure(figsize=(10, 7))
        cmap = sns.color_palette("BuGn", as_cmap=True)
        sns.heatmap(df_cm_normalized, annot=df_cm, fmt="d", cmap=cmap)
        plt.title('Confusion Matrix (Normalized) - DISEASE LATENT')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        os.makedirs(pjoin(self.opt.outdir, 'confusion_mat'), exist_ok=True)
        plt_path = pjoin(self.opt.outdir, 'confusion_mat', f'confusion_matrix_epoch_{epoch}.png')
        plt.savefig(plt_path)
        plt.close()
        if epoch == self.opt.max_epoch:
            wandb.log({"Confusion Matrix": wandb.Image(plt_path)})
        # wandb.log({"Classification Report": wandb.Html(f"<pre>{classification_report_markdown}</pre>"),
        #         "Confusion Matrix": wandb.Image(plt_path)})
