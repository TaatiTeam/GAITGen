import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import time
import os
import sys
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.eval_t2m import evaluation_vqvae
from utils.utils import print_current_loss
from utils.losses import MDWALoss


def def_value():
    return 0.0


class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()
                
            # self.cls_criterion = torch.nn.CrossEntropyLoss() # MDWALoss(alpha=0.1, beta=0.7)   # Classification loss criterion
            self.cls_CE_criterion = torch.nn.CrossEntropyLoss()
            self.cls_mdwa_criterion = MDWALoss(alpha=0.2, beta=0.8)
            self.cls_criterion = self.cls_CE_criterion  # Set the default classification criterion
            self.switch_clsloss = args.switch_clsloss # By defualt it is a very large number and acts as regulare cls_CE_criterion if smaller values: cls_criterion switches to cls_mdwa_criterion
            
            # # Separate the parameters for different learning rates
            # classifier_params = list(self.vq_model.classifier.parameters())
            # attention_params = list(self.vq_model.cls_selfatt.parameters()) if hasattr(self.vq_model, 'cls_selfatt') else []
            # rnn_params = list(self.vq_model.cls_rnn.parameters()) if hasattr(self.vq_model, 'cls_rnn') else []
            # conv_params = list(self.vq_model.cls_conv.parameters()) + list(self.vq_model.pool.parameters()) if hasattr(self.vq_model, 'cls_conv') else []
            # decoder_params = list(self.vq_model.decoder.parameters())
            # allcls_params_plus_decoder = set(classifier_params + attention_params + rnn_params + conv_params + decoder_params)
            # vqvae_excludingDecoder_params = [p for n, p in self.vq_model.named_parameters() if p not in allcls_params_plus_decoder]

            # self.opt_vq_model = optim.AdamW([
            #     {'params': decoder_params, 'lr': self.opt.lr},
            #     {'params': vqvae_excludingDecoder_params, 'lr': self.opt.lr * 0.1},
            #     {'params': attention_params, 'lr': self.opt.lr * 0.1},
            #     {'params': rnn_params, 'lr': self.opt.lr * 0.1},
            #     {'params': conv_params, 'lr': self.opt.lr * 0.1},
            #     {'params': classifier_params, 'lr': self.opt.lr * 0.1}
            # ], betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)

            self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)



        # self.critic = CriticWrapper(self.opt.dataset_name, self.opt.device)
        # wandb.watch(self.vq_model, log="all")

    def forward(self, batch_data, severity_labels=None, num_layers=None, return_cls_logits=False):
        motions = batch_data.detach().to(self.device).float()
        if 'Conditional' in self.vq_model.__class__.__name__:
            input = (motions, severity_labels.to(self.device))
        else:
            input = (motions,)
        if num_layers is None:
            pred_motion, loss_commit, perplexity, cls_logits = self.vq_model(*input)
        else:
            with torch.no_grad():
                pred_motion, loss_commit, perplexity, cls_logits = self.vq_model(*input, num_layers=num_layers)
        self.motions = motions
        self.pred_motion = pred_motion

        loss_rec = self.l1_criterion(pred_motion, motions)
        pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        loss_explicit = self.l1_criterion(pred_local_pos, local_pos)
        if severity_labels is not None and cls_logits is not None:
            severity_labels = severity_labels.to(self.device)
            loss_cls = self.cls_criterion(cls_logits, severity_labels)
        else:
            loss_cls = torch.tensor(0.0).to(self.device)
        loss = loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * loss_commit + self.opt.w_clsloss * loss_cls

        # return loss, loss_rec, loss_vel, loss_commit, perplexity
        # return loss, loss_rec, loss_percept, loss_commit, perplexity
        if return_cls_logits:
            return loss, loss_rec, loss_explicit, loss_commit, perplexity, loss_cls, cls_logits
        return loss, loss_rec, loss_explicit, loss_commit, perplexity, loss_cls

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
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.vq_model.to(self.device)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("========================> Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(eval_val_loader)))
        # val_loss = 0
        # min_val_loss = np.inf
        # min_val_epoch = epoch
        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())

        # best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_vqvae(
        #     self.opt.model_dir, eval_val_loader, self.vq_model, epoch, best_fid=1000,
        #     best_div=100, best_top1=0,
        #     best_top2=0, best_top3=0, best_matching=100,
        #     eval_wrapper=eval_wrapper, save=False)
        
        epoch_bar = tqdm(range(epoch, self.opt.max_epoch), desc="Epoch Progress", unit="epoch")
        for epoch in epoch_bar:
            self.vq_model.train()
            for i, batch_data in enumerate(train_loader):
                motions, severity_labels = batch_data
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                    
                # Update the classification criterion at the specified iteration
                if epoch == self.switch_clsloss:
                    self.cls_criterion = self.cls_mdwa_criterion

                loss, loss_rec, loss_vel, loss_commit, perplexity, loss_cls, cls_logits = self.forward(motions, severity_labels, return_cls_logits=True)

                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                logs['train/loss'] += loss.item()
                logs['train/loss_rec'] += loss_rec.item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['train/loss_vel'] += loss_vel.item()
                logs['train/loss_commit'] += loss_commit.item()
                logs['train/perplexity'] += perplexity.item()
                logs['train/loss_cls'] += loss_cls.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                        wandb.log({tag: value / self.opt.log_every, "iteration": it})
                    logs = defaultdict(def_value, OrderedDict())
                    # print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                    
                # if it == 1 or it % self.opt.codebook_plot_every == 0:
                #     self.visualize_and_log_codebook_vectors(it)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
                
                # train_all_labels, train_all_preds = [], []
                # preds = cls_logits.argmax(dim=1).cpu().numpy() 
                # train_all_labels.extend(severity_labels.cpu().numpy())
                # train_all_preds.extend(preds)
                # train_macro_f1 = f1_score(train_all_labels, train_all_preds, average='macro')      
                # wandb.log({'train/macro_f1': train_macro_f1, "iteration": it})


            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            # if epoch % self.opt.save_every_e == 0:
            #     self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            self.vq_model.eval()
            val_loss_rec = []
            val_loss_vel = []
            val_loss_commit = []
            val_loss = []
            val_loss_cls = []
            val_perpexity = []
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    motions, severity_labels = batch_data
                    loss, loss_rec, loss_vel, loss_commit, perplexity, loss_cls, cls_logits = self.forward(motions, severity_labels, return_cls_logits=True)
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_vel.append(loss_vel.item())
                    val_loss_commit.append(loss_commit.item())
                    val_loss_cls.append(loss_cls.item())
                    val_perpexity.append(perplexity.item())
                    
                    severity_labels = severity_labels.cpu().numpy()
                    preds = cls_logits.argmax(dim=1).cpu().numpy() 
                    all_labels.extend(severity_labels)
                    all_preds.extend(preds)

            report = classification_report(all_labels, all_preds, output_dict=True)
            classification_report_df = pd.DataFrame(report).transpose()
            classification_report_markdown = self.dataframe_to_markdown_table(classification_report_df)
            confusion_matrix_data = confusion_matrix(all_labels, all_preds)
            confusion_matrix_data_normalized = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis]
            df_cm_normalized = pd.DataFrame(confusion_matrix_data_normalized, index=[i for i in range(len(confusion_matrix_data_normalized))],
                                columns=[i for i in range(len(confusion_matrix_data_normalized))])
            df_cm = pd.DataFrame(confusion_matrix_data, index=[i for i in range(len(confusion_matrix_data))],
                                columns=[i for i in range(len(confusion_matrix_data))])
            plt.figure(figsize=(10, 7))
            cmap = sns.color_palette("BuGn", as_cmap=True)
            sns.heatmap(df_cm_normalized, annot=df_cm, fmt="d", cmap=cmap)
            plt.title('Confusion Matrix (Normalized)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt_path = pjoin(self.opt.eval_dir, f'confusion_matrix_epoch_{epoch}.png')
            plt.savefig(plt_path)
            plt.close()
            wandb.log({"Classification Report": wandb.Html(f"<pre>{classification_report_markdown}</pre>"),
                   "Confusion Matrix": wandb.Image(plt_path)})
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            
            wandb.log({"val/loss": sum(val_loss) / len(val_loss), 
                       "val/loss_rec": sum(val_loss_rec) / len(val_loss_rec), 
                       "val/loss_vel": sum(val_loss_vel) / len(val_loss_vel), 
                       "val/loss_commit": sum(val_loss_commit) / len(val_loss_commit), 
                       "val/loss_cls": sum(val_loss_cls) / len(val_loss_cls),
                       "val/perplexity": sum(val_perpexity) / len(val_perpexity), 
                       "val/macro_f1": macro_f1,
                       "epoch": epoch})  # Add this block
            
            # val_loss = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)

            # print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f' %
            #       (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss), 
            #        sum(val_loss_vel)/len(val_loss), sum(val_loss_commit)/len(val_loss)))

            # if sum(val_loss) / len(val_loss) < min_val_loss:
            #     min_val_loss = sum(val_loss) / len(val_loss)
            # # if sum(val_loss_vel) / len(val_loss_vel) < min_val_loss:
            # #     min_val_loss = sum(val_loss_vel) / len(val_loss_vel)
            #     min_val_epoch = epoch
            #     self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
            #     print('Best Validation Model So Far!~')

            # best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_vqvae(
            #     self.opt.model_dir, eval_val_loader, self.vq_model, epoch, best_fid=best_fid,
            #     best_div=best_div, best_top1=best_top1,
            #     best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper)


            if epoch % self.opt.evalvisual_every_e == 0:
                with torch.no_grad():
                    all_layers_data = []
                    for num_layers in range(1, self.vq_model.quantizer.num_quantizers + 1):
                        _, _, _, _, _, _ = self.forward(motions, torch.tensor(severity_labels).to(self.device), num_layers=num_layers)
                        data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0).detach().cpu().numpy()
                        all_layers_data.append(data)
                    save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                    os.makedirs(save_dir, exist_ok=True)
                    plot_eval(all_layers_data, save_dir)
                    # if plot_eval is not None:
                    #     save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     plot_eval(data, save_dir)

            # if epoch - min_val_epoch >= self.ospt.early_stop_e:
            #     print('Early Stopping!~')
