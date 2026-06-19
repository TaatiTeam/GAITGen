import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
import clip
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class DMaskTransformer(nn.Module):
    def __init__(self, code_dim_m, code_dim_d, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, **kargs):
        super(DMaskTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim_m = code_dim_m
        self.code_dim_d = code_dim_d
        self.code_dim = max(self.code_dim_m, self.code_dim_d)
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.num_motion_tokens = opt.num_motion_tokens
        self.num_disease_tokens = opt.num_disease_tokens
        self.num_special_tokens = 3  # mask, pad, motion_end
        self._num_tokens = self.num_motion_tokens + self.num_disease_tokens + self.num_special_tokens
        self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1
        self.motion_end_token_id = opt.num_tokens + 2


        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_classes' in vars(opt).keys(), 'num_classes must be provided for action condition mode'
        self.num_actions = opt.num_classes

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.output_process_motion = OutputProcess_Bert(out_feats=self.num_motion_tokens, latent_dim=latent_dim)
        self.output_process_disease = OutputProcess_Bert(out_feats=self.num_disease_tokens, latent_dim=latent_dim)

        self.token_emb = nn.Embedding(self._num_tokens, self.code_dim)

        self.apply(self.__init_weights)

        '''
        Preparing frozen weights
        '''

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.noise_schedule = cosine_schedule

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        # TODO: FIX for disentangled model
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)
        # self.token_emb.weight.requires_grad = False
        # self.token_emb_ready = True
        print("Token embedding initialized!")

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, md_ids, cond, padding_mask, idx_motion_end, force_mask=False):
        '''
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param idx_motion_end: maximum length of motion sequences in the batch
        :param force_mask: boolean
        :return:
            -logits: (b, num_token, seqlen)
        '''

        cond = self.mask_cond(cond, force_mask=force_mask)

        # print(motion_ids.shape)
        x = self.token_emb(md_ids)
        # print(x.shape)
        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(x)

        cond = self.cond_emb(cond).unsqueeze(0) #(1, b, latent_dim)

        x = self.position_enc(x)
        xseq = torch.cat([cond, x], dim=0) #(seqlen+1, b, latent_dim)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1) #(b, seqlen+1)
        
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:] #(seqlen, b, e)
        # output: shape (seq_len, bs, latent_dim)
        output = output.permute(1, 0, 2)  # Shape: (bs, seq_len, latent_dim)
        
        bs, seq_len, _ = output.shape
        n = (seq_len - 1)//2
        outputs_m = torch.zeros(bs, n, output.shape[2], device=output.device)
        outputs_d = torch.zeros(bs, n, output.shape[2], device=output.device)
        for i in range(bs):
            # Extract outputs for motion tokens
            motion_len = idx_motion_end[i]
            outputs_m[i, :motion_len] = output[i, :motion_len, :]
            # Extract outputs for disease tokens
            disease_len = motion_len # ToDo: assume same number of disease tokens as motion tokens
            outputs_d[i, :disease_len] = output[i, motion_len + 1:motion_len + 1 + disease_len, :]

        # Pass through output heads
        logits_m = self.output_process_motion(outputs_m.permute(1, 0, 2)) #(n, b, e) -> (b, ntoken, n)
        logits_d = self.output_process_disease(outputs_d.permute(1, 0, 2)) #(n, b, e) -> (b, ntoken, n)

        return logits_m, logits_d

    def forward(self, ids_m, ids_d, y, m_lens_m, m_lens_d):
        '''
        :param ids: (b, n)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''

        bs, n = ids_m.shape
        device = ids_m.device
        
        # Shift disease token IDs
        ids_d = ids_d + self.num_motion_tokens
        
        # Remove padding from ids_m and ids_d based on their lengths
        ids_m_nonpad = [ids_m[i].narrow(0, 0, m_lens_m[i]) for i in range(bs)]
        ids_d_nonpad = [ids_d[i].narrow(0, 0, m_lens_d[i]) for i in range(bs)]
        
        # Create motion end token IDs
        motion_end_token = torch.full((bs, 1), self.motion_end_token_id, device=device, dtype=torch.long)
        # Concatenate tokens for each sample
        ids_list = [torch.cat([ids_m_nonpad[i], motion_end_token[i], ids_d_nonpad[i]]) for i in range(bs)]
        m_lens = m_lens_m + 1 + m_lens_d # Total sequence length
        ntokens = n + 1 + n # Maximum sequence length (99 = 49+49+1)
        m_lens2 = torch.tensor([len(seq) for seq in ids_list], device=device)
        assert torch.equal(m_lens, m_lens2)
        
        # Explicitly pad each sequence to (n + 1 + n) length
        ids_list_padded = [F.pad(seq, (0, ntokens - len(seq)), value=self.pad_id) for seq in ids_list]
        # Stack the padded sequences into a batch (bs, n + 1 + n)
        ids = torch.stack(ids_list_padded, dim=0)  # Shape: (bs, ntokens)

        assert ntokens == ids.shape[1], f"number of tokens mismatch"

        non_pad_mask = lengths_to_mask(m_lens, ntokens) #(b, n*2+1)  # Positions that are valid (non-padding)
        ids = torch.where(non_pad_mask, ids, self.pad_id)
        
        # Create position indices 
        position_indices = torch.arange(ntokens, device=device).unsqueeze(0).expand(bs, -1) # (batch_size, seq_len)
        # Get motion end positions
        motion_end_positions = m_lens_m
        
        # Create token_type_mask
        # token_type_mask: (batch_size, seq_len) -> each element indicates the token type at that position.
        token_type_mask = torch.full((bs, ntokens), fill_value=2, dtype=torch.long, device=device)
        token_type_mask[position_indices < motion_end_positions.unsqueeze(1)] = 0  # Motion tokens
        token_type_mask[position_indices == motion_end_positions.unsqueeze(1)] = 1  # Motion end token
        token_type_mask = torch.where(~non_pad_mask, -1, token_type_mask)  # Pad tokens

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        '''
        Prepare mask
        '''
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1) # Positions to be MASKED are ALL TRUE

        mask &= non_pad_mask    # Do not mask padding tokens
        # Ensure the motion end token is not masked
        mask_motion_end = torch.ones_like(ids, dtype=torch.bool)
        mask_motion_end.scatter_(1, motion_end_positions.unsqueeze(1), False)
        mask &= mask_motion_end

        # Note this is our training target, not input
        labels = torch.where(mask, ids, self.mask_id)
        # Set the label at the motion <end> token position to ignore_index
        labels[torch.arange(bs, device=device), motion_end_positions] = self.mask_id
        
        # Initialize lists to hold labels for motion and disease tokens
        labels_m = torch.full((bs, n), self.mask_id, device=device)
        labels_d = torch.full((bs, n), self.mask_id, device=device)
        for i in range(bs):
            # Extract labels for motion tokens (positions before motion end token)
            labels_m[i, :motion_end_positions[i]] = labels[i, :motion_end_positions[i]]
            # Extract labels for disease tokens (positions after motion end token)
            disease_len = m_lens[i] - motion_end_positions[i] - 1
            labels_d[i, :motion_end_positions[i]] = labels[i, motion_end_positions[i] + 1:motion_end_positions[i] + 1 + disease_len]
        
        # Adjust labels_disease indices only for valid tokens, leave mask_id unchanged
        labels_d_adjusted = torch.where(labels_d != self.mask_id, labels_d - self.num_motion_tokens, labels_d)

        x_ids = ids.clone()

        # Further Apply Bert Masking Scheme
        # ===============
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        
        # Generate random tokens according to token_type_mask      
        rand_tokens = torch.randint(0, self.num_motion_tokens + self.num_disease_tokens, x_ids.shape, device=device)
        random_tokens = torch.where(token_type_mask == 0, rand_tokens % self.num_motion_tokens, rand_tokens)
        random_tokens = torch.where(token_type_mask == 2, rand_tokens % self.num_disease_tokens + self.num_motion_tokens, random_tokens)
        
        # Apply random token replacement where mask_rid is True
        x_ids = torch.where(mask_rid, random_tokens, x_ids)
        # ===============
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        logits_m, logits_d = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, motion_end_positions, force_mask)
        
        ce_loss_m, pred_id_m, acc_m = cal_performance(logits_m, labels_m, ignore_index=self.mask_id)
        ce_loss_d, pred_id_d, acc_d = cal_performance(logits_d, labels_d_adjusted, ignore_index=self.mask_id)
        
        # Combine losses and accuracies
        ce_loss = ce_loss_m + ce_loss_d
        acc = (acc_m + acc_d) / 2

        return ce_loss, pred_id_m, pred_id_d, acc

    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                m_lens_m, 
                                cond_scale=3,
                                force_mask=False):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, m_lens_m, force_mask=True)

        logits_m, logits_d = self.trans_forward(motion_ids, cond_vector, padding_mask, m_lens_m)
        if cond_scale == 1:
            return logits_m, logits_d

        aux_logits_m, aux_logits_d = self.trans_forward(motion_ids, cond_vector, padding_mask, m_lens_m, force_mask=True)

        scaled_logits_m = aux_logits_m + (logits_m - aux_logits_m) * cond_scale
        scaled_logits_d = aux_logits_d + (logits_d - aux_logits_d) * cond_scale
        return scaled_logits_m, scaled_logits_d

    def combine_ids_prev(self, logits_m, logits_d, m_lens_m, m_lens_d):
        device = next(self.parameters()).device
        batch_size = len(m_lens_m)
        m_lens = m_lens_m + 1 + m_lens_d  # Total sequence lengths including motion end token
        seq_len = max(m_lens)
        
        # Shape: (b, max_motion_len, total_num_tokens), initialize with -inf
        logits_m_full = torch.full((batch_size, max(m_lens_m), self._num_tokens), float('-inf'), device=device)
        logits_m_full[:, :, :self.num_motion_tokens] = logits_m[:, :max(m_lens_m), :]
        # Prepare the motion end logits for the whole batch
        motion_end_logit = torch.full((batch_size, 1, self._num_tokens), float('-inf'), device=device)
        motion_end_logit[:, :, self.motion_end_token_id] = 1e5  # Assign high score to motion end token
        # Prepare combined logits for disease tokens
        # Shape: (b, max_disease_len, total_num_tokens), initialize with -inf
        logits_d_full = torch.full((batch_size, max(m_lens_d), self._num_tokens), float('-inf'), device=device)
        logits_d_full[:, :, self.num_motion_tokens:self.num_motion_tokens + self.num_disease_tokens] = logits_d[:, :max(m_lens_d), :]       
        
        # Concatenate motion logits, motion end logits, and disease logits
        logits_seq = torch.cat([logits_m_full, motion_end_logit, logits_d_full], dim=1)  # (bs, seq_len, total_num_tokens)
        # Calculate padding length for each sequence in the batch
        pad_lens = seq_len - m_lens  # (bs,)
        # Masking out the logits beyond the actual sequence length using padding
        mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= m_lens.unsqueeze(1)
        logits = logits_seq.masked_fill(mask.unsqueeze(-1), float('-inf'))  # Apply padding to logits
        
        return logits
    def combine_ids(self, logits_m, logits_d, m_lens_m, m_lens_d):
        '''
        Combine logits_m and logits_d into logits for the full sequence,
        inserting the motion end token logits appropriately.
        Args:
            logits_m: Tensor of shape (batch_size, motion_len, num_motion_tokens) e.g., (bs, 49, 512)
            logits_d: Tensor of shape (batch_size, disease_len, num_disease_tokens) e.g., (bs, 49, 128)
            m_lens_m: Tensor of shape (batch_size,), lengths of motion sequences
            m_lens_d: Tensor of shape (batch_size,), lengths of disease sequences
        Returns:
            logits: Tensor of shape (batch_size, max_seq_len, total_num_tokens)
        '''
        device = logits_m.device
        batch_size = logits_m.size(0)

        # Calculate the maximum sequence length in the batch
        seq_lens = m_lens_m + 1 + m_lens_d  # +1 for motion end token
        max_seq_len = max(seq_lens)

        logits_list = []
        for i in range(batch_size):
            motion_len = m_lens_m[i]
            disease_len = m_lens_d[i]
            seq_len_i = seq_lens[i]  # Total length for sample i
            # Initialize logits for this sequence with -inf
            logits_seq = torch.full((seq_len_i, self._num_tokens), float('-1e9'), device=device)
            # Fill logits for motion tokens
            logits_seq[:motion_len, :self.num_motion_tokens] = logits_m[i, :motion_len, :]
            # Set high logit for motion end token
            logits_seq[motion_len, self.motion_end_token_id] = 1e5  # Assign high score to motion end token
            # Fill logits for disease tokens
            logits_seq[motion_len + 1:, self.num_motion_tokens:self.num_motion_tokens + self.num_disease_tokens] = logits_d[i, :disease_len, :]
            # Pad to max_seq_len if necessary
            pad_len = max_seq_len - seq_len_i
            if pad_len > 0:
                logits_seq = F.pad(logits_seq, (0, 0, 0, pad_len), value=float('-1e9'))
            logits_list.append(logits_seq)
        # Stack logits from all sequences
        logits = torch.stack(logits_list, dim=0)  # Shape: (batch_size, max_seq_len, total_num_tokens)
        return logits

    def separate_ids(self, ids, m_lens_m, m_lens_d):
        """
        Separate the generated ids into motion and disease ids based on their lengths.
        
        :param ids: Combined ids of shape (batch_size, seq_len)
        :param m_lens_m: Tensor of motion sequence lengths (batch_size,)
        :param m_lens_d: Tensor of disease sequence lengths (batch_size,)
        :return: ids_m, ids_d - Separated ids for motion and disease tokens
        """
        device = ids.device
        batch_size, seq_len = ids.shape
        
        ids_m_list = []
        ids_d_list = []
        
        for i in range(batch_size):
            motion_len = m_lens_m[i]
            disease_len = m_lens_d[i]
            
            # Extract ids for motion tokens
            ids_m = ids[i, :motion_len]
            ids_m_list.append(ids_m)
            
            # Extract ids for disease tokens (after motion end token)
            ids_d = ids[i, motion_len + 1:motion_len + 1 + disease_len]
            ids_d_list.append(ids_d)
        
        # Pad the sequences to ensure uniformity
        ids_m = pad_sequence(ids_m_list, batch_first=True, padding_value=self.pad_id)
        ids_d = pad_sequence(ids_d_list, batch_first=True, padding_value=self.pad_id)
        
        return ids_m, ids_d

            
    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens_m,
                 m_lens_d,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False
                 ):
        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers

        device = next(self.parameters()).device
        batch_size = len(m_lens_m)
        m_lens = m_lens_m + 1 + m_lens_d  # Total sequence lengths including motion end token
        seq_len = max(m_lens)


        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len) # True where padding

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        # Set motion end token at the appropriate positions
        motion_end_positions = m_lens_m  # Positions of motion end tokens
        batch_indices = torch.arange(batch_size, device=device)
        ids[batch_indices, motion_end_positions] = self.motion_end_token_id
        
        scores = torch.where(padding_mask, 1e5, 0.)
        scores = torch.where(ids==self.motion_end_token_id, 1e5, scores) # Do not mask motion end tokens
        starting_temperature = temperature
        
        # Create token_type_mask
        position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_type_mask = torch.full((batch_size, seq_len), fill_value=2, dtype=torch.long, device=device)
        token_type_mask[position_indices < motion_end_positions.unsqueeze(1)] = 0  # Motion tokens
        token_type_mask[position_indices == motion_end_positions.unsqueeze(1)] = 1  # Motion end token
        token_type_mask = torch.where(padding_mask, -1, token_type_mask)  # Pad tokens

        # Generation loop
        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # Determine the number of tokens to mask for each sequence
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(dim=1)  # (bs, seq_len)
            ranks = sorted_indices.argsort(dim=1)   # (bs, seq_len)
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits_m, logits_d = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  m_lens_m = m_lens_m,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)

            logits_m = logits_m.permute(0, 2, 1)  # (b, motion_len, ntoken)
            logits_d = logits_d.permute(0, 2, 1)  # (b, disease_len, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            
            logits = self.combine_ids(logits_m, logits_d, m_lens_m, m_lens_d)
            
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            
            for i in range(batch_size):
                assert torch.all(torch.any(~torch.isinf(filtered_logits[i]), dim=1)), "Each row must have at least one non-inf value."
            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids) #ToDo: not used for now - Check if motion <end> token needs to be -1
        ids = torch.where(ids==self.motion_end_token_id, -1, ids)
        
        motion_mask = token_type_mask == 0  # Positions where we expect motion tokens
        assert torch.all((ids[motion_mask] >= 0) & (ids[motion_mask] < self.num_motion_tokens)), "Invalid motion tokens detected!"
        disease_mask = token_type_mask == 2  # Positions where disease tokens should be
        assert torch.all((ids[disease_mask] >= self.num_motion_tokens) & (ids[disease_mask] < self.num_motion_tokens + self.num_disease_tokens)), "Invalid disease tokens detected!"
        
        # invalid_disease_ids = ids[disease_mask][(ids[disease_mask] < self.num_motion_tokens) | (ids[disease_mask] >= self.num_motion_tokens + self.num_disease_tokens)]
        # if len(invalid_disease_ids) > 0:
        #     invalid_positions = torch.nonzero((ids[disease_mask] < self.num_motion_tokens) | (ids[disease_mask] >= self.num_motion_tokens + self.num_disease_tokens), as_tuple=True)
        #     print(f"Invalid disease tokens detected at positions: {invalid_positions}")
        #     print(f"Invalid disease token values: {invalid_disease_ids}")
        #     assert False, "Disease token IDs out of range!"
        
        ids_m, ids_d = self.separate_ids(ids, m_lens_m, m_lens_d)
        
        ids_m = torch.where(ids_m==self.pad_id, -1, ids_m)
        ids_d = torch.where(ids_d==self.pad_id, -1, ids_d)
        # Update the ids_d with the adjusted disease token indices (removing padding and adjusting token range)
        ids_d = torch.where(ids_d != -1, ids_d - self.num_motion_tokens, ids_d)
        
        assert torch.all(((ids_d == -1) | ((ids_d >= 0) & (ids_d < self.num_disease_tokens)))), \
    "disease_ids contains values outside the range of the disease codebook."
        return ids, ids_m, ids_d


    @torch.no_grad()
    @eval_decorator
    def edit(self,
             conds,
             tokens,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             topk_filter_thres=0.9,
             gsample=False,
             force_mask=False,
             edit_mask=None,
             padding_mask=None,
             ):

        assert edit_mask.shape == tokens.shape if edit_mask is not None else True
        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask == None:
            padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        if edit_mask == None:
            mask_free = True
            ids = torch.where(padding_mask, self.pad_id, tokens)
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            scores = torch.where(edit_mask, 0., 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            ids = torch.where(edit_mask, self.mask_id, tokens)
            scores = torch.where(edit_mask, 0., 1e5)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = 0.16 if mask_free else self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            # is_mask = (torch.rand_like(scores) < 0.8) * ~padding_mask if mask_free else is_mask
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                # print("1111")
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                # print("2222")
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~edit_mask, 1e5) if mask_free else scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        # print("Final", ids.max(), ids.min())
        return ids

    @torch.no_grad()
    @eval_decorator
    def edit_beta(self,
                  conds,
                  conds_og,
                  tokens,
                  m_lens,
                  cond_scale: int,
                  force_mask=False,
                  ):

        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
                if conds_og is not None:
                    cond_vector_og = self.encode_text(conds_og)
                else:
                    cond_vector_og = None
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
            if conds_og is not None:
                cond_vector_og = self.enc_action(conds_og).to(device)
            else:
                cond_vector_og = None
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, tokens)  # Do not mask anything

        '''
        Preparing input
        '''
        # (b, num_token, seqlen)
        logits = self.forward_with_cond_scale(ids,
                                              cond_vector=cond_vector,
                                              cond_vector_neg=cond_vector_og,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)

        '''
        Updating scores
        '''
        probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
        tokens[tokens == -1] = 0  # just to get through an error when index = -1 using gather
        og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1))  # (b, seqlen, 1)
        og_tokens_scores = og_tokens_scores.squeeze(-1)  # (b, seqlen)

        return og_tokens_scores


class DResidualTransformer(nn.Module):
    def __init__(self, code_dim_m, code_dim_d, cond_mode, latent_dim=256, ff_size=1024, num_layers=8, cond_drop_prob=0.1,
                 num_heads=4, dropout=0.1, clip_dim=512, shared_codebook=False, share_weight=False,
                 clip_version=None, opt=None, **kargs):
        super(DResidualTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        # assert shared_codebook == True, "Only support shared codebook right now!"

        self.code_dim_m = code_dim_m
        self.code_dim_d = code_dim_d
        self.code_dim = max(self.code_dim_m, self.code_dim_d) # Max code dimension for compatibility
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.num_motion_tokens = opt.num_motion_tokens  # Number of motion tokens
        self.num_disease_tokens = opt.num_disease_tokens  # Number of disease tokens
        self.num_special_tokens = 2  # pad, motion_end
        self._num_tokens = self.num_motion_tokens + self.num_disease_tokens + self.num_special_tokens
        self.pad_id = opt.num_tokens  # Padding token
        self.motion_end_token_id = opt.num_tokens + 1  # Motion end token

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_classes' in vars(opt).keys(), 'num_classes must be provided for action condition mode'
        self.num_actions = opt.num_classes

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.encode_quant = partial(F.one_hot, num_classes=self.opt.num_quantizers)
        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        self.quant_emb = nn.Linear(self.opt.num_quantizers, self.latent_dim)
        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        else:
            raise KeyError("Unsupported condition mode!!!")

        # Single OutputProcess to handle the entire sequence
        self.output_process = OutputProcess(out_feats=self.code_dim, latent_dim=self.latent_dim)


        if shared_codebook:
            assert shared_codebook==0, "check implementation for the shared codebook"
            # ToDo: check for the shared codebook
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(self._num_tokens, self.code_dim)))
            self.token_embed_weight = token_embed.expand(opt.num_quantizers-1, self._num_tokens, self.code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(self._num_tokens, self.code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(self._num_tokens,)))
                # self.output_proj_bias = 0
                self.output_proj_weight = output_proj.expand(opt.num_quantizers-1, self._num_tokens, self.code_dim)
                self.output_proj_bias = output_bias.expand(opt.num_quantizers-1, self._num_tokens)

        else:
            if share_weight: # Shared weights between quantizers
                # Shared embeddings for middle quantizer layers (opt.num_quantizers - 2)
                self.m_embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, self.num_motion_tokens, self.code_dim)))
                self.d_embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, self.num_disease_tokens, self.code_dim)))
                # Shared token embeddings for motion and disease tokens (used in first and last quantizer layers)
                self.m_token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_motion_tokens, self.code_dim)))
                self.d_token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_disease_tokens, self.code_dim)))
                # Output projection weights for motion and disease tokens
                self.m_output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_motion_tokens, self.code_dim)))
                self.d_output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_disease_tokens, self.code_dim)))
                
                # Embeddings and projection weights for special tokens (motion_end_token and pad_token)
                self.special_embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, self.num_special_tokens, self.code_dim)))
                self.special_token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_special_tokens, self.code_dim)))
                self.special_output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_special_tokens, self.code_dim)))
                

                # self.m_embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, self.num_motion_tokens, self.code_dim)))
                # self.m_token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_motion_tokens, self.code_dim)))
                # self.m_output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_motion_tokens, self.code_dim)))
                
                # self.d_embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, self.num_disease_tokens, self.code_dim)))
                # self.d_token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_disease_tokens, self.code_dim)))
                # self.d_output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.num_disease_tokens, self.code_dim)))
                
                self.output_proj_bias = None
                self.registered = False
            else: # Separate weights for each quantizer
                assert share_weight==1, "check implementation for this part"
                output_proj_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, self._num_tokens, self.code_dim))

                self.output_proj_weight = nn.Parameter(output_proj_weight)
                self.output_proj_bias = nn.Parameter(torch.zeros(size=(opt.num_quantizers, self._num_tokens)))
                token_embed_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, self._num_tokens, self.code_dim))
                self.token_embed_weight = nn.Parameter(token_embed_weight)

        self.apply(self.__init_weights)
        self.shared_codebook = shared_codebook
        self.share_weight = share_weight

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

    # def

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text


    def q_schedule(self, bs, low, high):
        noise = uniform((bs,), device=self.opt.device)
        schedule = 1 - cosine_schedule(noise)
        return torch.round(schedule * (high - low)) + low

    def process_embed_proj_weight(self):
        if self.share_weight and (not self.shared_codebook):
            # if not self.registered:
            self.m_output_proj_weight = torch.cat([self.m_embed_proj_shared_weight, self.m_output_proj_weight_], dim=0)
            self.m_token_embed_weight = torch.cat([self.m_token_embed_weight_, self.m_embed_proj_shared_weight], dim=0)
            self.d_output_proj_weight = torch.cat([self.d_embed_proj_shared_weight, self.d_output_proj_weight_], dim=0)
            self.d_token_embed_weight = torch.cat([self.d_token_embed_weight_, self.d_embed_proj_shared_weight], dim=0)
            self.special_output_proj_weight = torch.cat([self.special_embed_proj_shared_weight, self.special_output_proj_weight_], dim=0)
            self.special_token_embed_weight = torch.cat([self.special_token_embed_weight_, self.special_embed_proj_shared_weight], dim=0)
            
            self.token_embed_weight = torch.cat([self.m_token_embed_weight, self.d_token_embed_weight, self.special_token_embed_weight], dim=1)  # Shape: (1, 5, total_num_tokens, code_dim)
                # self.registered = True
            # self.output_proj_weight = torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)

    def output_project(self, logits, qids, m_lens_m):
        '''
        :logits: (bs, code_dim, seqlen) seqlen: n + 1 + n
        :qids: (bs)
        :param m_lens_m: (bs,) - Lengths of the motion sequences in each batch

        :return:
            -logits (bs, ntoken, seqlen)
        '''
        bs, code_dim, seq_len = logits.shape
        device = logits.device
        
        # Split `logits` into motion, <end>, and disease based on `m_lens_m`
        logits_m = torch.zeros((bs, code_dim, max(m_lens_m)), device=device)
        logits_d = torch.zeros_like(logits_m)
        
        for i in range(bs):
            motion_len = m_lens_m[i]
            logits_m[i, :, :motion_len] = logits[i, :, :motion_len]  # Motion part
            logits_d[i, :, :motion_len] = logits[i, :, motion_len + 1:motion_len + 1 + motion_len]  # Disease part

        # Extract the projection weights for motion and disease
        # (num_qlayers-1, num_token, code_dim) -> (bs, ntoken, code_dim)
        m_output_proj_weight = self.m_output_proj_weight[qids]  # (bs, num_motion_tokens, code_dim)
        d_output_proj_weight = self.d_output_proj_weight[qids]  # (bs, num_disease_tokens, code_dim)
        # (num_qlayers, ntoken) -> (bs, ntoken)
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]
        
        # Apply projection for motion logits
        projected_logits_m = torch.einsum('bnc, bcs->bns', m_output_proj_weight, logits_m) # (bs, num_motion_tokens, n)
        if output_proj_bias is not None:
            projected_logits_m += output_proj_bias.unsqueeze(-1)
        # Apply projection for disease logits
        projected_logits_d = torch.einsum('bnc, bcs->bns', d_output_proj_weight, logits_d) # (bs, num_disease_tokens, n)
        if output_proj_bias is not None:
            projected_logits_d += output_proj_bias.unsqueeze(-1)
            
        # Recombine motion, <end>, and disease projections into one sequence
        output = torch.full((bs, projected_logits_m.shape[1] + projected_logits_d.shape[1], seq_len), float('-1e9'), device=device)
        for i in range(bs):
            motion_len = m_lens_m[i]
            output[i, :projected_logits_m.shape[1], :motion_len] = projected_logits_m[i, :, :motion_len]
            # output[i, :, motion_len] = self.motion_end_token_id # don't need to project motion end token as it will be masked for evaluation and remove during generation
            output[i, projected_logits_m.shape[1]:, motion_len + 1:motion_len + 1 + motion_len] = projected_logits_d[i, :, :motion_len]
        return output, projected_logits_m, projected_logits_d



    def trans_forward(self, motion_codes, qids, cond, padding_mask, force_mask=False):
        '''
        :param motion_codes: (b, seqlen, d) seqlen:n*2+1
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param qids: (b), quantizer layer ids
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :return:
            -logits: (b, num_token, seqlen)
        '''
        cond = self.mask_cond(cond, force_mask=force_mask)

        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(motion_codes)

        # (b, num_quantizer)
        q_onehot = self.encode_quant(qids).float().to(x.device)

        q_emb = self.quant_emb(q_onehot).unsqueeze(0)  # (1, b, latent_dim)
        cond = self.cond_emb(cond).unsqueeze(0)  # (1, b, latent_dim)

        x = self.position_enc(x)
        xseq = torch.cat([cond, q_emb, x], dim=0)  # (seqlen+2, b, latent_dim)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)  # (b, seqlen+2)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[2:]  # (seqlen, b, e)
        logits = self.output_process(output)
        return logits

    def forward_with_cond_scale(self,
                                motion_codes,
                                q_id,
                                cond_vector,
                                padding_mask,
                                m_lens_m,
                                cond_scale=3,
                                force_mask=False):
        bs = motion_codes.shape[0]
        # if cond_scale == 1:
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
        if force_mask:
            logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
            logits, logits_m, logits_d = self.output_project(logits, qids-1, m_lens_m)
            return logits

        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask)
        logits, logits_m, logits_d = self.output_project(logits, qids-1, m_lens_m)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
        aux_logits, aux_logits_m, aux_logits_d = self.output_project(aux_logits, qids-1, m_lens_m)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    def forward(self, all_indices_m, all_indices_d, y, m_lens_m):
        '''
        :param all_indices_m: (b, n, q) Motion token IDs with quantization layers
        :param all_indices_d: (b, n, q) Disease token IDs with quantization layers
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens_m: (b,)
        :return:
        '''

        self.process_embed_proj_weight()

        bs, ntokens, num_quant_layers = all_indices_m.shape
        device = all_indices_m.device
        
        assert torch.all((all_indices_d >= 0) & (all_indices_d < self.num_disease_tokens)), \
           "disease_ids contains values outside the range of the disease codebook." 
        
        # Shift disease token IDs to avoid overlap with motion token IDs
        all_indices_d_shifted = all_indices_d + self.num_motion_tokens
        # Remove padding based on lengths for both motion and disease tokens
        all_indices_m_nonpad = [all_indices_m[i].narrow(0, 0, m_lens_m[i]) for i in range(bs)]
        all_indices_d_nonpad_NOTshifted = [all_indices_d[i].narrow(0, 0, m_lens_m[i]) for i in range(bs)]
        all_indices_d_nonpad_shifted = [all_indices_d_shifted[i].narrow(0, 0, m_lens_m[i]) for i in range(bs)]
        # Create motion end token IDs
        motion_end_token = torch.full((bs, 1, num_quant_layers), self.motion_end_token_id, device=device, dtype=torch.long)
        # Concatenate motion, motion end, and disease tokens for each sample
        ids_list_shifted = [torch.cat([all_indices_m_nonpad[i], motion_end_token[i], all_indices_d_nonpad_shifted[i]], dim=0) for i in range(bs)]
        ids_list_NOTshifted = [torch.cat([all_indices_m_nonpad[i], motion_end_token[i], all_indices_d_nonpad_NOTshifted[i]], dim=0) for i in range(bs)]
    
        # Calculate the total length of each combined sequence
        m_lens = m_lens_m * 2 + 1 # Total length including motion end token
        # Find the maximum sequence length for padding
        ntokens_combined = ntokens * 2 + 1  # e.g., if ntokens is 49, then max sequence length will be 99
        
        # Pad each combined sequence to the maximum sequence length (n + 1 + n)
        ids_list_padded = [F.pad(seq, (0, 0, 0, ntokens_combined - seq.shape[0]), value=self.pad_id) for seq in ids_list_shifted]
        ids_list_padded_NOTshifted = [F.pad(seq, (0, 0, 0, ntokens_combined - seq.shape[0]), value=self.pad_id) for seq in ids_list_NOTshifted]
        all_indices = torch.stack(ids_list_padded, dim=0) # (bs, n + 1 + n, q)
        all_indices_NOTshifted = torch.stack(ids_list_padded_NOTshifted, dim=0) # (bs, n + 1 + n, q)

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = lengths_to_mask(m_lens, ntokens_combined)  # (b, n)

        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id) #(b, n, q)

        # randomly sample quantization layers to work on, [1, num_q)
        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)
        
        active_indices_NOTshifted = all_indices_NOTshifted[torch.arange(bs), :, active_q_layers]  # (b, n)
        active_indices = all_indices[torch.arange(bs), :, active_q_layers]  # (b, n)

        '''Prepare embeddings'''
        
        token_embed = repeat(self.token_embed_weight, 'q c d -> b c d q', b=bs)
        gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
        all_codes = token_embed.gather(1, gather_indices)  # (b, n, d, q-1)
        cumsum_codes = torch.cumsum(all_codes, dim=-1) #(b, n, d, q-1) computes the cumulative sum of the embeddings across the quantizer dimension

        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]
        
        # # Repeat and gather embeddings based on combined sequence `all_indices`
        # token_embed_m = repeat(self.m_token_embed_weight, 'q c d -> b c d q', b=bs)
        # token_embed_d = repeat(self.d_token_embed_weight, 'q c d -> b c d q', b=bs)
        # # Gather embeddings for both motion and disease using combined indices
        # gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed_m.shape[2])
        # motion_gather_indices = torch.clamp(gather_indices, max=self.num_motion_tokens - 1)
        # disease_gather_indices = torch.clamp(gather_indices - self.num_motion_tokens, min=0, max=self.num_disease_tokens - 1)
        # gathered_motion_codes = token_embed_m.gather(1, motion_gather_indices)
        # gathered_disease_codes = token_embed_d.gather(1, disease_gather_indices)
        # all_codes = torch.where(gather_indices < self.num_motion_tokens,  # Use condition to pick motion or disease embeddings
        #                         token_embed_m.gather(1, gather_indices),  # Motion
        #                         token_embed_d.gather(1, gather_indices))  # Disease
        
        # cumsum_codes = torch.cumsum(all_codes, dim=-1) #(b, n, d, q-1)
        # history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]
        
        # # Compute cumulative sum of the embeddings across the quantizer dimension
        # history_sum = torch.cumsum(all_codes, dim=-1)[:, :, :, active_q_layers - 1]  # Shape: (b, n*2+1, d)

        # # Repeat and gather motion embeddings
        # token_embed_m = repeat(self.m_token_embed_weight, 'q c d -> b c d q', b=bs)
        # gather_indices_m = repeat(all_indices_m[..., :-1], 'b n q -> b n d q', d=token_embed_m.shape[2])
        # all_codes_m = token_embed_m.gather(1, gather_indices_m)  # Shape: (b, n, d, q-1)
        # # Repeat and gather disease embeddings
        # token_embed_d = repeat(self.d_token_embed_weight, 'q c d -> b c d q', b=bs)
        # gather_indices_d = repeat(all_indices_d[..., :-1], 'b n q -> b n d q', d=token_embed_d.shape[2])
        # all_codes_d = token_embed_d.gather(1, gather_indices_d)  # Shape: (b, n, d, q-1)
        # # Compute cumulative sums for motion and disease embeddings along the quantizer dimension
        # cumsum_codes_m = torch.cumsum(all_codes_m, dim=-1)  # Shape: (b, n*2+1, d, q-1)
        # cumsum_codes_d = torch.cumsum(all_codes_d, dim=-1)  # Shape: (b, n, d, q-1)
        # Get history sum for motion and disease, selected by active_q_layers - 1
        # history_sum_m = cumsum_codes_m[torch.arange(bs), :, :, active_q_layers - 1]  # Shape: (b, n, d)
        # history_sum_d = cumsum_codes_d[torch.arange(bs), :, :, active_q_layers - 1]  # Shape: (b, n, d)
        # # Prepare the <end> token embedding with the same dimension as other tokens (latent_dim)
        # _, _, d, _ = cumsum_codes_m.shape
        # end_token_embedding = torch.zeros((bs, 1, d), device=history_sum_m.device)
        # # Concatenate motion, end token, and disease into a combined sequence for history_sum
        # history_sum_combined = torch.cat([history_sum_m, end_token_embedding, history_sum_d], dim=1)  # Shape: (b, 2n+1, d)


        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask, force_mask)
        logits, logits_m, logits_d = self.output_project(logits, active_q_layers-1, m_lens_m)
        active_indices_NOTshifted = torch.where(active_indices_NOTshifted == self.motion_end_token_id, self.pad_id, active_indices_NOTshifted) # Replace motion end token with pad token for evaluation
        active_indices = torch.where(active_indices == self.motion_end_token_id, self.pad_id, active_indices) # Replace motion end token with pad token for evaluation
        ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)

        return ce_loss, pred_id, acc

    def separate_ids(self, ids, m_lens_m, m_lens_d):
        """
        Separate the generated ids into motion and disease ids based on their lengths.
        
        :param ids: Combined ids of shape (batch_size, seq_len)
        :param m_lens_m: Tensor of motion sequence lengths (batch_size,)
        :param m_lens_d: Tensor of disease sequence lengths (batch_size,)
        :return: ids_m, ids_d - Separated ids for motion and disease tokens
        """
        device = ids.device
        batch_size, seq_len, q = ids.shape
        
        ids_m_list = []
        ids_d_list = []
        
        for i in range(batch_size):
            motion_len = m_lens_m[i]
            disease_len = m_lens_d[i]
            
            # Extract ids for motion tokens
            ids_m = ids[i, :motion_len, :]
            ids_m_list.append(ids_m)
            
            # Extract ids for disease tokens (after motion end token)
            ids_d = ids[i, motion_len + 1:motion_len + 1 + disease_len, :]
            ids_d_list.append(ids_d)
        
        # Pad the sequences to ensure uniformity
        ids_m = pad_sequence(ids_m_list, batch_first=True, padding_value=self.pad_id)
        ids_d = pad_sequence(ids_d_list, batch_first=True, padding_value=self.pad_id)
        
        return ids_m, ids_d
    
    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 motion_ids,
                 disease_ids,
                 conds,
                 m_lens_m,
                 temperature=1,
                 topk_filter_thres=0.9,
                 cond_scale=2,
                 num_res_layers=-1, # If it's -1, use all.
                 ):
        assert torch.all((disease_ids == -1)|(disease_ids >= 0) & (disease_ids < self.num_disease_tokens)), \
           "disease_ids contains values outside the range of the disease codebook." 

        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1] * 2 + 1 # n + 1 + n
        batch_size = len(conds)
        m_lens = m_lens_m + 1 + m_lens_m

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")
        
        disease_ids_shifted = disease_ids + self.num_motion_tokens
        motion_ids_nonpad = [motion_ids[i].narrow(0, 0, m_lens_m[i]) for i in range(batch_size)]
        disease_ids_nonpad = [disease_ids_shifted[i].narrow(0, 0, m_lens_m[i]) for i in range(batch_size)]
        motion_end_token = torch.full((batch_size, 1), self.motion_end_token_id, device=device, dtype=torch.long)
        
        ids_list = [torch.cat([motion_ids_nonpad[i], motion_end_token[i], disease_ids_nonpad[i]], dim=0) for i in range(batch_size)]
        ids_list = [F.pad(seq, (0, seq_len - len(seq)), value=self.pad_id) for seq in ids_list]
        ids = torch.stack(ids_list, dim=0) # (bs, seq_len)
        
        padding_mask = ~lengths_to_mask(m_lens, seq_len) # (b, seq_len)

        ids = torch.where(padding_mask, self.pad_id, ids)
        motion_end_mask = ids == self.motion_end_token_id
        all_indices = [ids]
        history_sum = 0
        num_quant_layers = self.opt.num_quantizers if num_res_layers==-1 else num_res_layers+1

        for i in range(1, num_quant_layers):
            # print(f"--> Working on {i}-th quantizer")
            # Start from all tokens being masked
            
            token_embed = self.token_embed_weight[i-1]                      # (num_tokens, d)
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)  # (b, num_tokens, d)
            gathered_ids = repeat(ids, 'b n -> b n d', d=token_embed.shape[-1]) # (b, seq_len, d)
            history_sum += token_embed.gather(1, gathered_ids) # (b, seq_len, d)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, m_lens_m, cond_scale=cond_scale)
            # logits = self.trans_forward(history_sum, qids, cond_vector, padding_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            # probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
            # # print(temperature, starting_temperature, steps_until_x0, timesteps)
            # # print(probs / temperature)
            # pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)
            
            ids = torch.where(padding_mask, self.pad_id, pred_ids)
            ids = torch.where(motion_end_mask, self.motion_end_token_id, ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        
        all_indices_m, all_indices_d = self.separate_ids(all_indices, m_lens_m, m_lens_m)
        
        
        # padding_mask = repeat(padding_mask, 'b n -> b n q', q=all_indices.shape[-1])
        # all_indices = torch.where(padding_mask, -1, all_indices)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
        all_indices = torch.where(all_indices==self.motion_end_token_id, -1, all_indices)
        # Update the ids_d with the adjusted disease token indices (removing padding and adjusting token range)
        all_indices_d = torch.where(all_indices_d != self.pad_id, all_indices_d - self.num_motion_tokens, all_indices_d)
        
        assert torch.all(((all_indices_d == self.pad_id) | ((all_indices_d >= 0) & (all_indices_d < self.num_disease_tokens)))), \
    "disease_ids contains values outside the range of the disease codebook."
    
        all_indices_d = torch.where(all_indices_d==self.pad_id, -1, all_indices_d)
        all_indices_m = torch.where(all_indices_m==self.pad_id, -1, all_indices_m)
    
        # all_indices = all_indices.masked_fill()
        return all_indices, all_indices_m, all_indices_d

    @torch.no_grad()
    @eval_decorator
    def edit(self,
            motion_ids,
            conds,
            m_lens,
            temperature=1,
            topk_filter_thres=0.9,
            cond_scale=2
            ):

        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # token_embed = repeat(self.token_embed_weight, 'c d -> b c d', b=batch_size)
        # gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        # history_sum = token_embed.gather(1, gathered_ids)

        # print(pa, seq_len)
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, motion_ids.shape)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0

        for i in range(1, self.opt.num_quantizers):
            # print(f"--> Working on {i}-th quantizer")
            # Start from all tokens being masked
            # qids = torch.full((batch_size,), i, dtype=torch.long, device=motion_ids.device)
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
            # logits = self.trans_forward(history_sum, qids, cond_vector, padding_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            # probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
            # # print(temperature, starting_temperature, steps_until_x0, timesteps)
            # # print(probs / temperature)
            # pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        # padding_mask = repeat(padding_mask, 'b n -> b n q', q=all_indices.shape[-1])
        # all_indices = torch.where(padding_mask, -1, all_indices)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
        # all_indices = all_indices.masked_fill()
        return all_indices