import random

import torch.nn as nn
from models.vq.encdec import Encoder, Decoder, ConditionEncoder, AdaptiveDecoder
from models.vq.residual_vq import ResidualVQ
import torch.nn.functional as F
import torch
    
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        
        self.predictor_opt = {
            'use_classifier': args.w_clsloss > 0,
            'pool_type': args.severity_pool,
        }
        self.classifier = nn.Linear(code_dim, args.num_classes)
        if self.predictor_opt['pool_type'] == 'attention':
            self.cls_selfatt = nn.MultiheadAttention(embed_dim=code_dim, num_heads=4)
        if self.predictor_opt['pool_type'] == 'rnn':
            # self.cls_rnn = nn.LSTM(code_dim, code_dim, batch_first=True)
            num_layers = 1
            self.cls_rnn = nn.GRU(input_size=code_dim, hidden_size=code_dim, num_layers=num_layers, batch_first=True)
        if self.predictor_opt['pool_type'] == 'conv':
            # self.cls_conv = nn.Conv1d(in_channels=code_dim, out_channels=code_dim, kernel_size=3, padding=1)
            self.cls_conv = nn.Conv1d(in_channels=code_dim, out_channels=code_dim, kernel_size=16, stride=1)
            self.cls_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)   #It is not J*3 it is 263 or J*3 representation of joints
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes #(bs, T/4, 6), (6, bs, code_dim, T/4)
    
    def forward_classifier(self, x):
        if self.predictor_opt['pool_type'] == 'mean':
            cls_x = x.mean(dim=2)  # Average over time dimension for classification
        elif self.predictor_opt['pool_type'] == 'max':
            cls_x = x.max(dim=2)[0]
        elif self.predictor_opt['pool_type'] == 'attention':
            x = x.permute(2, 0, 1)  # (T, N, C) required by nn.MultiheadAttention
            attn_output, attn_output_weights = self.cls_selfatt(x, x, x)
            attn_output = attn_output.permute(1, 2, 0)  # (N, C, T) back to original shape
            cls_x = attn_output.mean(dim=2)  # (N, C)
        elif self.predictor_opt['pool_type'] == 'rnn':
            # _, (hidden, _) = self.cls_rnn(x)
            x = x.permute(0, 2, 1)  # (N, T, C)
            gru_output, hidden = self.cls_rnn(x) # hidden has shape (num_layers, N, hidden_dim)
            cls_x = hidden[-1]  # (N, C)
        elif self.predictor_opt['pool_type'] == 'conv':
            # cls_x = self.cls_conv(x.transpose(1, 2)).max(dim=2)[0]
            # x is of shape (N, C, T)
            x = F.relu(self.cls_conv(x))
            x = self.cls_pool(x)  # (N, conv_dim, pool_size)
            # Flatten the pooled output and pass through the classifier
            cls_x = x.squeeze(-1)  # (N, conv_dim * pool_size)
            
        
        classification_logits = self.classifier(cls_x)  # Average over time dimension for classification
        return classification_logits

    def forward(self, x, num_layers=None):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5, num_layers=num_layers)
        
        if self.predictor_opt['use_classifier']:
            classification_logits = self.forward_classifier(x_quantized)
        else:
            classification_logits = None

        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity, classification_logits

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out
    
    def get_codebook_vectors(self):
        return self.quantizer.codebooks
        

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)
    
    
class Conditional_RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        self.num_classes = args.num_classes
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.label_projector = nn.Sequential(
            nn.Linear(args.num_classes, code_dim),
            nn.ReLU(),
        )
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        
        self.predictor_opt = {
            'use_classifier': args.w_clsloss > 0,
            'pool_type': args.severity_pool,
        }
        self.classifier = nn.Linear(code_dim, args.num_classes)
        if self.predictor_opt['pool_type'] == 'attention':
            self.cls_selfatt = nn.MultiheadAttention(embed_dim=code_dim, num_heads=4)
        if self.predictor_opt['pool_type'] == 'rnn':
            # self.cls_rnn = nn.LSTM(code_dim, code_dim, batch_first=True)
            num_layers = 1
            self.cls_rnn = nn.GRU(input_size=code_dim, hidden_size=code_dim, num_layers=num_layers, batch_first=True)
        if self.predictor_opt['pool_type'] == 'conv':
            # self.cls_conv = nn.Conv1d(in_channels=code_dim, out_channels=code_dim, kernel_size=3, padding=1)
            self.cls_conv = nn.Conv1d(in_channels=code_dim, out_channels=code_dim, kernel_size=16, stride=1)
            self.cls_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)   #It is not J*3 it is 263 or J*3 representation of joints
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x, y):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.condition_on_label(x_encoder, y)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes
    
    def condition_on_label(self, z, y, cond_type='add'):
        assert cond_type in ['concat', 'add']
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        projected_label = self.label_projector(y_one_hot).unsqueeze(-1).expand(-1, -1, z.shape[-1])
        if cond_type == 'concat':
            return torch.cat([z, projected_label], dim=1) # Shape: (bs, code_dim + label_embedding_dim, T/4)
        else:
            return z + projected_label     #z: (bs, code_dim, T/4)
    
    def forward_classifier(self, x):
        if self.predictor_opt['pool_type'] == 'mean':
            cls_x = x.mean(dim=2)  # Average over time dimension for classification
        elif self.predictor_opt['pool_type'] == 'max':
            cls_x = x.max(dim=2)[0]
        elif self.predictor_opt['pool_type'] == 'attention':
            x = x.permute(2, 0, 1)  # (T, N, C) required by nn.MultiheadAttention
            attn_output, attn_output_weights = self.cls_selfatt(x, x, x)
            attn_output = attn_output.permute(1, 2, 0)  # (N, C, T) back to original shape
            cls_x = attn_output.mean(dim=2)  # (N, C)
        elif self.predictor_opt['pool_type'] == 'rnn':
            # _, (hidden, _) = self.cls_rnn(x)
            x = x.permute(0, 2, 1)  # (N, T, C)
            gru_output, hidden = self.cls_rnn(x) # hidden has shape (num_layers, N, hidden_dim)
            cls_x = hidden[-1]  # (N, C)
        elif self.predictor_opt['pool_type'] == 'conv':
            # cls_x = self.cls_conv(x.transpose(1, 2)).max(dim=2)[0]
            # x is of shape (N, C, T)
            x = F.relu(self.cls_conv(x))
            x = self.cls_pool(x)  # (N, conv_dim, pool_size)
            # Flatten the pooled output and pass through the classifier
            cls_x = x.squeeze(-1)  # (N, conv_dim * pool_size)
            
        
        classification_logits = self.classifier(cls_x)  # Average over time dimension for classification
        return classification_logits

    def forward(self, x, y, num_layers=None):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in) # (bs, code_dim, T/4)
        
        x_encoder = self.condition_on_label(x_encoder, y)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5, num_layers=num_layers)
        if self.predictor_opt['use_classifier']:
            classification_logits = self.forward_classifier(x_quantized)
        else:
            classification_logits = None

        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity, classification_logits

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out
    
    def get_codebook_vectors(self):
        return self.quantizer.codebooks
    
    
class Disentangled_RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code_m=512,
                 nb_code_d=512,
                 code_dim_m=256,
                 code_dim_d=256,
                 output_emb_width=512,
                 down_t_motion=3,  # Temporal reduction for motion (e.g., T/4)
                 down_t_disease=4,  # Temporal reduction for disease (e.g., T/8)
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        self.use_style = True if args.style else False
        if not self.use_style and not args.mdcombine == 'add':
            assert output_emb_width == (code_dim_m * 2), "Output embedding width should be equal to the sum of motion and disease code dimensions"
        else:
            assert output_emb_width == (code_dim_m), "Output embedding width should be equal to the motion code dimensions"
        assert args.mdcombine in ['conc', 'add'], "Invalid motion-disease combination type"
        
        self.code_dim_m = code_dim_m
        self.code_dim_d = code_dim_d
        self.num_code_motion = nb_code_m
        self.num_code_disease = nb_code_d
        self.mdcombine = args.mdcombine   
        self.disease_dropprob = args.disease_dropprob   
        self.Healthyzeroout = args.Healthyzeroout  
        self.addfactor = args.addfactor

        # Two separate encoders for motion and disease
        self.motion_encoder = Encoder(input_width, code_dim_m, down_t_motion, stride_t, width, depth,
                                      dilation_growth_rate, activation=activation, norm=norm)
        if not args.conditional:
            self.disease_encoder = Encoder(input_width, code_dim_d, down_t_disease, stride_t, width, depth,
                                        dilation_growth_rate, activation=activation, norm=norm)
        else:
            self.disease_encoder = ConditionEncoder(input_width, code_dim_d, down_t_disease, stride_t, width, depth,
                                                dilation_growth_rate, activation=activation, norm=norm, num_classes=args.num_classes)
            
        if code_dim_d != code_dim_m:
            self.disease_fc = nn.Linear(code_dim_d, code_dim_m)
        
        # Separate quantizers for motion and disease factors
        rvqvae_config_motion = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code_m,
            'code_dim': code_dim_m,
            'args': args,
        }
        self.motion_quantizer = ResidualVQ(**rvqvae_config_motion)

        rvqvae_config_disease = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code_d,
            'code_dim': code_dim_d,
            'args': args,
        }
        self.disease_quantizer = ResidualVQ(**rvqvae_config_disease)
        
        # Cross-attention layer for combining motion and disease embeddings
        self.cross_attention = nn.MultiheadAttention(embed_dim=code_dim_m, num_heads=4)
        
        if self.use_style:
            self.style_dim = code_dim_d
            self.decoder = AdaptiveDecoder(input_width, output_emb_width, down_t_motion, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, style_dim=self.style_dim)
        else:
            # Decoder for the combined latent space (motion + disease)
            self.decoder = Decoder(input_width, output_emb_width, down_t_motion, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)   #It is not J*3 it is 263 or J*3 representation of joints
        x = x.permute(0, 2, 1).float()
        return x
    
    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x
    
    def zero_out_disease_latent(self, d_quantized, severity_labels):
        if not self.Healthyzeroout:
            return d_quantized
        mask = (severity_labels == 0).unsqueeze(1).unsqueeze(2).float()  # Shape: (batch_size, 1, 1)
        d_quantized = d_quantized * (1 - mask)
        return d_quantized
    
    def zero_out_disease_latent_randomly(self, d_quantized):
        if not self.training or self.disease_dropprob == 0.0:
            return d_quantized
        batch_size, _, _ = d_quantized.size()
        mask = (torch.rand(batch_size, 1, 1, device=d_quantized.device) > self.disease_dropprob).float()
        d_quantized = d_quantized * mask
        return d_quantized

    def encode(self, x, y=None):
        x_in = self.preprocess(x)
        motion_encoded = self.motion_encoder(x_in)
        disease_encoded = self.disease_encoder(x_in, y)
        m_code_idx, m_all_codes = self.motion_quantizer.quantize(motion_encoded, return_latent=True)
        d_code_idx, d_all_codes = self.disease_quantizer.quantize(disease_encoded, return_latent=True)
        m_quantized, m_codes, m_commit_loss, m_perplexity = self.motion_quantizer(motion_encoded, sample_codebook_temp=0.5, num_layers=None)
        d_quantized, d_codes, d_commit_loss, d_perplexity = self.disease_quantizer(disease_encoded, sample_codebook_temp=0.5, num_layers=None)
        return m_code_idx, d_code_idx, m_all_codes, d_all_codes, m_quantized, d_quantized
    
    def forward_motion_only(self, x, num_layers=None):
        x_in = self.preprocess(x)
        # Encode motion
        m_latent = self.motion_encoder(x_in)
        m_quantized, m_codes, m_commit_loss, m_perplexity = self.motion_quantizer(m_latent, sample_codebook_temp=0.5, num_layers=num_layers)
        # Decode
        x_out = self.decoder(m_quantized)
        return x_out, m_commit_loss, m_perplexity, m_quantized

    def forward(self, x, y=None, num_layers=None):
        # Preprocess input
        x_in = self.preprocess(x)

        # Encode motion
        m_latent = self.motion_encoder(x_in) # Shape: (B, code_dim_m, T/4)
        m_quantized, m_codes, m_commit_loss, m_perplexity = self.motion_quantizer(m_latent, sample_codebook_temp=0.5, num_layers=num_layers) # m_quantized: (B, code_dim_m, T/4) - m_codes: (B, T/4, 6)
        # Encode disease
        d_latent = self.disease_encoder(x_in, y)  # Shape: (B, code_dim_d, T/4)
        d_quantized, d_codes, d_commit_loss, d_perplexity = self.disease_quantizer(d_latent, sample_codebook_temp=0.5, num_layers=num_layers)
            

        # Apply cross-attention to align the embeddings
        # Shape: (T/4, B, code_dim_m) - Shape: (T/4, B, code_dim_d)
        # ToDo: Cannot handle different temporal down_sampling and codebook dim for now
        # attn_output, _ = self.cross_attention(m_quantized.permute(2, 0, 1),           
        #                                             d_quantized.permute(2, 0, 1), 
        #                                             d_quantized.permute(2, 0, 1))
        # print('attn_output', attn_output.shape) # (T/4, B, code_dim_m)

        # Concatenate aligned disease embedding to motion embedding
        # attn_output = attn_output.permute(1, 2, 0)  # Shape: (B, code_dim_m, T/4)
        # combined_embedding = torch.cat([m_quantized, attn_output], dim=1)
        if not self.use_style:
            if self.code_dim_d != self.code_dim_m:
                d_quantized = self.disease_fc(d_quantized.permute(0, 2, 1))  # Now disease has the same dimensionality as motion
                d_quantized = d_quantized.permute(0, 2, 1).contiguous()
                
            # Zero out disease latent for healthy samples
            d_quantized = self.zero_out_disease_latent(d_quantized, y)
            # Randomly zero out disease latent during training
            d_quantized = self.zero_out_disease_latent_randomly(d_quantized)
            
            if self.mdcombine == 'conc':
                combined_embedding = torch.cat([m_quantized, d_quantized], dim=1)
            else:
                combined_embedding = m_quantized + self.addfactor * d_quantized
            # Decode combined embeddings to reconstruct motion
            x_out = self.decoder(combined_embedding)
        else:
            x_out = self.decoder(m_quantized, style=d_quantized) 
            if self.code_dim_d != self.code_dim_m:
                d_quantized = self.disease_fc(d_quantized.permute(0, 2, 1))  # Just for classifiers and comparison
                d_quantized = d_quantized.permute(0, 2, 1).contiguous()                 
        
        total_perplexity = (m_perplexity + d_perplexity) / 2  # Average perplexity for both
        return x_out, m_commit_loss, d_commit_loss, total_perplexity, m_quantized, d_quantized
    
    
    def forward_decoder(self, motion_idx, disease_idx, y):
        # Retrieve latent vectors from quantized indices
        motion_codes = self.motion_quantizer.get_codes_from_indices(motion_idx)
        disease_codes = self.disease_quantizer.get_codes_from_indices(disease_idx)
        
        motion_codes = motion_codes.sum(dim=0).permute(0, 2, 1)
        disease_codes = disease_codes.sum(dim=0).permute(0, 2, 1)
        
        if not self.use_style:
            # Align dimensions for each residual layer output
            if self.code_dim_d != self.code_dim_m:
                disease_codes = self.disease_fc(disease_codes.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
            # only do Zero out disease latent for healthy samples if flag true
            disease_codes = self.zero_out_disease_latent(disease_codes, y)
            if self.mdcombine == 'conc':
                combined_latent = torch.cat([motion_codes, disease_codes], dim=1)
            else:
                combined_latent = motion_codes + self.addfactor * disease_codes
            # Decode the combined latent space to reconstruct the motion
            x_out = self.decoder(combined_latent)
        else:
            x_out = self.decoder(motion_codes, style=disease_codes)  
        return x_out
    
    def forward_decoder_MM(self, motion_idx, disease_idx, y, ctype='md'):
        # Retrieve latent vectors from quantized indices
        if ctype=='md':
            motion_codes = self.motion_quantizer.get_codes_from_indices(motion_idx)
            disease_codes = self.disease_quantizer.get_codes_from_indices(disease_idx)
        elif ctype=='mm':
            motion_codes = self.motion_quantizer.get_codes_from_indices(motion_idx)
            disease_codes = self.motion_quantizer.get_codes_from_indices(disease_idx)
        elif ctype=='dd':
            motion_codes = self.disease_quantizer.get_codes_from_indices(motion_idx)
            disease_codes = self.disease_quantizer.get_codes_from_indices(disease_idx)
        
        motion_codes = motion_codes.sum(dim=0).permute(0, 2, 1)
        disease_codes = disease_codes.sum(dim=0).permute(0, 2, 1)
        
        if not self.use_style:
            # Align dimensions for each residual layer output
            if self.code_dim_d != self.code_dim_m and ctype!='mm':
                disease_codes = self.disease_fc(disease_codes.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
                if ctype=='dd':
                    motion_codes = self.disease_fc(motion_codes.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
            # only do Zero out disease latent for healthy samples if flag true
            disease_codes = self.zero_out_disease_latent(disease_codes, y)
            if self.mdcombine == 'conc':
                combined_latent = torch.cat([motion_codes, disease_codes], dim=1)
            else:
                combined_latent = motion_codes + self.addfactor * disease_codes
            # Decode the combined latent space to reconstruct the motion
            x_out = self.decoder(combined_latent)
        else:
            x_out = self.decoder(motion_codes, style=disease_codes)  
        return x_out
    
    def get_codebook_vectors(self):
        return self.motion_quantizer.codebooks, self.disease_quantizer.codebooks
    

class SeverityPredictor(nn.Module):
    def __init__(self, code_dim, num_classes, pool_type, hidden_dim=128):
        super().__init__()
        self.pool_type = pool_type
        if pool_type == 'attention':
            self.cls_selfatt = nn.MultiheadAttention(embed_dim=code_dim, num_heads=4)
        if pool_type == 'rnn':
            num_layers = 1
            self.cls_rnn = nn.GRU(input_size=code_dim, hidden_size=code_dim, num_layers=num_layers, batch_first=True)
        if pool_type == 'conv':
            self.cls_conv = nn.Conv1d(in_channels=code_dim, out_channels=code_dim, kernel_size=16, stride=1)
            self.cls_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        self.model = nn.Linear(code_dim, num_classes)   
        # self.fc1 = nn.Linear(code_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is of shape (N, C, T)
        if self.pool_type == 'mean':
            cls_x = x.mean(dim=2)  # Average over time dimension
        elif self.pool_type == 'max':
            cls_x = x.max(dim=2)[0]
        elif self.pool_type == 'attention':
            x = x.permute(2, 0, 1)  # (T, N, C) required by nn.MultiheadAttention
            attn_output, attn_output_weights = self.cls_selfatt(x, x, x)
            attn_output = attn_output.permute(1, 2, 0)  # (N, C, T) back to original shape
            cls_x = attn_output.mean(dim=2)  # (N, C)
        elif self.pool_type == 'rnn':
            x = x.permute(0, 2, 1)  # (N, T, C)
            gru_output, hidden = self.cls_rnn(x) # hidden has shape (num_layers, N, hidden_dim)
            cls_x = hidden[-1]  # (N, C)
        elif self.pool_type == 'conv':
            x = F.relu(self.cls_conv(x))
            x = self.cls_pool(x)  # (N, conv_dim, pool_size)
            cls_x = x.squeeze(-1)  # Flatten the pooled output: (N, conv_dim * pool_size)
        elif self.pool_type == 'flatten':
            cls_x = x.view(x.size(0), -1)
        
        return self.model(cls_x)
        # cls_x = F.relu(self.fc1(cls_x))
        # output = self.fc2(cls_x)
        # return output
    
       
class Discriminator(nn.Module):
    def __init__(self, code_dim, num_classes, pool_type, window_size=16, down_t=2):
        super().__init__()
        self.pool_type = pool_type
        if pool_type == 'attention':
            self.att = nn.MultiheadAttention(embed_dim=code_dim, num_heads=4)
        if self.pool_type == 'mean' or pool_type == 'attention':
            inp_dim = code_dim
        else:
            inp_dim = code_dim * window_size // down_t ** 2
            
        self.model = nn.Linear(inp_dim, 1)

    def forward(self, x):
        if self.pool_type == 'attention':
            x = x.permute(2, 0, 1)  # (T, N, C) required by nn.MultiheadAttention
            attn_output, _ = self.att(x, x, x)
            attn_output = attn_output.permute(1, 2, 0)  # (N, C, T) back to original shape
            latent_vector = attn_output.mean(dim=2)  # (N, C)
        elif self.pool_type == 'mean':
            latent_vector = x.mean(dim=2)  # Average over time dimension
        elif self.pool_type == 'flatten':
            latent_vector = x.view(x.size(0), -1)
        return self.model(latent_vector)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output  # Inverts the gradient
    
def grad_reverse(x):
    return GradientReversalFunction.apply(x)