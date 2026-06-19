import torch.nn as nn
from models.vq.resnet import Resnet1D
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x, *args):
        return self.model(x)
    
class ConditionEncoder(nn.Module):
    def __init__(self,
                 input_width,
                 code_dim,
                 down_t,
                 stride_t,
                 width,
                 depth,
                 dilation_growth_rate,
                 activation='relu',
                 norm=None,
                 num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = Encoder(input_width, code_dim, down_t, stride_t, width, depth,
                                       dilation_growth_rate, activation=activation, norm=norm)
        self.label_projector = nn.Sequential(
            nn.Linear(num_classes, code_dim),  
            nn.ReLU()
        )
        
    def condition_on_label(self, z, y, cond_type='add'):
        assert cond_type in ['concat', 'add']
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        projected_label = self.label_projector(y_one_hot).unsqueeze(-1).expand(-1, -1, z.shape[-1])
        if cond_type == 'concat':
            return torch.cat([z, projected_label], dim=1) # Shape: (bs, code_dim + label_embedding_dim, T/4)
        else:
            return z + projected_label     #z: (bs, code_dim, T/4)
        
    def forward(self, x, y):
        # Encode the input motion (no conditioning yet)
        x_latent = self.encoder(x)
        conditioned_latent = self.condition_on_label(x_latent, y)
        return conditioned_latent
    


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)
    
    
class AdaptiveLayerNorm1d(nn.Module):
    def __init__(self, num_features, style_dim, style_tdependent=0, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.style_dim = style_dim
        self.style_tdependent = style_tdependent
        # Linear layers to compute gamma and beta from style vector
        if not style_tdependent:
            self.gamma = nn.Linear(style_dim, num_features)
            self.beta = nn.Linear(style_dim, num_features)
        else:
            # Change Linear layers to Conv1d to handle time dimension
            self.gamma = nn.Conv1d(style_dim, num_features, kernel_size=1)
            self.beta = nn.Conv1d(style_dim, num_features, kernel_size=1)
        # Initialize gamma to ones and beta to zeros
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, style):
        # x: (N, C, T)
        # style: (N, style_dim): time independent style vector
        # style: (N, style_dim, T): time dependent style vector
        N, C, T = x.size()
        # Compute layer norm
        if not self.style_tdependent:
            dim = [1, 2] # mean and std (N, 1, 1)
        else:
            dim = 1 # mean and std (N, 1, T)
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True) + self.eps
        x_norm = (x - mean) / std   # (N, C, T)
        # Compute adaptive parameters
        gamma = self.gamma(style)  # time dependent:(N, C, T) - time independent (N, C)
        beta = self.beta(style)    # time dependent:(N, C, T) - time independent (N, C)
        if not self.style_tdependent:
            gamma = gamma.unsqueeze(2)  # (N, C, 1)
            beta = beta.unsqueeze(2)    # (N, C, 1)
        # Apply adaptive layer norm
        out = gamma * x_norm + beta
        return out
    
class AdaptiveDecoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 style_dim=64,
                 style_tdependent=0):
        super().__init__()
        self.style_dim = style_dim
        self.style_tdependent = style_tdependent
        if not style_tdependent:
            self.style_processor = StyleProcessor(style_dim)
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(AdaptiveLayerNorm1d(width, style_dim, style_tdependent))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=None),  # Remove norm here
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1),
                AdaptiveLayerNorm1d(out_dim, style_dim, style_tdependent),  # Apply AdaLN after Conv1d
                nn.ReLU()
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(AdaptiveLayerNorm1d(width, style_dim, style_tdependent))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.ModuleList(blocks)

    def forward(self, x, style):
        # style: (N, style_dim, T)
        if not self.style_tdependent:
            style = self.style_processor(style)  # (N, style_dim)
        for layer in self.model:
            if isinstance(layer, AdaptiveLayerNorm1d):
                x = layer(x, style)
            elif isinstance(layer, nn.Sequential):
                # For the block that contains layers
                for sublayer in layer:
                    if isinstance(sublayer, AdaptiveLayerNorm1d):
                        x = sublayer(x, style)
                    else:
                        x = sublayer(x)
            else:
                x = layer(x)
        return x.permute(0, 2, 1)
    
    
class StyleProcessor(nn.Module):
    def __init__(self, code_dim_d, num_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=code_dim_d, num_heads=num_heads)
        # Optional: LayerNorm and activation
        self.layer_norm = nn.LayerNorm(code_dim_d)
        self.activation = nn.ReLU()

    def forward(self, d_quantized):
        # d_quantized: (N, code_dim_d, T_d)
        d_quantized = d_quantized.permute(2, 0, 1)
        attn_output, _ = self.self_attn(d_quantized, d_quantized, d_quantized)
        # Take the mean over the sequence length (T_d)
        style = attn_output.mean(dim=0)  # (N, code_dim_d)
        style = self.layer_norm(style)
        style = self.activation(style)
        return style