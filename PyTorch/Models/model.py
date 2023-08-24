import numpy as np
from torch import nn

from Models.Attention import Attention

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Permutation(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)

def model_factory(config):
    if config['Net_Type'][0] == 'T':
        model = Transformer(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'CC-T':
        model = CausalConvTran(config, num_classes=config['num_labels'])
    else:
        model = ConvTran(config, num_classes=config['num_labels'])
    
    return model


class Transformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        print('\n\n\n')
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        print('channel_size:  ', channel_size, '  seq_len:  ', seq_len)
        emb_size = config['emb_size']
        print('emb_size: ', 16)
        num_heads = config['num_heads']
        print('num_heads: ', num_heads)
        dim_ff = config['dim_ff']
        print('dim_ffL: ', dim_ff)
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(
            nn.Linear(channel_size, emb_size),
            nn.LayerNorm(emb_size, eps=1e-5)
        )

        if self.Fix_pos_encode == 'Sin':
            pass
        elif config['Fix_pos_encode'] == 'Learn':
            pass

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.attention_layer = Attention(emb_size, num_heads, config['dropout'])
        
        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)
        print('\n\n\n')


    def forward(self, x):
        print('Inputn shape: ', x.shape)
        x_src = self.embed_layer(x.permute(0, 2, 1))
        print('New shape of input: ', x.permute(0, 2, 1).shape)
        print('Shape of x_src after embedding layer: ', x_src.shape)
        # if self.Fix_pos_encode != None:
        #     x_src = self.Fix_pos_encode(x_src)

        att = x_src + self.attention_layer(x_src)  # residual conn btw input and out of attn
        
        print('Output shape of attention layer: ', att.shape)
        att = self.LayerNorm1(att)  # first layer norm in enc block
        print('Out shape after first Normalization layer: ', att.shape)


        out = att + self.FeedForward(att)  # res conn btw out of attn and ffn
        print('Out shap of self.FeedForward(att): ', self.FeedForward(att).shape)
        out = self.LayerNorm2(out)  # second layer norm in enc block

        print('Output shape of encoder block: ', out.shape)
        out = out.permute(0, 2, 1)
        print('Output shape of encoder block after permutation: ', out.shape)
        
        out = self.gap(out)
        print('Output of gap layer: ', out.shape)
        out = self.flatten(out)
        print('Outut of flatten layer: ', out.shape)
        out = self.out(out)
        print('Final output shape: ', out.shape)
        print('\n\n\n')
        return out



class CausalConvTran(nn.Module):
    pass

class ConvTran(nn.Module):
    pass