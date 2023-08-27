import numpy as np
from torch import nn

from Models.Attention import Attention, CustomeAttention

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
        # print('\n\n\n')
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        # print('channel_size:  ', channel_size, '  seq_len:  ', seq_len)
        emb_size = config['emb_size']
        # print('emb_size: ', 16)
        num_heads = config['num_heads']
        # print('num_heads: ', num_heads)
        dim_ff = config['dim_ff']
        # print('dim_ffL: ', dim_ff)
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

        self.LayerNorm1 = nn.LayerNorm(1, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(1, eps=1e-5)

        self.attention_layer = Attention(emb_size, num_heads, config['dropout'])
        self.custom_attention_layer = CustomeAttention(emb_size, num_heads)
        
        self.feed_norm = nn.LayerNorm(1, eps=1e-5)
        self.feed_conv1d_1 = nn.Conv1d(in_channels=1, out_channels=dim_ff, kernel_size=1,  stride=1, padding='same')
        self.feed_drop = nn.Dropout(0)
        self.feed_conv1d_2 = nn.Conv1d(in_channels=dim_ff, out_channels=1, kernel_size=1,  stride=1, padding='same')

        # self.FeedForward = nn.Sequential(
        #     nn.LayerNorm(1, eps=1e-5),
        #     nn.Conv1d(in_channels=251, out_channels=dim_ff, kernel_size=1,  stride=1, padding='same'),
        #     nn.Dropout(0),
        #     nn.Conv1d(in_channels=dim_ff, out_channels=1, kernel_size=1,  stride=1, padding='same'),
        # )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(251, 128,)
        self.relu = nn.ReLU()   
        self.dropout = nn.Dropout(0)
        self.out = nn.Linear(128, num_classes)
        # print('\n\n\n')

    def forward(self, x):
        # print('Inputn shape: ', x.shape)
        # x_src = self.embed_layer(x.permute(0, 2, 1))
        # print('Input shape before embedding: ', x.shape)
        # x = self.embed_layer(x.permute(0, 2, 1))

        x = x.permute(0, 2, 1)
        print('Encode shape: ', x.shape)
        x = self.LayerNorm1(x)
        # print('Input shape 111: ', x.shape)
        # print('Original x shape: ', x.shape)
        # print('x_src shape: ', x_src.shape)
        # print('New shape of input: ', x.permute(0, 2, 1).shape)
        # print('Shape of x_src after embedding layer: ', x_src.shape)
        # if self.Fix_pos_encode != None:
        #     x_src = self.Fix_pos_encode(x_src)
        # x = self.LayerNorm1(x)
        print('Input shape of Attention PyTorch: ', x.shape)
        
        att = x + self.attention_layer(x)  # residual conn btw input and out of attn
        
        print('Output of MHA: PyTorch: ', att.shape)

        att = self.dropout(att)
        # print('Output shape of attention layer: ', att.shape)
        # print('Output shape of attention layer: ', att.shape)
        # att = self.LayerNorm1(att)  # first layer norm in enc block
        # print('Out shape after first Normalization layer: ', att.shape)

        print('res shape before norm: ', att.shape)
        feed_norm = self.feed_norm(att)
        print('feed_norm shape: ', feed_norm.shape)
        print('permuted feed_norm shape: ', feed_norm.permute(0, 2, 1).shape)

        feed_conv1d_1 = self.feed_conv1d_1(feed_norm.permute(0, 2, 1))
        print('feed_conv1d_1 shape: ', feed_conv1d_1.shape)

        feed_drop = self.feed_drop (feed_conv1d_1)
        feed_conv1d_2 = self.feed_conv1d_2(feed_drop)
        print('feed_conv1d_2 shape: ', feed_conv1d_2.shape)
        print('attention shape: ', att.shape)

        out = att + feed_conv1d_2.permute(0, 2, 1)  # res conn btw out of attn and ffn
        # print('Out shap of self.FeedForward(att): ', out.shape)
        # out = self.LayerNorm2(out)  # second layer norm in enc block

        # print('Output shape of encoder block: ', out.shape)
        # out = out.permute(0, 2, 1)
        # print('Output shape of encoder block after permutation: ', out.shape)
        
        out = self.gap(out)
        # print('Output of gap layer: ', out.shape)
        out = self.flatten(out)
        # print('Outut of flatten layer: ', out.shape)
        out = self.relu(self.dense1(out))
        out = self.dropout(out)
        out = self.out(out)
        # print('Final output shape: ', out.shape)
        # print('\n\n\n')
        return out



class CausalConvTran(nn.Module):
    pass

class ConvTran(nn.Module):
    pass