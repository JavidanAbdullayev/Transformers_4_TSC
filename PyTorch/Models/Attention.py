import torch
import torch.nn as nn
from einops import rearrange
import pandas as pd

class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        print('emb_size: ', emb_size)
        self.key = nn.Linear(1, emb_size, bias=True)
        self.value = nn.Linear(1, emb_size, bias=True)
        self.query = nn.Linear(1, emb_size, bias=True)

        self.to_out = nn.Linear(emb_size, 1)
        # self.to_out = nn.Dropout(dropout)
        # self.to_out = nn.LayerNorm(emb_size)
    def forward(self, x):

        batch_size, seq_len, _ = x.shape
        # print('\n\n\n')
        print('Input shape of attention, x shape: ', x.shape)
        # print('ATTENTION MODULE')
        # print('batch_size: ', batch_size)
        # print('seq_len: ', seq_len)
        # print('_: ', _)
        # print("self.key(x).reshape(batch_size, seq_len, self.num_heads, -1) shape: ",  self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).shape)
        # print("self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1) shape: ",  self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1).shape)
       
        # print('self.value(x) shape: ', self.value(x).shape)
        # print('self.value(x).reshape(batch_size, seq_len, self.num_heads, -1) shape: ', self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).shape)
        # print('self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2) shape: ', self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2).shape)
        # print('\n\n\n')
        
        
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        print('Attention shape: ', attn.shape)
        # print('attn shape: ', attn.shape)
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)  

        # import matplotlib.pyplot as plt
        # plt.plot(x[0, :, 0].detach().cpu().numpy())
        # plt.show()

        out = torch.matmul(attn, v)
        print('out shape: ', out.shape)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        print('out shape  11: ', out.shape)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        print('out shape  22: ', out.shape)

        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        print('Last output shape: ', out.shape)
        return out

class CustomeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        output, _ = self.multihead_attention(x, x, x)
        output = output.permute(1, 0, 2)
        return output