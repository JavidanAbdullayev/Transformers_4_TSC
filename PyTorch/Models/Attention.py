import torch
import torch.nn as nn
from einops import rearrange
import pandas as pd

class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5   

        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        print('\n\n Attention start')
        batch_size, seq_len, _ = x.shape
        print('x shape in attention: ', x.shape)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        print('Output shape of self.key(x):  ', self.key(x).shape)
        print('Output shape of self.key(x).reshape(batch_size, seq_len, self.num_heads, -1):  ', self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).shape)
        print('Output shape of self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1):  ', self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1).shape)

        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        print('\nOutput shape of self.value(x):  ', self.value(x).shape)
        print('Output shape of self.value(x).reshape(batch_size, seq_len, self.num_heads, -1):  ', self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).shape)
        print('Output shape of self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2):  ', self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2).shape)


        # print('Output shape of value:  ', self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2).shape)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        print('\nOutput shape of self.query(x):  ', self.query(x).shape)
        print('Output shape of self.query(x).reshape(batch_size, seq_len, self.num_heads, -1):  ', self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).shape)
        print('Output shape of self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2):  ', self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2).shape)

        attn = torch.matmul(q, k) * self.scale
        print('\ntorch.matmul(q, k) shape: ', torch.matmul(q, k).shape)
        attn = nn.functional.softmax(attn,dim=-1)
        print('attn shape after softmax: ', attn.shape)

        out = torch.matmul(attn, v)
        print('out shape: ', out.shape)
        out = out.transpose(1, 2)
        print('out after transpose: ', out.shape)
        out = out.reshape(batch_size, seq_len, -1)
        print('out after reshape: ', out.shape)
        out = self.to_out(out)
        print('\n\n Attention end shape: ', out.shape)

        return out