from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from attention import AttentionLayer
from attention import PositionalEmbedding
from attention import Embedding


class Transformer(nn.Module):
    def __init__(self, vocab, n_layer, n_head, d_model, d_head, d_inner, dropout):
        super(Transformer, self).__init__()
        self.embedding = Embedding(vocab, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(AttentionLayer(n_head, d_model, d_head, d_inner, dropout))
        self.pe = PositionalEmbedding(d_model)
        self.w_bias = nn.Parameter(torch.randn(n_head, 1, d_head), requires_grad=True)
        self.r_bias = nn.Parameter(torch.randn(n_head, 1, d_head), requires_grad=True)
        self.prob = nn.Sequential(nn.Linear(d_model, vocab))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        embed = self.drop(self.embedding(x))
        q_len = x.size(1)
        positions = torch.arange(q_len - 1, -1, -1.0, device=embed.device, dtype=embed.dtype).repeat(embed.size(0), 1)
        pe = self.drop(self.pe(positions))
        for layer in self.layers:
            embed = layer(embed, pe, self.w_bias, self.r_bias)
        output = self.prob(embed)
        return output.permute(0, 2, 1)

    def decode(self, x, length):
        counter = 1
        for _ in tqdm(range(length)):
            embed = self.embedding(x)
            q_len = x.size(1)
            positions = torch.arange(q_len - 1,
                                     -1,
                                     -1.0,
                                     device=embed.device,
                                     dtype=embed.dtype).repeat(embed.size(0), 1)
            pe = self.pe(positions)
            for layer in self.layers:
                embed = layer(embed, pe, self.w_bias, self.r_bias)
            note = Categorical(F.softmax(self.prob(embed)[:, -1], dim=-1)).sample().unsqueeze(dim=1)
            x = torch.cat((x, note), dim=-1)
            counter += 1
        return x
