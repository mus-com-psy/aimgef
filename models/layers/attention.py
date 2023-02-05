import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocab, d):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.d = d

    def forward(self, event):
        embedding = self.embedding(event) * (self.d ** 0.5)
        return embedding


class PositionalEmbedding(nn.Module):
    def __init__(self, d_emb):
        super(PositionalEmbedding, self).__init__()
        self.d_emb = d_emb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_emb, 2.0) / d_emb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid = torch.einsum('bi,j->bij', [pos_seq, self.inv_freq])
        pe = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return pe


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.layer_norm(x + self.net(x))
        return output


class AttentionLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout):
        super(AttentionLayer, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.pff = PositionwiseFeedForward(d_model, d_inner, dropout)
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / (d_head ** 0.5)

    @staticmethod
    def shift(x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        ones = torch.ones((x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    def forward(self, w, r, w_bias, r_bias):
        r = self.r_net(r)
        q = self.q_net(w)
        k = self.k_net(w)
        v = self.v_net(w)

        batch_size = w.size(0)
        q_len = w.size(1)
        k_len = k.size(1)
        r_len = r.size(1)
        q = q.view(batch_size, q_len, self.n_head, self.d_head).transpose(1, 2).contiguous()
        k = k.view(batch_size, k_len, self.n_head, self.d_head).transpose(1, 2).contiguous()
        v = v.view(batch_size, k_len, self.n_head, self.d_head).transpose(1, 2).contiguous()
        r = r.view(batch_size, r_len, self.n_head, self.d_head).transpose(1, 2).contiguous()

        rw_q = q + w_bias
        ac = torch.einsum('bnid,bnjd->bnij', [rw_q, k])

        rr_q = q + r_bias
        bd = torch.einsum('bnid,bnjd->bnij', [rr_q, r])
        bd = self.shift(bd)

        attn_score = (ac + bd) * self.scale

        mask = torch.triu(torch.ones((q_len, k_len), device=w.device), diagonal=1).bool()
        attn_score.masked_fill_(mask[None, None, :, :], -1e9)

        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.drop(attn_prob)
        attn_vec = torch.einsum('bnij,bnjd->bnid', [attn_prob, v])

        attn_vec = attn_vec.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        output = self.pff(self.layer_norm(w + attn_out))

        return output
