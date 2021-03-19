import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing, vocab_size, ignore_index=0, reduction='sum'):
        assert 0.0 <= label_smoothing <= 1.0
        super(SmoothCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, x, target):
        assert self.vocab_size == x.size(-1)
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)

        ce = self.cross_entropy_with_logits(q_prime, x)
        if self.reduction == 'mean':
            lengths = torch.sum(target != self.ignore_index)
            return ce.sum() / lengths
        elif self.reduction == 'sum':
            return ce.sum()
        else:
            raise NotImplementedError

    @staticmethod
    def cross_entropy_with_logits(p, q):
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)


def vae_loss(x, y, mu, log_var):
    loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    ce = loss_function(x, y)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return ce, kld


def ce_loss(x, y):
    loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    ce = loss_function(x, y)
    return ce
