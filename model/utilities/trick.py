import torch


def timer(start, end):
    h, re = divmod(end - start, 3600)
    m, s = divmod(re, 60)
    return h, m, s


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std
