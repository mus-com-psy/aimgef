import torch


def timer(start, end):
    h, re = divmod(end - start, 3600)
    m, s = divmod(re, 60)
    return h, m, s
